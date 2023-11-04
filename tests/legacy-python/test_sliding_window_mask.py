# fmt: off
"""For testing `_make_sliding_window_mask` in mistral.py"""

import unittest

import numpy as np
import tvm
from mlc_llm.relax_model.mistral import _make_sliding_window_mask
from tvm import relax
from tvm.runtime import ShapeTuple


def _create_vm():
    # pylint: disable=too-many-locals
    bb = relax.BlockBuilder()

    # Step 1: Build `_make_sliding_window_mask()` into an IRModule
    bsz = tvm.tir.Var("bsz", "int64")
    seq_length = tvm.tir.Var("seq_length", "int64")  # tgt_len
    kv_seq_len = tvm.tir.Var("kv_seq_len", "int64")
    sliding_window = tvm.tir.Var("sliding_window", "int64")

    with bb.function("main"):
        # Convert to relax.Var because params to an IRModule function needs to be relax.Var
        bsz_shape = relax.Var("bsz", relax.ShapeStructInfo((bsz,)))
        seq_length_shape = relax.Var("seq_length", relax.ShapeStructInfo((seq_length,)))
        kv_seq_len_shape = relax.Var("kv_seq_len", relax.ShapeStructInfo((kv_seq_len,)))
        sliding_window_shape = relax.Var("sliding_window", relax.ShapeStructInfo((sliding_window,)))

        # Convert back to tir.Var since `_prepare_sliding_window_mask` needs it to be tir.Var
        with bb.dataflow():
            bsz_input = bsz_shape.struct_info.values[0]
            seq_length_input = seq_length_shape.struct_info.values[0]
            kv_seq_len_input = kv_seq_len_shape.struct_info.values[0]
            sliding_window_input = sliding_window_shape.struct_info.values[0]
            mask = _make_sliding_window_mask(
                (bsz_input, seq_length_input),
                kv_seq_len_input,
                sliding_window_input,
                "float32",
            )
            params = [
                bsz_shape,
                seq_length_shape,
                kv_seq_len_shape,
                sliding_window_shape,
            ]
            gv = bb.emit_output(mask)
        bb.emit_func_output(gv, params)

    # Step 2. Optimize IRModule
    mod = bb.get()
    mod = relax.pipeline.get_pipeline()(mod)  # pylint: disable=no-value-for-parameter
    with tvm.target.Target("cuda"):
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    # Step 3. Deploy to GPU
    ex = relax.build(mod, "cuda")
    vm = relax.VirtualMachine(ex, tvm.cuda())  #pylint: disable=redefined-outer-name
    return vm


vm = _create_vm()

class SlidingWindowMaskTest(unittest.TestCase):
    """
    The sliding window mask is based on figure 3 of the Mistral paper.
    There are three cases when making a mask: first prefill, subsequent prefill,
    and decoding.

    1. First Prefill
    This is when the cache is empty (i.e. kv_seq_len == 0). If tgt_len <= sliding_window,
    this is just a normal causal mask. Otherwise, e.g. tgt_len = 3, WS = 2, we create a
    mask below:
    1, 0, 0
    1, 1, 0
    0, 1, 1

    2. Subsequent Prefill
    This is when the cache is not empty and yet tgt_len > 1.
    e.g. t0-t4 in cache; current input is t5-t7; WS=5
        0, 1, 2, 3, 4, | 5, 6, 7
        
        0, 1, 1, 1, 1, | 1, 0, 0
        0, 0, 1, 1, 1, | 1, 1, 0
        0, 0, 0, 1, 1, | 1, 1, 1
          [in cache]    [current]

    3. Decode
    It will always be ones with shape (1 + kv_seq_len) since cache_size equals sliding_window.
    Note that a prefilling (first or subsequent) with chunk_size of 1 is equivalent to a decode
    in mask making.
    """

    ################### 1. TESTS FOR FIRST PREFILL ###################
    def test_first_prefill_chunk_size_smaller_than_ws(self):
        """
        When chunk size < WS, we return a normal causal mask.
        Here, chunk size 3, WS 5.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([3])  # chunk size is 3
        kv_seq_len = ShapeTuple([3])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
            [3.402823e38, -3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, 3.402823e38],
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_first_prefill_chunk_size_equals_ws(self):
        """
        When chunk_size == WS, we also return a normal causal mask.
        Here both chunk size and WS are 5.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([5])
        kv_seq_len = ShapeTuple([5])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
            [3.402823e38, -3.402823e38, -3.402823e38, -3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, -3.402823e38, -3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, 3.402823e38, -3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, 3.402823e38, 3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, 3.402823e38, 3.402823e38, 3.402823e38],
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_first_prefill_chunk_size_greater_than_ws(self):
        """
        When chunk_size > WS, return a normal causal mask but each row only has at most WS 1's.
        Here chunk_size = 5, WS=3.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([5])
        kv_seq_len = ShapeTuple([5])
        sliding_window = ShapeTuple([3])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
            [3.402823e38, -3.402823e38, -3.402823e38, -3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, -3.402823e38, -3.402823e38, -3.402823e38],
            [3.402823e38, 3.402823e38, 3.402823e38, -3.402823e38, -3.402823e38],
            [-3.402823e38, 3.402823e38, 3.402823e38, 3.402823e38, -3.402823e38],
            [-3.402823e38, -3.402823e38, 3.402823e38, 3.402823e38, 3.402823e38],
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_first_prefill_chunk_size_one(self):
        """
        Corner case: the prompt only has 1 token.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([1])
        kv_seq_len = ShapeTuple([1])
        sliding_window = ShapeTuple([3])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
            [3.402823e38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    ################### 2. TESTS FOR SUBSEQUENT PREFILL ###################
    def test_subsequent_prefill_1(self):
        """
        Test 1: chunk size is 3, WS is 5, cache carrying t0, t1, t2; input t3, t4, t5.
        """

        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([3])
        kv_seq_len = ShapeTuple([6])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
        # pylint: disable=line-too-long
        #   |                 IN CACHE                   |             CURRENT CHUNK                |
        #          t0              t1             t2             t3           t4             t5
            [ 3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38],
            [ 3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38],
            [-3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_subsequent_prefill_2(self):
        """
        Test 2: chunk size is 3, WS is 5, cache carrying t1 - t5 (t0 is overwritten);
        input t6, t7, t8.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([3])
        kv_seq_len = ShapeTuple([8])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
        # pylint: disable=line-too-long
        #   |                              IN CACHE                                    |             CURRENT CHUNK                |
        #          t1              t2             t3             t4           t5             t6             t7             t8
            [-3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38],
            [-3.402823e+38, -3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38],
            [-3.402823e+38, -3.402823e+38, -3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_subsequent_prefill_3(self):
        """
        Test 3: chunk size is 5, WS is 5, cache carrying t0-t4; input t5-t9.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([5])
        kv_seq_len = ShapeTuple([10])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
        # pylint: disable=line-too-long
        # |                         IN CACHE                                       |                            CURRENT CHUNK                               |
        #     t0              t1             t2             t3           t4             t5             t6             t7             t8             t9
        [-3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38, -3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38, 3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_subsequent_prefill_4(self):
        """
        Test 4: chunk size is 5, WS is 3, cache carrying t2-t4 (t0, t1 did not
        stay in cache); input t5-t9.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([5])
        kv_seq_len = ShapeTuple([8])
        sliding_window = ShapeTuple([3])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
        # pylint: disable=line-too-long
        # |                 IN CACHE                 |                             CURRENT CHUNK                               |
        #     t2              t3             t4              t5           t6             t7             t8              t9
        [-3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38, -3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, -3.402823e+38],
        [-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_subsequent_prefill_5(self):
        """
        Test 5: chunk size is 5, WS is 5, cache carrying t5-t9 (t0-t4 overwritten);
        input t10 (remainder of a prompt). Note that this test can also be 
        viewed as a decode. That is, prefilling a chunk of size 1, is the same is decoding.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([1])
        kv_seq_len = ShapeTuple([6])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
        # pylint: disable=line-too-long
        #   |                            IN CACHE                                     |CURRENT CHUNK|
        #          t5             t6             t7             t8            t9            t10
            [-3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    ################### 3. TESTS FOR DECODE ###################
    def test_decode_1(self):
        """
        Test 1: chunk size is 5, WS is 5, cache carrying t5-t9 (t0-t4 overwritten);
        input t10 (decoding).
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([1])
        kv_seq_len = ShapeTuple([6])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
        # pylint: disable=line-too-long
        #   |                            IN CACHE                                     |CURRENT CHUNK|
        #          t5             t6             t7             t8            t9            t10
            [-3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38,  3.402823e+38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)

    def test_decode_2(self):
        """
        Test 2 (Cache not full): prompt is size 4, WS is 5, cache carrying t0-t3; input t4.
        """
        bsz = ShapeTuple([1])
        seq_length = ShapeTuple([1])
        kv_seq_len = ShapeTuple([5])
        sliding_window = ShapeTuple([5])

        result = vm["main"](bsz, seq_length, kv_seq_len, sliding_window)

        correct = np.array([[[
        #   |                          IN CACHE                         |CURRENT CHUNK|
        #          t0             t1             t2             t3            t4
            [3.402823e+38,  3.402823e+38,  3.402823e+38,  3.402823e+38, 3.402823e+38]
        ]]]).astype("float32")

        np.testing.assert_array_equal(result.numpy(), correct)


if __name__ == "__main__":
    unittest.main()
