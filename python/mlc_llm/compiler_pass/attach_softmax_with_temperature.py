"""A compiler pass that attaches two-stage softmax with temperature."""

import tvm
from tvm import relax, tir
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.script import tir as T

from ..support.max_thread_check import get_max_num_threads_per_block


@tvm.transform.module_pass(opt_level=0, name="AttachSoftmaxWithTemperature")
class AttachSoftmaxWithTemperature:  # pylint: disable=too-few-public-methods
    """Rewrites one-shot softmax into two-stage softmax."""

    def __init__(self, target: tvm.target.Target) -> None:
        self.target = target

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        return _Rewriter(mod, self.target).transform()


@mutator
class _Rewriter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: IRModule, target: tvm.target.Target) -> None:
        super().__init__(mod)
        self.mod = mod
        self.target = target
        self.chunk_size = 4096

    def transform(self) -> IRModule:
        """Entry point"""
        batch_size = tir.SizeVar("batch_size", "int64")
        vocab_size = tir.SizeVar("vocab_size", "int64")
        dtype = "float32"
        logits = relax.Var("logits", relax.TensorStructInfo([batch_size, 1, vocab_size], dtype))
        temperature = relax.Var("temperature", relax.TensorStructInfo([batch_size], dtype))
        with self.builder_.function("softmax_with_temperature", params=[logits, temperature]):
            with self.builder_.dataflow():
                output_struct_info = logits.struct_info  # pylint: disable=no-member
                new_shape = relax.ShapeExpr([batch_size, vocab_size])
                logits = relax.call_pure_packed(
                    "vm.builtin.reshape",
                    logits,
                    new_shape,
                    sinfo_args=relax.TensorStructInfo(new_shape, dtype),
                )
                f_chunk_lse, f_softmax_with_lse = _get_lse_and_softmax_func(
                    self.target, self.chunk_size
                )
                chunked_result_struct_info = relax.TensorStructInfo(
                    (batch_size, (vocab_size + self.chunk_size - 1) // self.chunk_size),
                    "float32",
                )
                chunked_results = self.builder_.emit(
                    relax.call_tir(
                        self.builder_.add_func(f_chunk_lse, "chunk_lse"),
                        args=[logits, temperature],
                        out_sinfo=[chunked_result_struct_info, chunked_result_struct_info],
                    )
                )
                chunked_sum = chunked_results[0]
                chunked_max = chunked_results[1]
                softmax = self.builder_.emit(
                    relax.call_tir(
                        self.builder_.add_func(f_softmax_with_lse, "softmax_with_chunked_sum"),
                        args=[logits, temperature, chunked_sum, chunked_max],
                        out_sinfo=logits.struct_info,
                    )
                )
                softmax = self.builder_.emit_output(
                    relax.call_pure_packed(
                        "vm.builtin.reshape",
                        softmax,
                        output_struct_info.shape,
                        sinfo_args=output_struct_info,
                    )
                )
            self.builder_.emit_func_output(softmax)
        return self.builder_.get()


def _get_lse_and_softmax_func(  # pylint: disable=too-many-locals,too-many-statements
    target: tvm.target.Target, chunk_size: int
):
    # NOTE: A quick note on the softmax implementation.
    # We once tried to multiply every element by log2e which can be computed
    # potentially more efficiently on hardware.
    # However, when the input values are large, multiplying by the factor of log2e
    # causes numerical issue in float32 dtype.
    # This leads to the softmax output not summing up to 1.
    # For numerical stability, we removed the log2e factor and switched back
    # to the standard log/exp computation.

    # The kernels below handle both the cases of temperature=0 and temperature != 0.
    # - When temperature is not 0, the first kernel computes the log-sum-exp of
    # chunks (subtracted by the max value in chunk), and the max values of chunks.
    # The second kernel merges the log-sum-exp with the maximum values.
    # - When temperature is 0, the first kernel computes the max value and the counts
    # of the max value. The second kernel merges the max and counts, and set the
    # softmax of the maximum values to "max_value / max_count".

    # pylint: disable=invalid-name
    @T.prim_func
    def chunk_lse(  # pylint: disable=too-many-locals
        var_A: T.handle,
        var_temperature: T.handle,
        var_chunked_sum: T.handle,
        var_chunked_max: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        batch_size = T.int64(is_size_var=True)
        vocab_size = T.int64(is_size_var=True)
        num_chunks = T.int64(is_size_var=True)
        A = T.match_buffer(var_A, (batch_size, vocab_size), dtype="float32")
        temperature = T.match_buffer(var_temperature, (batch_size,), dtype="float32")
        chunked_sum = T.match_buffer(var_chunked_sum, (batch_size, num_chunks), dtype="float32")
        chunked_max = T.match_buffer(var_chunked_max, (batch_size, num_chunks), dtype="float32")
        A_pad = T.alloc_buffer((batch_size, num_chunks, T.int64(chunk_size)), dtype="float32")
        temp_max = T.alloc_buffer((batch_size, num_chunks), dtype="float32")
        temp_sum = T.alloc_buffer((batch_size, num_chunks), dtype="float32")

        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("pad"):
                v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                A_pad[v0, v1, v2] = T.if_then_else(
                    v1 * T.int64(chunk_size) + v2 < vocab_size,
                    T.if_then_else(
                        temperature[v0] > T.float32(1e-5),
                        A[v0, v1 * T.int64(chunk_size) + v2] / temperature[v0],
                        A[v0, v1 * T.int64(chunk_size) + v2],
                    ),
                    T.min_value("float32"),
                )
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("max"):
                v0, v1, v2 = T.axis.remap("SSR", [l0, l1, l2])
                with T.init():
                    temp_max[v0, v1] = T.min_value("float32")
                temp_max[v0, v1] = T.max(temp_max[v0, v1], A_pad[v0, v1, v2])
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("sum_exp"):
                v0, v1, v2 = T.axis.remap("SSR", [l0, l1, l2])
                with T.init():
                    temp_sum[v0, v1] = T.float32(0)
                temp_sum[v0, v1] += T.if_then_else(
                    v1 * T.int64(chunk_size) + v2 < vocab_size,
                    T.Select(
                        temperature[v0] > T.float32(1e-5),
                        T.exp(A_pad[v0, v1, v2] - temp_max[v0, v1]),
                        T.cast(A_pad[v0, v1, v2] == temp_max[v0, v1], "float32"),
                    ),
                    T.float32(0),
                )
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(1)):
            with T.block("log"):
                v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                chunked_sum[v0, v1] = T.Select(
                    temperature[v0] > T.float32(1e-5),
                    T.log(temp_sum[v0, v1]),
                    temp_sum[v0, v1],
                )
                chunked_max[v0, v1] = temp_max[v0, v1]

    @T.prim_func
    def softmax_with_chunked_sum(
        var_A: T.handle,
        var_temperature: T.handle,
        var_chunked_sum: T.handle,
        var_chunked_max: T.handle,
        var_softmax: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
        batch_size = T.int64(is_size_var=True)
        vocab_size = T.int64(is_size_var=True)
        num_chunks = T.int64(is_size_var=True)
        A = T.match_buffer(var_A, (batch_size, vocab_size), dtype="float32")
        temperature = T.match_buffer(var_temperature, (batch_size,), dtype="float32")
        chunked_sum = T.match_buffer(var_chunked_sum, (batch_size, num_chunks), dtype="float32")
        chunked_max = T.match_buffer(var_chunked_max, (batch_size, num_chunks), dtype="float32")
        softmax = T.match_buffer(var_softmax, (batch_size, vocab_size), dtype="float32")
        temp_max = T.alloc_buffer((batch_size,), dtype="float32")
        temp_sum = T.alloc_buffer((batch_size,), dtype="float32")
        for l0, l1 in T.grid(batch_size, num_chunks):
            with T.block("max"):
                v0, v1 = T.axis.remap("SR", [l0, l1])
                with T.init():
                    temp_max[v0] = T.min_value("float32")
                temp_max[v0] = T.max(temp_max[v0], chunked_max[v0, v1])
        for l0, l1 in T.grid(batch_size, num_chunks):
            with T.block("sum_exp"):
                v0, v1 = T.axis.remap("SR", [l0, l1])
                with T.init():
                    temp_sum[v0] = T.float32(0)
                temp_sum[v0] += T.Select(
                    temperature[v0] > T.float32(1e-5),
                    T.exp(chunked_sum[v0, v1] + chunked_max[v0, v1] - temp_max[v0]),
                    T.cast(chunked_max[v0, v1] == temp_max[v0], "float32") * chunked_sum[v0, v1],
                )
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("log_pad"):
                v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                if v1 * T.int64(chunk_size) + v2 < vocab_size:
                    softmax[v0, v1 * T.int64(chunk_size) + v2] = T.if_then_else(
                        temperature[v0] > T.float32(1e-5),
                        T.exp(
                            A[v0, v1 * T.int64(chunk_size) + v2] / temperature[v0]
                            - (T.log(temp_sum[v0]) + temp_max[v0])
                        ),
                        T.cast(A[v0, v1 * T.int64(chunk_size) + v2] == temp_max[v0], "float32")
                        / temp_sum[v0],
                    )

    sch = tvm.tir.Schedule(IRModule({"softmax_with_chunked_sum": softmax_with_chunked_sum}))

    def apply_gpu_schedule(target, sch):
        max_threads = get_max_num_threads_per_block(target)
        TX = 32
        TY = max_threads // TX
        unroll_depth = 64
        # pylint: enable=invalid-name

        sch.work_on("softmax_with_chunked_sum")
        l0, l1, l2 = sch.get_loops("log_pad")
        bx = sch.fuse(l0, l1)
        sch.bind(bx, "blockIdx.x")
        unroll, ty, tx = sch.split(l2, [None, TY, TX])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(unroll, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(unroll, ann_key="pragma_unroll_explicit", ann_val=1)

        for block_name in ["sum_exp", "max"]:
            block = sch.get_block(block_name)
            sch.set_scope(block, buffer_index=0, storage_scope="shared")
            sch.compute_at(block, bx)
            r_loop = sch.get_loops(block)[-1]
            r_loop, tx = sch.split(r_loop, [None, TX])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")
            sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

        return chunk_lse, sch.mod["softmax_with_chunked_sum"]

    if target.kind.name == "llvm":
        return chunk_lse, sch.mod["softmax_with_chunked_sum"]
    return apply_gpu_schedule(target, sch)
