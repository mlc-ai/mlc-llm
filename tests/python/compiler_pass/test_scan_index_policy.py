"""MLC's scan hierarchy budget must match its subsequent index narrowing."""

import pytest
import tvm
from tvm import tirx
from tvm.relax.backend.gpu_generic import gpu_2d_continuous_cumsum
from tvm.script import ir as I
from tvm.script import relax as R

from mlc_llm.compiler_pass.pipeline import _mlc_llm_pipeline


@pytest.mark.parametrize("target_kind", ["metal", "webgpu", "cuda"])
def test_scan_dispatch_matches_pipeline_index_policy(target_kind):
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")):
            return R.cumsum(x, axis=-1)

    @tvm.ir.instrument.pass_instrument
    class ScanPassesOnly:
        """Exercise the actual pipeline's dispatch and narrowing on a minimal graph."""

        def __init__(self):
            self.dispatched = None
            self.narrowed = False

        def should_run(self, _mod, info):
            return info.name in (
                "_pipeline",
                "sequential",
                "DispatchSortScan",
                "tirx.NarrowDataType",
            )

        def run_after_pass(self, mod, info):
            if info.name == "DispatchSortScan":
                self.dispatched = mod["gpu_2d_continuous_cumsum"]
            elif info.name == "tirx.NarrowDataType":
                self.narrowed = True

    target = tvm.target.Target(target_kind, host="llvm")
    pipeline = _mlc_llm_pipeline(
        target,
        variable_bounds={"batch_size": 1},
        metadata={"pipeline_parallel_stages": 1},
    )
    instrument = ScanPassesOnly()
    with target, tvm.transform.PassContext(instruments=[instrument]):
        result = pipeline(Module)

    index_bits = 64 if target_kind == "cuda" else 32
    expected = gpu_2d_continuous_cumsum(
        in_dtype="float32", out_dtype="float32", index_bits=index_bits
    )
    tvm.ir.assert_structural_equal(instrument.dispatched, expected)
    assert instrument.narrowed == (index_bits == 32)
    if index_bits == 32:
        expected_mod = tirx.transform.ForceNarrowIndexToInt32()(tvm.IRModule({"main": expected}))
        expected = expected_mod["main"]
    tvm.ir.assert_structural_equal(result["gpu_2d_continuous_cumsum"], expected)
