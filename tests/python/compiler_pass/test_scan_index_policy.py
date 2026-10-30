"""MLC scan dispatch follows the target's supported index width."""

import pytest
import tvm_tirx
from tvm.relax import script as R
from tvm.relax.backend.gpu_generic import gpu_2d_continuous_cumsum
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

from mlc_llm.compiler_pass.pipeline import _mlc_llm_pipeline


@pytest.mark.parametrize("target_kind", ["metal", "webgpu", "cuda"])
def test_scan_dispatch_matches_pipeline_index_policy(target_kind):
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((m, n), "float32")):
            return R.cumsum(x, axis=-1)

    @tvm_tirx.ir.transform.pass_instrument
    class ScanPassesOnly:
        """Exercise the pipeline's scan dispatch on a minimal graph."""

        def __init__(self):
            self.dispatched = None

        def should_run(self, _mod, info):
            return info.name in (
                "_pipeline",
                "sequential",
                "DispatchSortScan",
            )

        def run_after_pass(self, mod, info):
            if info.name == "DispatchSortScan":
                self.dispatched = mod["gpu_2d_continuous_cumsum"]

    target = tvm_tirx.target.Target(target_kind, host="llvm")
    pipeline = _mlc_llm_pipeline(
        target,
        variable_bounds={"batch_size": 1},
        metadata={"pipeline_parallel_stages": 1},
    )
    instrument = ScanPassesOnly()
    with target, tvm_tirx.transform.PassContext(instruments=[instrument]):
        result = pipeline(Module)

    index_bits = 32 if target_kind == "webgpu" else 64
    expected = gpu_2d_continuous_cumsum(
        in_dtype="float32", out_dtype="float32", index_bits=index_bits
    )
    tvm_tirx.ir.assert_structural_equal(instrument.dispatched, expected)
    tvm_tirx.ir.assert_structural_equal(result["gpu_2d_continuous_cumsum"], expected)
