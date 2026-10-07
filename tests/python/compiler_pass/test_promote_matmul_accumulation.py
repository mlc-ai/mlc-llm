"""An explicit accumulation policy must preserve ABI and improve reduction accuracy."""

import numpy as np
import pytest
import tvm
from tvm import relax, tirx

from mlc_llm.compiler_pass.promote_matmul_accumulation import PromoteMatmulAccumulation


def model(dtype="float16", out_dtype=None, dynamic=False):
    rows = tirx.Var("rows", "int64") if dynamic else 2
    x = relax.Var("x", relax.TensorType((rows, 4096), dtype))
    w = relax.Var("w", relax.TensorType((4096, 3), dtype))
    builder = relax.BlockBuilder()
    with builder.function("main", [x, w]):
        with builder.dataflow():
            value = builder.emit(relax.op.matmul(x, w, out_dtype=out_dtype))
            result = builder.emit_output(value)
        builder.emit_func_output(result)
    return builder.get()


def test_symbolic_output_abi_and_idempotence():
    original = model(dynamic=True)
    promoted = PromoteMatmulAccumulation()(original)
    tvm.ir.assert_structural_equal(original["main"].ret_ty, promoted["main"].ret_ty)
    tvm.ir.assert_structural_equal(promoted, PromoteMatmulAccumulation()(promoted))
    assert 'out_dtype="float32"' in promoted.script()
    assert "astype" in promoted.script()


@pytest.mark.parametrize(
    "dtype,out_dtype", [("float32", None), ("float16", "float32"), ("int32", None)]
)
def test_preserves_other_arithmetic(dtype, out_dtype):
    original = model(dtype, out_dtype)
    tvm.ir.assert_structural_equal(original, PromoteMatmulAccumulation()(original))


def test_fp32_reduction_matches_oracle_with_fp16_inputs_and_outputs():
    original = model()
    promoted = PromoteMatmulAccumulation()(original)
    executable = tvm.compile(promoted, target="llvm")
    vm = relax.VirtualMachine(executable, tvm.cpu())
    rng = np.random.default_rng(821)
    x = rng.normal(size=(2, 4096)).astype("float16")
    w = rng.normal(size=(4096, 3)).astype("float16")
    expected = (x.astype("float32") @ w.astype("float32")).astype("float16")
    actual = vm["main"](tvm.runtime.tensor(x), tvm.runtime.tensor(w)).numpy()
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, atol=0.0625, rtol=0.001)
