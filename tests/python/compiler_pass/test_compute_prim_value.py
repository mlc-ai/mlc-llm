"""Scalar call_tir arguments computed from symbolic shapes must survive VM lowering."""

import pytest
import tvm
from tvm import relax, tirx
from tvm.script import s_tir as Ts
from tvm.script import tirx as T

from mlc_llm.compiler_pass.pipeline import _mlc_llm_pipeline

pytestmark = [pytest.mark.unittest]


@pytest.mark.parametrize("target_kind", ["metal", "cuda"])
def test_computed_scalar_argument_lowers(target_kind):
    """The offsets LLaVA's crop kernel receives are derived from a float resize of the input dims.

    Without ComputePrimValue ahead of VMShapeLower, VMShapeLower rewrites only the symbolic
    leaves of such an expression and CodeGenVM is left holding the arithmetic on top of them.
    """
    n = T.dynamic("n", "int64")

    @Ts.prim_func(private=True)
    def take(x: T.Buffer((n,), "float32"), off: T.int64, y: T.Buffer((n,), "float32")):
        for i in T.serial(n):
            with Ts.sblock("copy"):
                vi = Ts.axis.spatial(n, i)
                y[vi] = x[vi] + T.Cast("float32", off)

    h = tirx.Var("h", "int64")
    w = tirx.Var("w", "int64")
    resized = tirx.Cast(
        "int64",
        tirx.const(336.0, "float32")
        * (
            tirx.Cast("float32", tirx.Select(w > h, w, h))
            / tirx.Cast("float32", tirx.Select(w < h, w, h))
        ),
    )
    top = tirx.floordiv(resized - tirx.const(336, "int64"), tirx.const(2, "int64"))

    x = relax.Var("x", relax.TensorType((h,), "float32"))
    xm = relax.Var("xm", relax.TensorType((h, w), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, xm]):
        with bb.dataflow():
            gv = bb.add_func(take, "take")
            call = relax.call_tir(gv, [x, top], out_ty=relax.TensorType((h,), "float32"))
            out = bb.emit_output(call)
        bb.emit_func_output(out)
    mod = bb.get()

    target = tvm.target.Target(target_kind, host="llvm")
    pipeline = _mlc_llm_pipeline(
        target,
        variable_bounds={"batch_size": 1},
        metadata={"pipeline_parallel_stages": 1},
    )
    with target:
        lowered = pipeline(mod)
    # Stop at VM codegen, which is where an unlowered PrimExpr argument is rejected. The TIR half of
    # the build would need a working device toolchain.
    relax.vm_build._vmcodegen(relax.ExecBuilder(), lowered)
