"""Symbolic shape rewriting with TVM's structural traversal API."""

import tvm
from tvm import relax, tirx

from mlc_llm.compiler_pass.lift_global_buffer_alloc import _resolve_tir_var_mapping
from mlc_llm.compiler_pass.pipeline_parallel_rewrite import _PipelineParallelRewriter


def test_pipeline_shapes_share_fresh_symbols():
    n = tirx.Var("n", "int64")
    rewriter = _PipelineParallelRewriter(tvm.IRModule())
    replacements = {}
    shapes = rewriter._update_shape([n + 1, 2 * n, tirx.IntImm("int64", 8)], replacements)

    assert len(replacements) == 1
    fresh = replacements[n]
    assert not fresh.same_as(n)
    tvm.ir.assert_structural_equal(shapes[0], fresh + 1)
    tvm.ir.assert_structural_equal(shapes[1], 2 * fresh)
    tvm.ir.assert_structural_equal(shapes[2], tirx.IntImm("int64", 8))
    tvm.ir.assert_structural_equal(rewriter._update_shape([n], replacements)[0], fresh)


def test_lifted_buffer_shape_uses_caller_symbols():
    n = tirx.Var("n", "int64")
    m = tirx.Var("m", "int64")
    source = tirx.decl_buffer((n,), "float32", name="source")
    output = tirx.decl_buffer((n,), "float32", name="output")
    func = tirx.PrimFunc([source, output], tirx.Evaluate(0))
    x = relax.Var("x", relax.TensorType((m,), "float32"))
    call = relax.call_tir(tvm.ir.GlobalVar("copy"), [x], out_ty=relax.TensorType((m,), "float32"))

    shapes, resolved = _resolve_tir_var_mapping(
        func, call, [relax.TensorType((n * 2 + 1,), "float32")]
    )
    assert resolved
    tvm.ir.assert_structural_equal(shapes[0].shape.values[0], m * 2 + 1)
