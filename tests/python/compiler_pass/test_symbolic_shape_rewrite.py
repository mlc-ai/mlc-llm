"""Symbolic shape rewriting with TVM's structural traversal API."""

import tvm_tirx
from tvm import relax
from tvm_tirx import ir, tirx

from mlc_llm.compiler_pass.lift_global_buffer_alloc import _resolve_tir_var_mapping
from mlc_llm.compiler_pass.pipeline_parallel_rewrite import _PipelineParallelRewriter


def test_pipeline_shapes_share_fresh_symbols():
    n = tirx.Var("n", "int64")
    rewriter = _PipelineParallelRewriter(tvm_tirx.IRModule())
    replacements = {}
    shapes = rewriter._update_shape([n + 1, 2 * n, ir.IntImm("int64", 8)], replacements)

    assert len(replacements) == 1
    fresh = replacements[n]
    assert not fresh.same_as(n)
    tvm_tirx.ir.assert_structural_equal(shapes[0], fresh + 1)
    tvm_tirx.ir.assert_structural_equal(shapes[1], 2 * fresh)
    tvm_tirx.ir.assert_structural_equal(shapes[2], ir.IntImm("int64", 8))
    tvm_tirx.ir.assert_structural_equal(rewriter._update_shape([n], replacements)[0], fresh)


def test_lifted_buffer_shape_uses_caller_symbols():
    n = tirx.Var("n", "int64")
    m = tirx.Var("m", "int64")
    source = tirx.Var("source", tirx.TensorType((n,), "float32"))
    output = tirx.Var("output", tirx.TensorType((n,), "float32"))
    func = tirx.Function([source, output], tvm_tirx.ir.Evaluate(0))
    x = relax.Var("x", relax.TensorType((m,), "float32"))
    call = relax.call_tir(
        tvm_tirx.ir.GlobalVar("copy"), [x], ty_args=[relax.TensorType((m,), "float32")]
    )

    shapes, resolved = _resolve_tir_var_mapping(
        func, call, [relax.TensorType((n * 2 + 1,), "float32")]
    )
    assert resolved
    tvm_tirx.ir.assert_structural_equal(shapes[0].shape.values[0], m * 2 + 1)
