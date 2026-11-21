"""Exported MoE kernels respect WebGPU workgroup limits and retain tail guards."""

import ast
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from tvm.relax.frontend import nn
from tvm_tirx.target import Target

from mlc_llm.op import moe_misc


@pytest.mark.parametrize("operation", ["topk", "softmax_topk", "indices"])
@pytest.mark.parametrize(
    "target_name, expected_threads",
    [
        (None, 1024),
        ("cuda", 1024),
        ("webgpu", 256),
        ({"kind": "webgpu", "max_num_threads": 128}, 128),
    ],
)
def test_exported_thread_limits_and_tails(operation, target_name, expected_threads):
    class Module(nn.Module):
        def forward(self, values, indices):
            if operation == "topk":
                return moe_misc.gating_topk(values, 2)
            if operation == "softmax_topk":
                return moe_misc.gating_softmax_topk(values, 2)
            return moe_misc.get_indices(values, indices)

    module = Module()
    spec = {
        "forward": {
            "values": nn.spec.Tensor(
                ("tokens", 4) if operation != "indices" else ("entries",),
                "float32" if operation != "indices" else "int32",
            ),
            "indices": nn.spec.Tensor(("tokens", 2), "int32"),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        }
    }
    with Target(target_name) if target_name else nullcontext():
        exported, _, _ = module.export_tvm(spec=spec, allow_extern=True)
    tree = ast.parse(exported.script())
    loops = {}
    guards = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            for keyword in node.iter.keywords:
                if keyword.arg == "annotations":
                    annotation = ast.literal_eval(keyword.value)
                    if "thread_binding" in annotation:
                        loops[annotation["thread_binding"]] = node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "where" and "batch_size" in ast.unparse(node):
                guards.append(node.args[0])
        if isinstance(node, ast.If) and "batch_size" in ast.unparse(node.test):
            guards.append(node.test)
    thread = loops["threadIdx.x"]
    block = loops["blockIdx.x"]
    assert ast.literal_eval(thread.iter.args[1]) == expected_threads
    assert len(guards) == 1

    # Evaluate the actual exported grid and predicate on both sides of each tail.
    def evaluate(expression, bindings):
        return eval(  # pylint: disable=eval-used
            compile(ast.Expression(expression), "<exported-moe-index>", "eval"),
            {"__builtins__": {}},
            bindings,
        )

    for rows in (1, expected_threads + 1, 2 * expected_threads - 1):
        count = rows * (2 if operation == "indices" else 1)
        bindings = {
            "T": SimpleNamespace(int64=int, int32=int, Cast=lambda _dtype, value: value),
            "batch_size": rows,
            block.target.id: (count - 1) // expected_threads,
        }
        assert (
            evaluate(block.iter.args[-1], bindings)
            == (count + expected_threads - 1) // expected_threads
        )
        for offset, valid in ((0, True), (1, False)):
            bindings[thread.target.id] = (count - 1) % expected_threads + offset
            assert evaluate(guards[0], bindings) == valid


@pytest.mark.parametrize("operation", [moe_misc.gating_topk, moe_misc.gating_softmax_topk])
def test_gating_initializes_each_local_expert_index(operation):
    class Module(nn.Module):
        def forward(self, scores):
            return operation(scores, 8)

    with Target("webgpu"):
        exported, _, _ = Module().export_tvm(
            spec={
                "forward": {
                    "scores": nn.spec.Tensor(("tokens", 128), "float16"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            allow_extern=True,
        )
    init = next(
        node
        for node in ast.walk(ast.parse(exported.script()))
        if isinstance(node, ast.With)
        and ast.unparse(node.items[0].context_expr) == "Ts.sblock('init')"
    )
    initializers = [
        (ast.literal_eval(store.targets[0].slice), ast.literal_eval(store.value))
        for store in init.body
        if isinstance(store, ast.Assign) and not isinstance(store.value, ast.Call)
    ]
    assert initializers == [(i, -1) for i in range(8)]
