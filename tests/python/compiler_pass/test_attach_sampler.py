"""The greedy sampler returns the token the sort-based sampler picks at temperature zero."""

import numpy as np
import pytest
import tvm_tirx
from tvm import relax
from tvm_tirx.runtime import tensor

from mlc_llm.compiler_pass.attach_sampler import (
    AttachGPUSamplingFunc,
    _attach_argmax_logits,
    _attach_argsort_func,
)

# test category "unittest"
pytestmark = [pytest.mark.unittest]

ACTIVE_VOCAB_SIZE = 150000
VOCAB_SIZE = 151936


def _sampler_module(target: tvm_tirx.target.Target) -> tvm_tirx.IRModule:
    metadata = {"active_vocab_size": ACTIVE_VOCAB_SIZE}
    return AttachGPUSamplingFunc(target, {"batch_size": 4}, metadata)(tvm_tirx.IRModule())


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "webgpu"},
        {"kind": "webgpu", "supports_subgroups": True},
    ],
)
def test_webgpu_argmax_builds(target):
    target = tvm_tirx.target.Target(target, host="llvm")
    with target:
        relax.build(_sampler_module(target), target=target)


def test_argmax_matches_sort_path():
    device = tvm_tirx.metal()
    if not device.exist:
        pytest.skip("Metal runtime is unavailable")
    target = tvm_tirx.target.Target("metal", host="llvm")
    # The pass attaches the argmax for WebGPU only; build it for Metal to run it here.
    bb = relax.BlockBuilder()
    _attach_argmax_logits(bb, target, ACTIVE_VOCAB_SIZE)
    _attach_argsort_func(bb)
    with target:
        mod = relax.backend.DispatchSortScan()(bb.finalize())
        vm = relax.VirtualMachine(relax.build(mod, target=target), device)

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 1, VOCAB_SIZE)).astype("float32")
    # Ties across workgroups, ties inside one workgroup, and a larger logit past the active
    # vocabulary, which the sampler must skip.
    logits[0, 0, [5000, 70000, 140000]] = 10
    logits[1, 0, [1030, 1027]] = 10
    logits[2, 0, ACTIVE_VOCAB_SIZE + 3] = 20
    logits[3, 0, :ACTIVE_VOCAB_SIZE] = -np.inf
    logits[3, 0, 149999] = -1
    tokens = vm["argmax_logits"](tensor(logits, device)).numpy()
    expected = np.argmax(logits[:, 0, :ACTIVE_VOCAB_SIZE], axis=-1)
    np.testing.assert_array_equal(tokens, expected)
    assert list(tokens[:2]) == [5000, 1027]

    # The sort-based path puts equal probabilities in index order.
    probs = (logits[:, 0, :] == logits[:, 0, :ACTIVE_VOCAB_SIZE].max(-1, keepdims=True)).astype(
        "float32"
    )
    probs[:, ACTIVE_VOCAB_SIZE:] = 0
    probs /= probs.sum(-1, keepdims=True)
    _, sorted_indices = vm["argsort_probs"](tensor(probs, device))
    np.testing.assert_array_equal(sorted_indices.numpy()[:, 0], expected)
