# pylint: disable=invalid-name,missing-docstring
"""Tests for the naive attention fallback when head_dim % 16 != 0.

The TIR attention kernel (_attention_sequence_prefill) produces incorrect results
for head_dim values that are not multiples of 16 (e.g. 72 for SigLIP). The fallback
in mlc_llm/op/attention.py uses matmul+softmax instead. This test verifies correctness
of that fallback path against a numpy reference.
"""

import math

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, spec as nn_spec

from mlc_llm import op as op_ext

# test category "op_correctness"
pytestmark = [pytest.mark.op_correctness]


class _AttentionModule(nn.Module):
    """Minimal module wrapping op_ext.attention for testing."""

    def __init__(self, h: int, d: int, seq_len: int):
        self.h = h
        self.d = d
        self.seq_len = seq_len

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return op_ext.attention(q, k, v, None)

    def get_default_spec(self):
        # Use concrete seq_len so TVM can generate proper kernels
        return nn_spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "q": nn_spec.Tensor([1, self.seq_len, self.h, self.d], "float32"),
                    "k": nn_spec.Tensor([1, self.seq_len, self.h, self.d], "float32"),
                    "v": nn_spec.Tensor([1, self.seq_len, self.h, self.d], "float32"),
                    "$": {"param_mode": "none", "effect_mode": "none"},
                },
            },
            self,
        )


def _numpy_attention(q, k, v):
    """Numpy reference for attention without causal mask.

    q, k, v: (1, s, h, d) float32
    Returns: (1, s, h*d) float32
    """
    b, s, h, d = q.shape
    # (b, h, s, d)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(d)
    # (b, h, s, s)
    attn_weights = np.matmul(q_t, k_t.transpose(0, 1, 3, 2)) * scale
    # softmax along last axis
    attn_weights = attn_weights - np.max(attn_weights, axis=-1, keepdims=True)
    attn_weights = np.exp(attn_weights)
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    # (b, h, s, d)
    out = np.matmul(attn_weights, v_t)
    # (b, s, h*d)
    return out.transpose(0, 2, 1, 3).reshape(b, s, h * d)


@pytest.mark.parametrize("d", [72, 40])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seq_len", [8, 32])
def test_attention_naive_fallback(d, h, seq_len):
    """Test that the naive matmul+softmax fallback produces correct results
    for head_dim values not divisible by 16."""
    assert d % 16 != 0, f"head_dim={d} is divisible by 16, this test targets non-divisible values"

    np.random.seed(42)
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(0)

    # Build module
    with target:
        attn_mod = _AttentionModule(h=h, d=d, seq_len=seq_len)
        mod, _ = attn_mod.export_tvm(spec=attn_mod.get_default_spec())

    # Compile
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    # Generate random inputs
    q_np = np.random.randn(1, seq_len, h, d).astype("float32")
    k_np = np.random.randn(1, seq_len, h, d).astype("float32")
    v_np = np.random.randn(1, seq_len, h, d).astype("float32")

    # Run TVM
    q_tvm = tvm.runtime.tensor(q_np, dev)
    k_tvm = tvm.runtime.tensor(k_np, dev)
    v_tvm = tvm.runtime.tensor(v_np, dev)
    result = vm["forward"](q_tvm, k_tvm, v_tvm)
    result_np = result.numpy()

    # Numpy reference
    expected = _numpy_attention(q_np, k_np, v_np)

    tvm.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
