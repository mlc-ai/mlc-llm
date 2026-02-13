"""Utility Relax op helpers for LoRA.

This is a *temporary* pure-Python implementation that builds the LoRA fused
projection as a composition of existing Relax ops so that the graph works on
all targets today.  Once a dedicated C++ op / fused schedule lands we can swap
this helper out behind the same call-site without touching the rest of the
Python stack.
"""

from __future__ import annotations

from typing import Union

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def lora_dense(
    x: Tensor,
    base_weight: Tensor,
    lora_weight: Tensor,
    alpha: Union[float, Tensor],
) -> Tensor:  # noqa: D401 – not property
    """LoRA-aware dense layer.

    Computes ``Y = dense(x, base_weight) + alpha * dense(x, lora_weight)`` using
    existing Relax building blocks.  Because it relies purely on public ops it
    will run on any backend that already supports *dense*.

    Parameters
    ----------
    x : Tensor
        Input activations of shape (batch, in_features).
    base_weight : Tensor
        Pre-trained weight matrix of shape (out_features, in_features).
    lora_weight : Tensor
        Low-rank LoRA delta matrix of shape (out_features, in_features).
    alpha : float or Tensor
        Scaling factor to apply to the LoRA contribution.
    """

    out_base = op.matmul(x, op.permute_dims(base_weight))
    out_lora = op.matmul(x, op.permute_dims(lora_weight))

    if not isinstance(alpha, nn.Tensor):
        alpha = nn.const(alpha, x.dtype)  # pylint: disable=no-member

    return out_base + out_lora * alpha
