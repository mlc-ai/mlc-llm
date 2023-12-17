"""Operators enabled by external modules."""
import math

from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from .. import extern as _extern
from ..extern import configure

__all__ = [
    "attention",
    "configure",
]


def attention(  # pylint: disable=invalid-name
    q: nn.Tensor,
    k: nn.Tensor,
    v: nn.Tensor,
    casual_mask: nn.Tensor,
) -> nn.Tensor:
    """Attention with casual mask.

    --- Variables ---
    s: sequence length of the current query
    t: total sequence length
    d: head dimension
    h, h_q: number of heads in query
    h_kv: number of heads in key and value
    b: batch size = 1

    --- Shapes ---
    q: [b, s, h_q, d]
    k: [t, h_kv, d]
    v: [t, h_kv, d]
    o: [1, s, hidden = h_q * d]

    --- Computation ---

    .. code-block:: python

        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=1)
            v = v.repeat(h_q // h_kv, axis=1)
        q -> [b, h, s, d]
        k, v -> [b, h, t, d]
        attn = q @ k^T / sqrt(d)  # [b, h, s, t]
        attn = softmax_with_mask(attn, casual_mask, axis=-1)
        o = attn @ v  # [b, h, s, d]
        o -> [b, s, h * d]
    """
    assert q.ndim == 4 and k.ndim == 3 and v.ndim == 3
    b, s, h_q, d = q.shape
    t, h_kv, _ = k.shape
    assert b == 1, "batch size must be 1"

    # FlashInfer Implementation
    extern_store = _extern.get_store()
    if (
        extern_store.flashinfer
        and q.dtype == "float16"
        and k.dtype == "float16"
        and v.dtype == "float16"
    ):
        return extern_store.flashinfer.single_batch(q, k, v)

    # Fallback Implementation
    k = op.reshape(k, [b, t, h_kv, d])
    v = op.reshape(v, [b, t, h_kv, d])
    if h_kv != h_q:
        k = k.repeat(h_q // h_kv, axis=2)
        v = v.repeat(h_q // h_kv, axis=2)
    q = q.permute_dims([0, 2, 1, 3])
    k = k.permute_dims([0, 2, 1, 3])
    v = v.permute_dims([0, 2, 1, 3])
    attn_weights = op.matmul(  # [b, h, s, t]
        q,  # [b, h, s, d]
        k.permute_dims([0, 1, 3, 2]),  # [b, h, d, t]
    ) / math.sqrt(d)
    dtype = attn_weights.dtype
    attn_weights = attn_weights.maximum(tir.min_value(dtype)).minimum(casual_mask)
    if dtype == "float32":
        attn_weights = op.softmax(attn_weights, axis=-1)
    else:
        attn_weights = op.softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
    output = op.matmul(attn_weights, v)  # [b, h, s, d] <= [b, h, s, t] x [b, h, t, d]
    output = output.permute_dims([0, 2, 1, 3])  #  [b, s, h, d]
    output = output.reshape([b, s, h_q * d])  # [b, s, h * d]
    return output
