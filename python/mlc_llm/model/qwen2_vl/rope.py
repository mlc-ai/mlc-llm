from typing import Dict, Tuple
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op, Tensor

from mlc_llm.nn import RopeMode

def _rotate_half(x: Tensor) -> Tensor:
    """
    Swap paired features (..., 2k, 2k+1) -> (..., -x_{2k+1}, x_{2k})
    Implemented with reshape/concat to avoid custom kernels.
    """
    *lead, d = x.shape
    assert d % 2 == 0, "head_dim must be even for RoPE"
    x2 = op.reshape(x, (*lead, d // 2, 2))
    x_even = x2[..., 0]               # (..., D/2)
    x_odd  = x2[..., 1]               # (..., D/2)
    y = op.concatenate((-x_odd, x_even), axis=-1)  # (..., D)
    return y

def precompute_rope_cache(
    dim: int,
    num_heads: int,
    max_seq_len: int,
    rope_mode: int = RopeMode.NORMAL,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> Dict[str, Tensor]:
    """
    Build broadcastable cos/sin caches for RoPE:
      cos, sin shapes: (max_seq_len, 1, 1, dim)
    They broadcast to (B, L, H, D) and allow fast slicing with an offset.

    We use the standard RoPE frequencies:
      inv_freq = rope_scale / (rope_theta ** (arange(0, dim, 2) / dim))
      freqs = outer(arange(L), inv_freq)   # (L, D/2)
    Then expand to (L, 1, 1, D) by 'pair-repeat' so it aligns with interleaved coords.
    """
    assert dim % 2 == 0, "head_dim must be even"

    # positions: (L,)
    pos = op.arange(0, max_seq_len, dtype="float32")             # (L,)
    pos = op.reshape(pos, (max_seq_len, 1))                      # (L, 1)

    # inverse frequencies: (D/2,)
    two_step = op.arange(0, dim, 2, dtype="float32")             # (D/2,)
    inv = op.divide(two_step, relax.const(dim, "float32"))       # (D/2,)  -> i/d
    base_pow = op.power(relax.const(rope_theta, "float32"), inv) # rope_theta ** (i/d)
    inv_freq = op.divide(relax.const(rope_scale, "float32"), base_pow)  # (D/2,)

    inv_freq = op.reshape(inv_freq, (1, dim // 2))               # (1, D/2)
    # freqs: (L, D/2)
    freqs = op.multiply(pos, inv_freq)

    # cos/sin: (L, D/2)
    cos = op.cos(freqs)
    sin = op.sin(freqs)

    # Expand to (L, 1, 1, D) by repeating the last axis twice (pair-wise)
    # Do this via broadcast_to + reshape (avoid needing a repeat op).
    cos = op.reshape(cos, (max_seq_len, 1, 1, dim // 2, 1))      # (L,1,1,D/2,1)
    sin = op.reshape(sin, (max_seq_len, 1, 1, dim // 2, 1))      # (L,1,1,D/2,1)

    cos = op.broadcast_to(cos, (max_seq_len, 1, 1, dim // 2, 2)) # (L,1,1,D/2,2)
    sin = op.broadcast_to(sin, (max_seq_len, 1, 1, dim // 2, 2)) # (L,1,1,D/2,2)

    cos = op.reshape(cos, (max_seq_len, 1, 1, dim))              # (L,1,1,D)
    sin = op.reshape(sin, (max_seq_len, 1, 1, dim))              # (L,1,1,D)

    # Return a simple dict; caller (your attention) selects per head-group scale
    return {"cos": cos, "sin": sin, "dim": dim, "max_seq_len": max_seq_len}

def apply_rotary_emb(
    x: Tensor,            # (B, L, H, D)
    rope_cache: Dict[str, Tensor],
    offset: int = 0,
    num_heads: int | None = None,   # kept for API parity; not required here
) -> Tensor:
    """
    Apply RoPE to `x` using cached cos/sin. The cache is typically produced per head-group
    (i.e., per M-RoPE scale). We simply slice the proper [offset:offset+L] window and
    broadcast-multiply.

    Returns: Tensor of same shape as x.
    """
    cos, sin = rope_cache["cos"], rope_cache["sin"]  # (Lmax,1,1,D)
    B, L, H, D = x.shape

    # Slice to the current sequence window
    cos_t = cos[offset: offset + L]                 # (L,1,1,D)
    sin_t = sin[offset: offset + L]                 # (L,1,1,D)

    # Broadcast to (B,L,H,D)
    # Expand (L,1,1,D) -> (1,L,1,1,D) then reshape to (B,L,H,D) via broadcast_to
    cos_t = op.expand_dims(cos_t, axis=0)          # (1,L,1,1,D)
    sin_t = op.expand_dims(sin_t, axis=0)          # (1,L,1,1,D)

    cos_t = op.broadcast_to(cos_t, (B, L, H, D))   # (B,L,H,D)
    sin_t = op.broadcast_to(sin_t, (B, L, H, D))   # (B,L,H,D)

    return x * cos_t + _rotate_half(x) * sin_t
