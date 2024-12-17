"""Operators for positional embeddings, e.g. RoPE."""

import math
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

from tvm import tir
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T

# pylint: disable=invalid-name


def rope_freq_default(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
    """Compute the inverse frequency of RoPE and then return the cosine and sine of it.

    Parameters
    ----------
    s : tir.Var
        The position index.

    d : tir.Var
        The dimension index.

    d_range : int
        The maximum dimension index.

    theta : float
        The theta value in RoPE, which controls the frequency.

    dtype : str
        The data type of the output.

    Returns
    -------
    cos_freq : Tensor
        The cosine of the inverse frequency.
    sin_freq : Tensor
        The sine of the inverse frequency.
    var_map: Dict[tir.Var, tir.PrimExpr]
        The common expression map.
    """
    freq = s / tir.power(theta, d * 2 % d_range / tir.const(d_range, "float32"))
    freq_var = tir.Var("freq", "float32")
    cos_freq = tir.cos(freq_var).astype(dtype)
    sin_freq = tir.sin(freq_var).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def rope_freq_gptj(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
    """Compute the inverse frequency of RoPE for gptj RoPE scaling."""
    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(d_range, "float32"))
    freq_var = tir.Var("freq", "float32")
    cos_freq = tir.cos(freq_var).astype(dtype)
    sin_freq = tir.sin(freq_var).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def rope_freq_llama3(  # pylint: disable=too-many-arguments,too-many-locals
    s: tir.Var,
    d: tir.Var,
    d_range: int,
    theta: float,
    dtype: str,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: float,
):
    """Compute the inverse frequency of RoPE for llama3 RoPE scaling."""
    orig_freq = tir.const(1, "float32") / tir.power(
        theta, d * 2 % d_range / tir.const(d_range, "float32")
    )
    orig_freq_var = tir.Var("orig_freq", "float32")
    inv_diff_freq_factor = 1.0 / (high_freq_factor - low_freq_factor)
    llama3_inv_scaling_factor = 1.0 / factor
    llama3_alpha = original_max_position_embeddings / (2 * math.pi) * inv_diff_freq_factor
    llama3_beta = low_freq_factor * inv_diff_freq_factor
    smooth = tir.max(0.0, tir.min(1.0, llama3_alpha * orig_freq_var - llama3_beta))
    smoothed_freq = s * (
        (1.0 - smooth) * orig_freq_var * llama3_inv_scaling_factor + smooth * orig_freq_var
    )
    smoothed_freq_var = tir.Var("smoothed_freq", "float32")
    cos_freq = tir.cos(smoothed_freq_var).astype(dtype)
    sin_freq = tir.sin(smoothed_freq_var).astype(dtype)
    return cos_freq, sin_freq, {smoothed_freq_var: smoothed_freq, orig_freq_var: orig_freq}


def rope_freq_longrope(  # pylint: disable=too-many-arguments
    s: tir.Var,
    d: tir.Var,
    d_range: int,
    theta: float,
    dtype: str,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    ext_factors: Optional[T.Buffer] = None,
):
    """Compute the inverse frequency of RoPE for longrope scaling."""
    scale = max_position_embeddings / original_max_position_embeddings
    scaling_factor = (
        math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))
        if scale > 1.0
        else 1.0
    )
    divisor = tir.power(theta, d * 2 % d_range / tir.const(d_range, "float32"))
    if ext_factors is not None:
        divisor = ext_factors[d % (d_range // 2)] * divisor
    freq = s / divisor
    freq_var = tir.Var("freq", "float32")
    cos_freq = (tir.cos(freq_var) * scaling_factor).astype(dtype)
    sin_freq = (tir.sin(freq_var) * scaling_factor).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def yarn_find_correction_dim(
    num_rotations: int,
    d: tir.Var,
    theta: float,
    max_position_embeddings: int,
):
    """Inverse dim formula to find dim based on number of rotations"""
    return (d * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(theta)
    )


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    d: tir.Var,
    theta: float,
    max_position_embeddings: int,
):
    """Find the correction range based on the number of rotations"""
    low = tir.floor(yarn_find_correction_dim(low_rot, d, theta, max_position_embeddings))
    high = tir.ceil(yarn_find_correction_dim(high_rot, d, theta, max_position_embeddings))
    return tir.max(low, 0), tir.min(high, d - 1)


def rope_freq_yarn(
    s: tir.Var,
    d: tir.Var,
    d_range: int,
    theta: float,
    dtype: str,
    original_max_position_embeddings: int,
    scaling_factor: float,
    beta_fast: int,
    beta_slow: int,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Compute the inverse frequency of RoPE for yarn RoPE scaling."""
    freq_extra = tir.const(1, "float32") / tir.power(
        theta, d * 2 % d_range / tir.const(d_range, "float32")
    )

    freq_inter = tir.const(1, "float32") / tir.power(
        scaling_factor * theta, d * 2 % d_range / tir.const(d_range, "float32")
    )

    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, d, theta, original_max_position_embeddings
    )
    high = tir.if_then_else(low == high, high + 0.001, high)
    inv_freq_mask = tir.const(1, "float32") - tir.max(
        tir.min((d - low) / (high - low), 1.0), 0.0
    ).astype("float32")
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    freq = s * inv_freq
    freq_var = tir.Var("freq", "float32")
    cos_freq = tir.cos(freq_var).astype(dtype)
    sin_freq = tir.sin(freq_var).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def switch_rope_freq_func(rope_scaling: Dict[str, Any]) -> Callable:
    """Return the RoPE inverse frequency computation function based
    on the given RoPE scaling.
    """
    if "rope_type" not in rope_scaling:
        return rope_freq_default
    if rope_scaling["rope_type"] == "gptj":
        return rope_freq_gptj
    if rope_scaling["rope_type"] == "llama3":
        return partial(
            rope_freq_llama3,
            factor=rope_scaling["factor"],
            low_freq_factor=rope_scaling["low_freq_factor"],
            high_freq_factor=rope_scaling["high_freq_factor"],
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        )
    if rope_scaling["rope_type"] == "longrope":
        return partial(
            rope_freq_longrope,
            max_position_embeddings=rope_scaling["max_position_embeddings"],
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        )
    if rope_scaling["rope_type"] == "yarn":
        return partial(
            rope_freq_yarn,
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            scaling_factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
    raise ValueError(f'Unsupported RoPE scaling type: {rope_scaling["rope_type"]}')


# mypy: disable-error-code="attr-defined"


def llama_rope(  # pylint: disable=too-many-arguments
    qkv: Tensor,
    total_seq_len: tir.Var,
    theta: float,
    scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    rope_scaling: Dict[str, Any],
    rotary_dim: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Llama-style RoPE. Given a fused QKV tensor, it returns three tensors, Q, K, and V, where Q
    and K are rotated by RoPE while V remains unchanged.

    Parameters
    ----------
    qkv : Tensor
        The fused QKV tensor of shape: [batch_size, seq_len, #q_heads + #kv_heads * 2, head_dim]

    total_seq_len : tir.Var
        The total sequence length after being concatenated with KVCache. It is used to compute the
        offset of RoPE.

    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    rotary_dim : Optional[int]
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.

    Returns
    -------
    q : Tensor
        The query tensor of shape [batch_size, seq_len, #q_heads, head_dim] w/ RoPE applied

    k : Tensor
        The key tensor of shape [batch_size, seq_len, #kv_heads, head_dim] w/ RoPE applied

    v : Tensor
        The value tensor of shape [batch_size, seq_len, #kv_heads, head_dim] w/o RoPE applied
    """
    _, _, fused_heads, head_dim = qkv.shape
    assert fused_heads == num_q_heads + num_kv_heads * 2
    if rotary_dim is None:
        rotary_dim = head_dim
    dtype = qkv.dtype
    scale = tir.const(scale, dtype)

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Buffer,
        b: tir.Var,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        offset: tir.Var,
    ):
        cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
            (s + offset) * scale, d, rotary_dim, theta, dtype
        )
        cos = cos_freq * x[b, s, h, d]
        if rope_scaling["rope_type"] == "gptj":
            sin = sin_freq * tir.if_then_else(
                d % 2 == 0,
                -x[b, s, h, d + 1],
                x[b, s, h, d - 1],
            )
        else:
            sin = sin_freq * tir.if_then_else(
                d < rotary_dim // 2,
                -x[b, s, h, d + rotary_dim // 2],
                x[b, s, h, d - rotary_dim // 2],
            )
        expr = cos + sin
        for var, value in var_map.items():
            expr = tir.Let(var, value, expr)
        return expr

    @T.prim_func(private=True)
    def fused_rope(  # pylint: disable=too-many-locals
        var_qkv: T.handle,
        var_q: T.handle,
        var_k: T.handle,
        var_v: T.handle,
        total_seq_len: T.int64,
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tir.noalias": T.bool(True),
            }
        )
        batch_size = T.int64()
        seq_len = T.int64()
        qkv = T.match_buffer(var_qkv, (batch_size, seq_len, fused_heads, head_dim), dtype)
        q = T.match_buffer(var_q, (batch_size, seq_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (batch_size, seq_len, num_kv_heads, head_dim), dtype)
        v = T.match_buffer(var_v, (batch_size, seq_len, num_kv_heads, head_dim), dtype)
        for iters in T.grid(batch_size, seq_len, fused_heads, head_dim):
            with T.block("llama_fused_rope"):
                b, s, h, d = T.axis.remap("SSSS", iters)
                if h < num_q_heads:
                    q[b, s, h, d] = T.if_then_else(
                        d < rotary_dim,
                        _rope(qkv, b, s, h, d, total_seq_len - seq_len),
                        qkv[b, s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[b, s, h - num_q_heads, d] = T.if_then_else(
                        d < rotary_dim,
                        _rope(qkv, b, s, h, d, total_seq_len - seq_len),
                        qkv[b, s, h, d],
                    )
                else:
                    v[b, s, h - (num_q_heads + num_kv_heads), d] = qkv[b, s, h, d]

    b, s, _, _ = qkv.shape
    return op.tensor_ir_op(  # pylint: disable=no-member
        fused_rope,
        "llama_rope",
        args=[qkv, total_seq_len],
        out=(
            Tensor.placeholder((b, s, num_q_heads, head_dim), dtype),
            Tensor.placeholder((b, s, num_kv_heads, head_dim), dtype),
            Tensor.placeholder((b, s, num_kv_heads, head_dim), dtype),
        ),
    )


def llama_rope_with_position_map(  # pylint: disable=too-many-arguments
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
    rope_scaling: Dict[str, Any],
    rotary_dim: Optional[int] = None,
):
    """Return the TIR function that computes Llama-style RoPE with q position map.

    Parameters
    ----------
    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    head_dim : int
        The number of features on each head.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    dtype : str
        The dtype of qkv data.

    rotary_dim : int
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.
    """
    fused_heads = num_q_heads + num_kv_heads * 2
    if rotary_dim is None:
        rotary_dim = head_dim
    scale = tir.const(scale, "float32")
    is_longrope_scaling = rope_scaling.get("rope_type") == "longrope"

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Buffer,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        pos: tir.Var,
        ext_factors: Optional[T.Buffer] = None,
    ):
        kwargs = {}
        if ext_factors:
            kwargs["ext_factors"] = ext_factors
        cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
            pos * scale, d, rotary_dim, theta, "float32", **kwargs
        )
        cos = cos_freq * x[s, h, d].astype("float32")
        if "rope_type" in rope_scaling and rope_scaling["rope_type"] == "gptj":
            sin = sin_freq * tir.if_then_else(
                d % 2 == 0,
                -x[s, h, d + 1],
                x[s, h, d - 1],
            ).astype("float32")
        else:
            sin = sin_freq * tir.if_then_else(
                d < rotary_dim // 2,
                -x[s, h, d + rotary_dim // 2],
                x[s, h, d - rotary_dim // 2],
            ).astype("float32")
        expr = (cos + sin).astype(dtype)
        for var, value in var_map.items():
            expr = tir.Let(var, value, expr)
        return expr

    @T.prim_func
    def fused_rope(  # pylint: disable=too-many-locals
        var_qkv: T.handle,
        var_position_map: T.handle,
        var_q: T.handle,
        var_k: T.handle,
        var_v: T.handle,
        apply_rope: T.int32,
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tir.noalias": T.bool(True),
            }
        )
        seq_len = T.int32()
        position_map_elem_offset = T.int32()
        qkv = T.match_buffer(var_qkv, (seq_len, fused_heads, head_dim), dtype)
        q = T.match_buffer(var_q, (seq_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (seq_len, num_kv_heads, head_dim), dtype)
        v = T.match_buffer(var_v, (seq_len, num_kv_heads, head_dim), dtype)
        position_map = T.match_buffer(
            var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset
        )
        for iters in T.grid(seq_len, fused_heads, head_dim):
            with T.block("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", iters)
                if h < num_q_heads:
                    q[s, h, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[s, h - num_q_heads, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                else:
                    v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    @T.prim_func
    def fused_rope_longrope_scaling(  # pylint: disable=too-many-locals
        var_qkv: T.handle,
        var_position_map: T.handle,
        var_q: T.handle,
        var_k: T.handle,
        var_v: T.handle,
        ext_factors: T.Buffer((head_dim // 2,), "float32"),  # type: ignore
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tir.noalias": T.bool(True),
            }
        )
        seq_len = T.int64()
        position_map_elem_offset = T.int64()
        qkv = T.match_buffer(var_qkv, (seq_len, fused_heads, head_dim), dtype)
        q = T.match_buffer(var_q, (seq_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (seq_len, num_kv_heads, head_dim), dtype)
        v = T.match_buffer(var_v, (seq_len, num_kv_heads, head_dim), dtype)
        position_map = T.match_buffer(
            var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset
        )
        for iters in T.grid(seq_len, fused_heads, head_dim):
            with T.block("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", iters)
                if h < num_q_heads:
                    q[s, h, d] = T.if_then_else(
                        d < rotary_dim,
                        _rope(
                            qkv,
                            s,
                            h,
                            d,
                            position_map[s],
                            ext_factors if is_longrope_scaling else None,
                        ),
                        qkv[s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[s, h - num_q_heads, d] = T.if_then_else(
                        d < rotary_dim,
                        _rope(
                            qkv,
                            s,
                            h,
                            d,
                            position_map[s],
                            ext_factors if is_longrope_scaling else None,
                        ),
                        qkv[s, h, d],
                    )
                else:
                    v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    if is_longrope_scaling:
        return fused_rope_longrope_scaling
    return fused_rope
