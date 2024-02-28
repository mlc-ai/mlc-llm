"""Operators for positional embeddings, e.g. RoPE."""

from typing import Optional, Tuple

from tvm import tir
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T
from tvm.target import Target

from ..support.max_thread_check import (
    check_thread_limits,
    get_max_num_threads_per_block,
)

# pylint: disable=invalid-name


def rope_freq(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
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
    """
    freq = s / tir.power(theta, d * 2 % d_range / tir.const(d_range, "float32"))
    cos_freq = tir.cos(freq).astype(dtype)
    sin_freq = tir.sin(freq).astype(dtype)
    return cos_freq, sin_freq


# mypy: disable-error-code="attr-defined"


def llama_rope(  # pylint: disable=too-many-arguments
    qkv: Tensor,
    total_seq_len: tir.Var,
    theta: float,
    num_q_heads: int,
    num_kv_heads: int,
    scale: float = 1.0,
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
        cos_freq, sin_freq = rope_freq((s + offset) * scale, d, rotary_dim, theta, dtype)
        cos = cos_freq * x[b, s, h, d]
        sin = sin_freq * tir.if_then_else(
            d < rotary_dim // 2,
            -x[b, s, h, d + rotary_dim // 2],
            x[b, s, h, d - rotary_dim // 2],
        )
        return cos + sin

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
    rotary_dim: int = None,
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
    scale = tir.const(scale, dtype)

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Buffer,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        pos: tir.Var,
    ):
        cos_freq, sin_freq = rope_freq(pos * scale, d, rotary_dim, theta, dtype)
        cos = cos_freq * x[s, h, d]
        sin = sin_freq * tir.if_then_else(
            d < rotary_dim // 2,
            -x[s, h, d + rotary_dim // 2],
            x[s, h, d - rotary_dim // 2],
        )
        return cos + sin

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
        seq_len = T.int64()
        qkv = T.match_buffer(var_qkv, (seq_len, fused_heads, head_dim), dtype)
        q = T.match_buffer(var_q, (seq_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (seq_len, num_kv_heads, head_dim), dtype)
        v = T.match_buffer(var_v, (seq_len, num_kv_heads, head_dim), dtype)
        position_map = T.match_buffer(var_position_map, (seq_len,), "int32")
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

    return fused_rope


# pylint: disable=line-too-long,too-many-arguments,too-many-nested-blocks,invalid-name


def llama_inplace_rope(
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
    target: Target,  # pylint: disable=unused-argument
    rotary_dim: Optional[int] = None,
):
    """Return the TIR function that inplace computes Llama-style RoPE with q position offset.

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

    target : Target
        The target to build the model to.

    rotary_dim : Optional[int]
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    VEC_SIZE = 4
    bdx = (head_dim + VEC_SIZE - 1) // VEC_SIZE  # T.ceildiv(head_dim, VEC_SIZE)
    bdy = 32
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    # TODO(mlc-team): Check correctness after `bdy` backoff
    while bdx * bdy > max_num_threads_per_block and bdy > 1:
        bdy //= 2
    check_thread_limits(target, bdx=bdx, bdy=bdy, bdz=1, gdz=1)

    def _rope(
        x: T.Buffer,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        rope_offset: tir.Var,
        instance_offset: tir.Var,
    ):
        cos_freq, sin_freq = rope_freq((s + rope_offset) * scale, d, rotary_dim, theta, dtype)
        cos = cos_freq * x[s + instance_offset, h, d]
        sin = sin_freq * tir.if_then_else(
            d < rotary_dim // 2,
            -x[s + instance_offset, h, d + rotary_dim // 2],
            x[s + instance_offset, h, d - rotary_dim // 2],
        )
        return cos + sin

    # fmt: off
    @T.prim_func
    def tir_rotary(  # pylint: disable=too-many-locals
        var_q: T.handle,
        var_k: T.handle,
        var_append_len_indptr: T.handle,
        var_rope_offsets: T.handle,
        _0: T.int32,
        _1: T.int32,
        _2: T.int32,
        _3: T.int32,
        _4: T.int32,
        _5: T.float32,
        _6: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        total_len = T.int32()
        batch_size = T.int32()
        q = T.match_buffer(var_q, (total_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (total_len, num_kv_heads, head_dim), dtype)
        rope_offsets = T.match_buffer(var_rope_offsets, (batch_size,), "int32")
        append_len_indptr = T.match_buffer(var_append_len_indptr, (batch_size + 1,), "int32")
        with T.block():
            for b_h in T.thread_binding(batch_size * (num_q_heads + num_kv_heads), thread="blockIdx.x"):
                b: T.int32 = b_h // (num_q_heads + num_kv_heads)
                h: T.int32 = b_h % (num_q_heads + num_kv_heads)
                instance_offset: T.int32 = append_len_indptr[b]
                rope_offset: T.int32 = rope_offsets[b]
                append_len: T.int32 = append_len_indptr[b + 1] - append_len_indptr[b]
                for s0 in range(T.ceildiv(append_len, bdy)):
                    for s1 in T.thread_binding(bdy, thread="threadIdx.y"):
                        for d0 in T.thread_binding(bdx, thread="threadIdx.x"):
                            for d1 in T.vectorized(VEC_SIZE):
                                s: T.int32 = s0 * bdy + s1
                                d: T.int32 = d0 * VEC_SIZE + d1
                                if s < append_len and d < rotary_dim:
                                    if h < num_q_heads:
                                        q[s + instance_offset, h, d] = _rope(q, s, h, d, rope_offset, instance_offset)
                                    else:
                                        k[s + instance_offset, h - num_q_heads, d] = _rope(k, s, h - num_q_heads, d, rope_offset, instance_offset)
    return tir_rotary


# pylint: enable=line-too-long,too-many-arguments,too-many-nested-blocks,invalid-name
