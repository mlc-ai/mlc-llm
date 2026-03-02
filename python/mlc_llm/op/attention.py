"""Operators enabled by external modules."""

import tvm
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.support import logging

from . import extern as _extern

logger = logging.getLogger(__name__)


WARN_FLASHINFER_GROUP_SIZE = False
WARN_FLASHINFER_HEAD_DIM = False


def attention(  # pylint: disable=invalid-name,too-many-locals,too-many-statements,too-many-arguments, unused-argument
    q: nn.Tensor,
    k: nn.Tensor,
    v: nn.Tensor,
    casual_mask: nn.Tensor,
    attn_score_scaling_factor: float = 1.0,
    qk_dtype: str = None,
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
        attn = q @ k^T / sqrt(d) * attn_score_scaling_factor  # [b, h, s, t]
        attn = softmax_with_mask(attn, casual_mask, axis=-1)
        o = attn @ v  # [b, h, s, d]
        o -> [b, s, h * d]

    --- Other params ---
    qk_dtype: if set, `matmul(Q, K, out_dtype=qk_dtype)`, (otherwise use `q.dtype` as `out_dtype`).
        For FlashInfer, if "float32", sets `allow_fp16_qk_reduction` to False; otherwise no effect.
    """
    assert q.ndim == 4 and k.ndim in [3, 4] and v.ndim in [3, 4]
    b, s, h_q, d = q.shape
    t, h_kv, _ = k.shape[-3:]
    group_size = h_q // h_kv

    def _fallback():
        from tvm.relax.frontend.nn.llm.kv_cache import (  # pylint: disable=import-outside-toplevel
            _attention_sequence_prefill,
        )

        nonlocal q, k, v, qk_dtype
        if k.ndim == 3:
            k = op.reshape(k, [b, t, h_kv, d])
        if v.ndim == 3:
            v = op.reshape(v, [b, t, h_kv, d])
        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=2)
            v = v.repeat(h_q // h_kv, axis=2)

        target = tvm.target.Target("cuda")
        attn_output, _ = op.tensor_ir_op(
            _attention_sequence_prefill(  # pylint: disable=no-value-for-parameter
                h_kv=h_kv,
                h_q=h_q,
                d=d,
                dtype=q.dtype,
                target=target,
                sm_scale=attn_score_scaling_factor / (d**0.5),
            ),
            "sequence_prefill",
            [q, k, v],
            [
                Tensor.placeholder([b, s, h_q, d], q.dtype),
                Tensor.placeholder([b, s, h_q], q.dtype),
            ],
        )

        output = op.reshape(attn_output, shape=(b, s, h_q * d))
        return output

    # FlashInfer Implementation
    if (
        _extern.get_store().flashinfer
        and attn_score_scaling_factor == 1.0
        and q.dtype == "float16"
        and k.dtype == "float16"
        and v.dtype == "float16"
    ):
        if group_size not in [1, 4, 6, 8]:
            global WARN_FLASHINFER_GROUP_SIZE  # pylint: disable=global-statement
            if not WARN_FLASHINFER_GROUP_SIZE:
                WARN_FLASHINFER_GROUP_SIZE = True
                logger.warning(
                    "FlashInfer only supports group size in [1, 4, 6, 8], but got %d. Skip and "
                    "fallback to default implementation.",
                    group_size,
                )
            return _fallback()
        if d not in [128]:
            global WARN_FLASHINFER_HEAD_DIM  # pylint: disable=global-statement
            if not WARN_FLASHINFER_HEAD_DIM:
                WARN_FLASHINFER_HEAD_DIM = True
                logger.warning(
                    "FlashInfer only supports head_dim in [128], but got %d. Skip and fallback to "
                    "default implementation.",
                    d,
                )
            return _fallback()
        rope_theta = 0.0
        rope_scale = 1.0
        qkv_layout = 0  # "NHD", N for seq_len, H for num_heads, D for head_dim
        rotary_mode = 0  # "kNone"
        casual = 1  # True
        fp16_qk = 1  # True
        if qk_dtype == "float32":
            fp16_qk = 0  # False

        # 32MB scratchpad
        scratch = op.empty([8192 * 1024], dtype="float32")  # pylint: disable=no-member

        def _decode():
            return op.extern(
                name="flashinfer.single_decode",
                args=[
                    q,
                    k,
                    v,
                    scratch,
                    qkv_layout,
                    rotary_mode,
                    rope_scale,
                    rope_theta,
                ],
                out=nn.Tensor.placeholder((b, s, h_q * d), dtype="float16"),
            )

        def _prefill():
            return op.extern(
                name="flashinfer.single_prefill",
                args=[
                    q,
                    k,
                    v,
                    scratch,
                    casual,
                    qkv_layout,
                    rotary_mode,
                    fp16_qk,
                    rope_scale,
                    rope_theta,
                ],
                out=nn.Tensor.placeholder((b, s, h_q * d), dtype="float16"),
            )

        if isinstance(s, int) and s == 1:
            func = "decode"
        else:
            func = "prefill"
        return {
            "decode": _decode,
            "prefill": _prefill,
        }[func]()

    # Fallback Implementation
    return _fallback()
