"""Attention components for the LFM3 architecture."""

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm_tirx import tirx

from mlc_llm.model.lfm2.lfm2_model import Lfm2Attention
from mlc_llm.nn import PagedKVCache

from .lfm3_config import Lfm3Config


def _apply_rope(x: Tensor, positions: Tensor, theta: float, fraction: float) -> Tensor:
    """Rotate paired halves using full-head frequencies and zero-tail partial RoPE."""
    head_dim = x.shape[2]
    rotated_pairs = int(fraction * head_dim) // 2

    def compute(x_in: te.Tensor, pos: te.Tensor):
        def rotate(i, h, d):
            pair = d % (head_dim // 2)
            angle = pos[i].astype("float32") / tirx.power(
                tirx.const(theta, "float32"), (2 * pair).astype("float32") / head_dim
            )
            partner = tirx.if_then_else(
                d < head_dim // 2, -x_in[i, h, d + head_dim // 2], x_in[i, h, d - head_dim // 2]
            )
            # HF casts cos/sin to the activation dtype before applying the rotation.
            cos = tirx.cos(angle).astype(x_in.dtype)
            sin = tirx.sin(angle).astype(x_in.dtype)
            return tirx.if_then_else(
                pair < rotated_pairs, x_in[i, h, d] * cos + partner * sin, x_in[i, h, d]
            )

        return te.compute(x_in.shape, rotate, name="lfm3_rope")

    return op.tensor_expr_op(compute, "lfm3_rope", [x, positions])


class Lfm3Attention(Lfm2Attention):
    """LFM2 query/key normalization with per-head output gates and per-layer RoPE."""

    def __init__(self, config: Lfm3Config, layer_type: str):
        super().__init__(config)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        rope = config.rope_parameters[layer_type]
        self.rope_theta = rope["rope_theta"]
        self.partial_rotary_factor = rope.get("partial_rotary_factor", 1.0)

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        query_positions: Tensor,
    ):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        qkv = op.reshape(self.qkv_proj(hidden_states), (b, s, h_q + 2 * h_kv, d))
        q, k, v = op.split(qkv, [h_q, h_q + h_kv], axis=2)
        qk = op.reshape(
            op.concat([self.q_layernorm(q), self.k_layernorm(k)], dim=2),
            (b * s, h_q + h_kv, d),
        )
        qk = _apply_rope(qk, query_positions, self.rope_theta, self.partial_rotary_factor)
        qkv = op.concat([op.reshape(qk, (b, s, h_q + h_kv, d)), v], dim=2)
        output = paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, h_q, sm_scale=d**-0.5)
        gate = op.reshape(op.sigmoid(self.gate_proj(hidden_states)), (b, s, h_q, 1))
        return self.out_proj(op.reshape(output * gate, (b, s, h_q * d)))
