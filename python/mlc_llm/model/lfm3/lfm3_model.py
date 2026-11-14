"""Attention components for the LFM3 architecture."""

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm_tirx import tirx

from mlc_llm.model.lfm2.lfm2_model import (
    Lfm2Attention,
    Lfm2Embedding,
    Lfm2ForCausalLM,
    Lfm2MLP,
    Lfm2ShortConv,
    _causal_depthwise_conv1d,
)
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.rnn_state import RNNState

from .lfm3_config import Lfm3Config


def _apply_rope(x: Tensor, positions: Tensor, theta: float, fraction: float) -> Tensor:
    """Rotate paired halves using full-head frequencies and zero-tail partial RoPE."""
    head_dim = x.shape[2]
    rotated_pairs = int(fraction * head_dim) // 2

    def compute(x_in: te.Tensor, pos: te.Tensor):
        def rotate(i, h, d):
            pair = d % (head_dim // 2)
            inv_freq = tirx.const(1.0, "float32") / tirx.power(
                tirx.const(theta, "float32"), (2 * pair).astype("float32") / head_dim
            )
            angle = pos[i].astype("float32") * inv_freq
            partner = tirx.if_then_else(
                d < head_dim // 2, -x_in[i, h, d + head_dim // 2], x_in[i, h, d - head_dim // 2]
            )
            # HF casts cos/sin to the activation dtype before applying the rotation.
            cos = tirx.cos(angle).astype(x_in.dtype).astype("float32")
            sin = tirx.sin(angle).astype(x_in.dtype).astype("float32")
            return tirx.if_then_else(
                pair < rotated_pairs,
                (x_in[i, h, d].astype("float32") * cos + partner.astype("float32") * sin).astype(
                    x_in.dtype
                ),
                x_in[i, h, d],
            )

        return te.compute(x_in.shape, rotate, name="lfm3_rope")

    return op.tensor_expr_op(compute, "lfm3_rope", [x, positions])


class Lfm3Attention(Lfm2Attention):
    """LFM2 query/key normalization with per-head output gates and per-layer RoPE."""

    def __init__(self, config: Lfm3Config, layer_type: str):
        super().__init__(config)
        self.qkv_proj.out_dtype = "float32"
        self.out_proj.out_dtype = "float32"
        self.gate_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads, bias=False, out_dtype="float32"
        )
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
        qkv = op.reshape(
            self.qkv_proj(hidden_states).astype(hidden_states.dtype), (b, s, h_q + 2 * h_kv, d)
        )
        q, k, v = op.split(qkv, [h_q, h_q + h_kv], axis=2)
        qk = op.reshape(
            op.concat([self.q_layernorm(q), self.k_layernorm(k)], dim=2),
            (b * s, h_q + h_kv, d),
        )
        qk = _apply_rope(qk, query_positions, self.rope_theta, self.partial_rotary_factor)
        qkv = op.concat([op.reshape(qk, (b, s, h_q + h_kv, d)), v], dim=2)
        output = paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, h_q, sm_scale=d**-0.5)
        gate = op.reshape(
            op.sigmoid(self.gate_proj(hidden_states).astype(hidden_states.dtype)), (b, s, h_q, 1)
        )
        return self.out_proj(op.reshape(output * gate, (b, s, h_q * d))).astype(hidden_states.dtype)


class Lfm3MLP(Lfm2MLP):
    """Accumulate projections in float32 while retaining the activation dtype."""

    def __init__(self, config: Lfm3Config):
        super().__init__(config)
        self.gate_up_proj.out_dtype = "float32"
        self.down_proj.out_dtype = "float32"

    def forward(self, x: Tensor):
        gate, up = op.split(self.gate_up_proj(x).astype(x.dtype), 2, axis=-1)
        return self.down_proj(op.silu(gate) * up).astype(x.dtype)


class Lfm3ShortConv(Lfm2ShortConv):
    """LFM2 convolution with float32 projection accumulation."""

    def __init__(self, config: Lfm3Config, layer_id: int):
        super().__init__(config, layer_id)
        self.in_proj.out_dtype = "float32"
        self.out_proj.out_dtype = "float32"

    def forward(self, hidden_states: Tensor, state: RNNState):
        b, _, d = hidden_states.shape
        gate_b, gate_c, x = op.split(
            self.in_proj(hidden_states).astype(hidden_states.dtype), 3, axis=-1
        )
        conv_state = state.get(self.conv_layer_id, 0, (b, self.kernel_size - 1, d), self.dtype)
        conv_out, conv_state = _causal_depthwise_conv1d(gate_b * x, conv_state, self.conv.weight)
        state = state.set(self.conv_layer_id, 0, conv_state)
        return self.out_proj(gate_c * conv_out).astype(hidden_states.dtype), state


class Lfm3DecoderLayer(nn.Module):
    def __init__(self, config: Lfm3Config, layer_type: str, category_id: int):
        self.is_attention_layer = layer_type != "conv"
        self.category_id = category_id
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(config, layer_type)
        else:
            self.conv = Lfm3ShortConv(config, category_id)
        self.feed_forward = Lfm3MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(
                config.hidden_size, -1, config.norm_eps, bias=False
            )
            self.post_ffn_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)

    def forward(self, hidden_states, paged_kv_cache, state, positions):
        out = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            out = self.self_attn(out, paged_kv_cache, self.category_id, positions)
        else:
            out, state = self.conv(out, state)
        if self.use_sandwich_norm:
            out = self.post_operator_norm(out)
        hidden_states = hidden_states + out
        out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            out = self.post_ffn_norm(out)
        return hidden_states + out, state


class Lfm3Model(nn.Module):
    def __init__(self, config: Lfm3Config):
        self.embed_tokens = Lfm2Embedding(config.vocab_size, config.hidden_size)
        layers = []
        counts = {"attention": 0, "conv": 0}
        for layer_type in config.layer_types:
            category = "conv" if layer_type == "conv" else "attention"
            layers.append(Lfm3DecoderLayer(config, layer_type, counts[category]))
            counts[category] += 1
        self.layers = nn.ModuleList(layers)
        self.embedding_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache, state: RNNState):
        positions = paged_kv_cache.get_query_positions(inputs.shape[0] * inputs.shape[1])
        hidden_states = inputs
        for layer in self.layers:
            hidden_states, state = layer(hidden_states, paged_kv_cache, state, positions)
        return self.embedding_norm(hidden_states), state


class Lfm3ForCausalLM(Lfm2ForCausalLM):
    def __init__(self, config: Lfm3Config):
        # Reuse LFM2's hybrid entry points and recurrent-state layout.
        self.model = Lfm3Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_layers = config.num_attention_layers
        self.num_conv_layers = config.num_conv_layers
        self.conv_kernel_size = config.conv_L_cache
        self.vocab_size = config.vocab_size
        self.dtype = "float32"
        self.attn_kind = [
            "mha_sliding" if t == "sliding_attention" else "mha"
            for t in config.layer_types
            if t != "conv"
        ]
        self.sliding_window = config.sliding_window

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind=self.attn_kind,
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            layer_sliding_window_size=self.sliding_window,
            num_hidden_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=1,
            dtype=self.dtype,
        )
