"""Text decoder for Gemma 4 31B and 26B-A4B.

Architecture: Transformers e7200cd9a98f0c874edcdf8f7263b7d19530b403.
Global KV heads are replicated into a uniform cache; local heads are zero padded.
"""

import dataclasses
import math
from typing import Any

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from mlc_llm import op as op_ext
from mlc_llm.model.gemma.gemma_model import GemmaEmbedding
from mlc_llm.model.model_utils import index_last_token
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support.config import ConfigBase

from .gemma4_audio import Gemma4RMSNorm
from .gemma4_model import Gemma4TextMLP, Gemma4TextRotaryEmbedding
from .gemma4_moe import Gemma4LargeExperts, Gemma4Router


@dataclasses.dataclass
class Gemma4LargeConfig(ConfigBase):
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int = 256
    global_head_dim: int = 512
    vocab_size: int = 262144
    layer_types: list[str] | None = None
    sliding_window: int = 1024
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float | None = 30.0
    rope_parameters: dict | None = None
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_k_eq_v: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_bidirectional_attention: str | None = "vision"
    hidden_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None
    tensor_parallel_shards: int = 1
    context_window_size: int = -1
    prefill_chunk_size: int = -1
    sliding_window_size: int = -1
    max_batch_size: int = 1
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, source):
        values = dict(source)
        text = values.pop("text_config", None)
        if text is not None:
            values.update(text)
        return super().from_dict(values)

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("Invalid Gemma 4 layer types")
        if (
            self.num_kv_shared_layers
            or self.hidden_size_per_layer_input
            or self.use_double_wide_mlp
        ):
            raise ValueError("Large Gemma 4 text requires unshared KV and no per-layer inputs")
        if (
            not self.attention_k_eq_v
            or not self.tie_word_embeddings
            or self.attention_bias
            or self.attention_dropout
        ):
            raise ValueError("Unsupported large Gemma 4 projection configuration")
        if self.use_bidirectional_attention not in (None, "vision"):
            raise ValueError("Only causal text attention is supported")
        if self.tensor_parallel_shards != 1 or self.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError("Gemma 4 requires one shard and tanh GeLU")
        for x in (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_global_key_value_heads,
            self.head_dim,
            self.global_head_dim,
            self.vocab_size,
            self.sliding_window,
        ):
            if type(x) is not int or x <= 0:
                raise ValueError("Gemma 4 dimensions must be positive integers")
        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.num_key_value_heads % self.num_global_key_value_heads
        ):
            raise ValueError("Cache KV replication requires divisible head counts")
        if self.head_dim % 2 or self.global_head_dim % 2:
            raise ValueError("RoPE head dimensions must be even")
        if self.enable_moe_block:
            if (
                any(
                    type(x) is not int or x <= 0
                    for x in (self.num_experts, self.top_k_experts, self.moe_intermediate_size)
                )
                or self.top_k_experts > self.num_experts
            ):
                raise ValueError("Invalid Gemma 4 expert configuration")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            }
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("RMSNorm epsilon must be finite and positive")
        if self.final_logit_softcapping is not None and (
            not math.isfinite(self.final_logit_softcapping) or self.final_logit_softcapping <= 0
        ):
            raise ValueError("Logit softcap must be finite and positive")
        if set(self.rope_parameters) != {"sliding_attention", "full_attention"}:
            raise ValueError("Both local and global RoPE configurations are required")
        for layer_type, expected in (
            ("sliding_attention", "default"),
            ("full_attention", "proportional"),
        ):
            rope = self.rope_parameters[layer_type]
            if not isinstance(rope, dict) or rope.get("rope_type") != expected:
                raise ValueError("Unsupported Gemma 4 RoPE type")
            if set(rope) - {"rope_type", "rope_theta", "partial_rotary_factor"}:
                raise ValueError("Unsupported Gemma 4 RoPE option")
            theta = rope.get("rope_theta", 0)
            factor = rope.get("partial_rotary_factor", 1.0)
            if not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
                raise ValueError("RoPE theta must be finite and positive")
            if (
                not isinstance(factor, (int, float))
                or not math.isfinite(factor)
                or not 0 < factor <= 1
            ):
                raise ValueError("Invalid partial rotary factor")
            dim = self.head_dim if layer_type == "sliding_attention" else self.global_head_dim
            if factor * dim < 2 or int(factor * dim) % 2 or factor * dim != int(factor * dim):
                raise ValueError("Rotary dimensions must be a positive even integer")
            if layer_type == "sliding_attention" and factor != 1:
                raise ValueError("Local default RoPE must rotate the complete head")
        self.context_window_size = (
            self.max_position_embeddings
            if self.context_window_size <= 0
            else self.context_window_size
        )
        self.prefill_chunk_size = (
            min(self.sliding_window, self.context_window_size)
            if self.prefill_chunk_size <= 0
            else min(self.prefill_chunk_size, self.sliding_window)
        )
        self.sliding_window_size = -1

    @property
    def first_kv_shared_layer(self):
        return self.num_hidden_layers

    @property
    def physical_head_dim(self):
        return max(self.head_dim, self.global_head_dim)

    def head_dim_for_layer(self, index):
        return (
            self.global_head_dim if self.layer_types[index] == "full_attention" else self.head_dim
        )

    def dimensions(self, index):
        full = self.layer_types[index] == "full_attention"
        return (
            self.head_dim_for_layer(index),
            self.num_global_key_value_heads if full else self.num_key_value_heads,
            full,
        )


def pack_cache_heads(value, physical_heads, physical_dim):
    """Repeat consecutive KV heads and pad features, preserving GQA groups."""
    batch, length, heads, dim = value.shape
    repeats = physical_heads // heads

    def compute(x):
        return te.compute(
            (batch, length, physical_heads, dim), lambda b, s, h, d: x[b, s, h // repeats, d]
        )

    if repeats != 1:
        value = op.tensor_expr_op(compute, "gemma4_repeat_kv", [value])
    if dim != physical_dim:
        value = op.pad(value, [0, 0, 0, 0, 0, 0, 0, physical_dim - dim])
    return value


class Gemma4LargeAttention(nn.Module):
    def __init__(self, config, index):
        self.config = config
        self.layer_idx = index
        self.head_dim, self.num_kv_heads, self.shared_projection = config.dimensions(index)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        if not self.shared_projection:
            self.v_proj = nn.Linear(
                config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config, index)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(self, hidden, cache, positions):
        batch, length, _ = hidden.shape
        query = op.reshape(
            self.q_proj(hidden), (batch, length, self.config.num_attention_heads, self.head_dim)
        )
        key = op.reshape(self.k_proj(hidden), (batch, length, self.num_kv_heads, self.head_dim))
        value = (
            key
            if self.shared_projection
            else op.reshape(self.v_proj(hidden), (batch, length, self.num_kv_heads, self.head_dim))
        )
        query, key, value = self.q_norm(query), self.k_norm(key), self.v_norm(value)
        query, key = self.rotary_emb(query, key, positions)
        c = self.config
        query = pack_cache_heads(query, c.num_attention_heads, c.physical_head_dim)
        key = pack_cache_heads(key, c.num_key_value_heads, c.physical_head_dim)
        value = pack_cache_heads(value, c.num_key_value_heads, c.physical_head_dim)
        result = cache.attention_with_fused_qkv(
            self.layer_idx,
            op.concat([query, key, value], dim=2),
            c.num_attention_heads,
            sm_scale=1.0,
        )
        if self.head_dim != c.physical_head_dim:
            result = op.split(result, [self.head_dim], axis=-1)[0]
        return self.o_proj(
            op.reshape(
                result, (hidden.shape[0], hidden.shape[1], c.num_attention_heads * self.head_dim)
            )
        )


class Gemma4LargeLayer(nn.Module):
    def __init__(self, config, index):
        self.config = config
        self.self_attn = Gemma4LargeAttention(config, index)
        self.mlp = Gemma4TextMLP(config, index)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            setattr(self, name, Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps))
        self.layer_scalar = nn.Parameter((1,))
        if config.enable_moe_block:
            self.router = Gemma4Router(config)
            self.router.proj.no_quantization = True
            self.experts = Gemma4LargeExperts(config)
            for name in (
                "pre_feedforward_layernorm_2",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
            ):
                setattr(self, name, Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps))

    def forward(self, hidden, cache, positions):
        hidden = hidden + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(hidden), cache, positions)
        )
        residual = hidden
        dense = self.mlp(self.pre_feedforward_layernorm(hidden))
        if self.config.enable_moe_block:
            flat = op.reshape(residual, (-1, self.config.hidden_size))
            _, weights, indices = self.router(flat)
            routed = self.experts(self.pre_feedforward_layernorm_2(flat), indices, weights)
            routed = op.reshape(routed, residual.shape)
            dense = self.post_feedforward_layernorm_1(dense) + self.post_feedforward_layernorm_2(
                routed
            )
        return (residual + self.post_feedforward_layernorm(dense)) * self.layer_scalar


class Gemma4LargeModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.embed_tokens = GemmaEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Gemma4LargeLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, cache):
        positions = cache.get_query_positions(hidden.shape[0] * hidden.shape[1])
        for layer in self.layers:
            hidden = layer(hidden, cache, positions)
        return self.norm(hidden)


class Gemma4LargeForCausalLM(nn.Module):
    def __init__(self, config):
        self.config = config
        self.language_model = Gemma4LargeModel(config)
        self.dtype = "float32"

    def to(self, dtype=None):
        super().to(dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids):
        return self.language_model.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

    def get_logits(self, hidden):
        logits = self.language_model.embed_tokens.lm_head_forward(hidden).astype("float32")
        cap = self.config.final_logit_softcapping
        return op.tanh(logits / cap) * cap if cap is not None else logits

    def prefill(self, input_embed, paged_kv_cache):
        op_ext.configure()
        return self.get_logits(
            index_last_token(self.language_model(input_embed, paged_kv_cache))
        ), paged_kv_cache

    def decode(self, input_embed, paged_kv_cache):
        op_ext.configure()
        return self.get_logits(self.language_model(input_embed, paged_kv_cache)), paged_kv_cache

    def batch_prefill(self, input_embeds, logit_positions, paged_kv_cache):
        op_ext.configure()
        return self.get_logits(
            op.take(self.language_model(input_embeds, paged_kv_cache), logit_positions, axis=1)
        ), paged_kv_cache

    def batch_decode(self, input_embeds, paged_kv_cache):
        return self.decode(input_embeds, paged_kv_cache)

    def batch_verify(self, input_embeds, paged_kv_cache):
        return self.decode(input_embeds, paged_kv_cache)

    def create_paged_kv_cache(
        self,
        max_batch_size,
        max_total_seq_len,
        prefill_chunk_size,
        page_size,
        support_sliding_window,
    ):
        c = self.config
        return PagedKVCache.create_generic(
            attn_kind=["mha" if t == "full_attention" else "mha_sliding" for t in c.layer_types],
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=c.num_hidden_layers,
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            qk_head_dim=c.physical_head_dim,
            v_head_dim=c.physical_head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=10000,
            dtype=self.dtype,
            layer_sliding_window_size=c.sliding_window,
        )

    def get_default_spec(self):
        packed = {"param_mode": "packed", "effect_mode": "none"}
        cache = nn.spec.Object(object_type=PagedKVCache)
        h = self.config.hidden_size
        spec = {"embed": {"input_ids": nn.spec.Tensor(["seq_len"], "int32"), "$": packed}}
        for name, shape in (
            ("prefill", [1, "seq_len", h]),
            ("decode", [1, 1, h]),
            ("batch_prefill", [1, "seq_len", h]),
            ("batch_decode", ["batch_size", 1, h]),
            ("batch_verify", [1, "seq_len", h]),
        ):
            entry = {
                "input_embed" if name in ("prefill", "decode") else "input_embeds": nn.spec.Tensor(
                    shape, self.dtype
                )
            }
            if name == "batch_prefill":
                entry["logit_positions"] = nn.spec.Tensor(["batch_size"], "int32")
            entry.update(paged_kv_cache=cache, **{"$": packed})
            spec[name] = entry
        spec["create_paged_kv_cache"] = dict.fromkeys(
            (
                "max_batch_size",
                "max_total_seq_len",
                "prefill_chunk_size",
                "page_size",
                "support_sliding_window",
            ),
            int,
        )
        spec["create_paged_kv_cache"]["$"] = {"param_mode": "none", "effect_mode": "none"}
        return nn.spec.ModuleSpec.from_raw(spec, self)
