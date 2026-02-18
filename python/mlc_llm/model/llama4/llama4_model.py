"""
Implementation for Llama4 architecture.
"""

import dataclasses
from typing import Any, Dict, Optional

import tvm
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm import position_embedding

from mlc_llm import op as op_ext
from mlc_llm.model.qwen3.qwen3_model import ACT2FN
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Llama4TextConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Text portion of the Llama model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    use_qk_norm: bool
    interleave_moe_layer_step: int
    num_experts_per_tok: int
    num_local_experts: int
    hidden_act: str
    tie_word_embeddings: bool = False
    position_embedding_base: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    num_key_value_heads: int = 0
    head_dim: int = 0
    attn_scale: float = 0.1
    floor_scale: int = 8192
    vocab_size: int = 202048
    attention_bias: bool = False
    attn_temperature_tuning: bool = True
    no_rope_layers: list[int] = None
    no_rope_layer_interval: int = 4
    moe_layers: list[int] = None

    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):  # pylint: disable=too-many-branches
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.rope_scaling is not None:
            if "rope_type" not in self.rope_scaling:
                self.rope_scaling = None
            else:
                assert (
                    self.rope_scaling["rope_type"] == "llama3"
                ), f"Unsupported RoPE scaling type {self.rope_scaling['rope_type']} for Llama"

        # Define which layers to avoid RoPE
        if self.no_rope_layers == []:
            self.no_rope_layers = None

        default_no_rope_layers = [
            int((layer_idx + 1) % self.no_rope_layer_interval != 0)
            for layer_idx in range(self.num_hidden_layers)
        ]

        self.no_rope_layers = self.no_rope_layers if self.no_rope_layers else default_no_rope_layers

        # Define which layers to apply MoE
        self.moe_layers = (
            self.moe_layers
            if self.moe_layers is not None
            else list(
                range(
                    self.interleave_moe_layer_step - 1,
                    self.num_hidden_layers,
                    self.interleave_moe_layer_step,
                )
            )
        )


@dataclasses.dataclass
class Llama4Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Llama model."""

    text_config: Llama4TextConfig
    tensor_parallel_shards: int = 1
    context_window_size: int = 0
    pipeline_parallel_stages: int = 1
    prefill_chunk_size: int = 0
    max_batch_size: int = 1
    disaggregation: bool = False
    max_position_embeddings = 4096 * 32
    vocab_size: int = 202048

    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        text_config_dict: Dict[str, Any]
        if isinstance(self.text_config, ConfigBase):
            text_config_dict = dataclasses.asdict(self.text_config)
        else:
            text_config_dict = dict(self.text_config)

        for k, v in text_config_dict.pop("kwargs", {}).items():
            text_config_dict[k] = v

        self.text_config = Llama4TextConfig.from_dict(text_config_dict)  # type: ignore

        if self.context_window_size == 0:
            # Fall back to max_position_embeddings

            self.context_window_size = self.max_position_embeddings
            logger.info(
                "%s not found in config.json. Falling back to %s (%d)",
                bold("context_window_size"),
                bold("max_position_embeddings"),
                self.context_window_size,
            )

        if self.text_config.num_key_value_heads == 0:
            self.text_config.num_key_value_heads = self.text_config.num_attention_heads
        if self.text_config.head_dim == 0:
            self.text_config.head_dim = (
                self.text_config.hidden_size // self.text_config.num_attention_heads
            )
        assert self.text_config.num_attention_heads % self.text_config.num_key_value_heads == 0
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)


# pylint: disable=invalid-name,missing-docstring


class Llama4TextMLP(nn.Module):
    def __init__(self, config: Llama4Config):
        super().__init__()
        if config.text_config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.text_config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = (
            config.text_config.intermediate_size // config.tensor_parallel_shards
        )
        self.gate_up_proj = nn.Linear(
            in_features=config.text_config.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, config.text_config.hidden_size, bias=False
        )

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        inter_out = op.silu(x1) * x2

        return self.down_proj(inter_out)


class LlamaEmbedding(nn.Embedding):
    """The embedding module that can be shared with the final lm_head. From Qwen2Embedding."""

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class Llama4TextL2Norm(nn.Module):
    def __init__(self, eps, hidden_size):
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x):
        weight = op.ones((self.hidden_size,), dtype=x.dtype)
        return op.rms_norm(x, weight=weight, axes=[-1], epsilon=self.eps)


class Llama4TextAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Llama4Config, layer_idx):
        self.head_dim = config.text_config.head_dim
        self.attn_scale = config.text_config.attn_scale
        self.floor_scale = config.text_config.floor_scale
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_kv_heads = config.text_config.num_key_value_heads
        self.num_q_heads = config.text_config.num_attention_heads // config.tensor_parallel_shards
        assert config.text_config.num_key_value_heads % config.tensor_parallel_shards == 0, (
            f"num_kv_heads({config.text_config.num_key_value_heads}) must be divisible by "
            f"tensor_parallel_shards"
        )

        assert config.text_config.num_key_value_heads >= config.tensor_parallel_shards, (
            f"Too large tensor_parallel_shards, must be smaller than "
            f"{config.text_config.num_key_value_heads}"
        )
        self.num_kv_heads = config.text_config.num_key_value_heads // config.tensor_parallel_shards
        self.q_proj = nn.Linear(
            config.text_config.hidden_size,
            self.num_q_heads * self.head_dim,
            bias=config.text_config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.text_config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.text_config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.text_config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.text_config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim,
            config.text_config.hidden_size,
            bias=config.text_config.attention_bias,
        )

        self.attn_temperature_tuning = config.text_config.attn_temperature_tuning
        self.use_rope = config.text_config.no_rope_layers[layer_idx]

        self.layer_idx = layer_idx

        self.rope_theta = config.text_config.rope_theta
        self.rope_scaling = config.text_config.rope_scaling
        self.rope_scaling["rope_type"] = "llama4"

        self.use_qk_norm = config.text_config.use_qk_norm
        self.rms_norm_eps = config.text_config.rms_norm_eps

        self.q_norm = Llama4TextL2Norm(self.rms_norm_eps, self.head_dim)
        self.k_norm = Llama4TextL2Norm(self.rms_norm_eps, self.head_dim)

    def forward(  # pylint: disable=too-many-locals
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        cache_position,
    ):
        d, h_q = self.head_dim, self.num_q_heads
        b, s, _ = hidden_states.shape
        # QKV Projection
        query_states = op.reshape(self.q_proj(hidden_states), (b, s, -1, d))
        key_states = op.reshape(self.k_proj(hidden_states), (b, s, -1, d))
        value_states = op.reshape(self.v_proj(hidden_states), (b, s, -1, d))

        if self.use_rope:
            qkv = op.concat([query_states, key_states, value_states], dim=2)

            apply_rope = tvm.tir.IntImm("int64", 1)

            rotary_emb = position_embedding.llama4_rope_with_position_map(
                theta=self.rope_theta,
                scale=1.0,
                head_dim=self.head_dim,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                dtype=query_states.dtype,
                rope_scaling=self.rope_scaling,
            )

            query_states, key_states, value_states = op.tensor_ir_op(
                rotary_emb,
                "llama4_rope_with_position_map",
                args=[op.squeeze(qkv, axis=0), cache_position, apply_rope],
                out=(
                    Tensor.placeholder((s, h_q, d), query_states.dtype),
                    Tensor.placeholder((s, self.num_kv_heads, d), query_states.dtype),
                    Tensor.placeholder((s, self.num_kv_heads, d), query_states.dtype),
                ),
            )
            query_states = query_states.reshape(b, s, h_q, d)
            key_states = key_states.reshape(b, s, self.num_kv_heads, d)
            value_states = value_states.reshape(b, s, self.num_kv_heads, d)

        if self.use_qk_norm and self.use_rope:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                op.log(
                    op.floor(
                        (op.astype(cache_position, query_states.dtype) + 1.0) / self.floor_scale
                    )
                    + 1.0
                )
                * self.attn_scale
                + 1.0
            )

            attn_scales = op.broadcast_to(attn_scales.reshape(1, s, 1, 1), (b, s, 1, 1))
            query_states = query_states * attn_scales

        qkv = op.concat([query_states, key_states, value_states], dim=2)

        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_q_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class Llama4TextExperts(nn.Module):
    def __init__(self, config: Llama4Config):
        self.num_experts = config.text_config.num_local_experts
        self.intermediate_size = (
            config.text_config.intermediate_size // config.tensor_parallel_shards
        )
        self.hidden_size = config.text_config.hidden_size
        self.expert_dim = self.intermediate_size

        self.gate_up_proj = nn.Parameter(
            shape=(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(shape=(self.num_experts, self.expert_dim, self.hidden_size))
        self.act_fn = ACT2FN[config.text_config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(self.gate_up_proj.shape[0], -1, self.hidden_size)
        gate_up = op.matmul(hidden_states, self.gate_up_proj)
        gate, up = op.chunk(gate_up, chunks=2, dim=-1)
        next_states = op.matmul((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.reshape(-1, self.hidden_size)
        return next_states


class Llama4Router(nn.Module):
    def __init__(self, config: Llama4Config):
        self.num_experts = config.text_config.num_local_experts
        self.top_k = config.text_config.num_experts_per_tok
        self.intermediate_size = self.num_experts // config.tensor_parallel_shards
        self.router = nn.Linear(
            in_features=config.text_config.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
        )

    def forward(self, hidden_states):
        router_logits = self.router(hidden_states)
        router_top_value, router_indices = op_ext.moe_misc.gating_topk(router_logits, self.top_k)

        j_axis = op.arange(0, self.num_experts)
        j_axis = op.unsqueeze(j_axis, 0)
        idx_exp = op.unsqueeze(router_indices, -1)
        mask = op.equal(idx_exp, j_axis)
        val_exp = op.unsqueeze(router_top_value, -1)
        neg_inf = op.full(mask.shape, -1e9, dtype=hidden_states.dtype)
        masked_vals = op.where(mask, val_exp, neg_inf)
        router_scores = op.max(masked_vals, axis=1)

        router_scores = op.sigmoid(router_scores)
        return router_scores, router_logits


class Llama4TextMoe(nn.Module):
    def __init__(self, config: Llama4Config):
        self.top_k = config.text_config.num_experts_per_tok
        self.hidden_dim = config.text_config.hidden_size
        self.num_experts = config.text_config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = Llama4Router(config)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, _ = self.router(hidden_states)

        routed_in = op.broadcast_to(
            hidden_states.reshape(1, *hidden_states.shape),
            [router_scores.shape[1], *hidden_states.shape],
        )
        routed_in = routed_in.reshape(-1, self.hidden_dim)

        routed_in = routed_in * op.permute_dims(router_scores, axes=[1, 0]).reshape(-1, 1)

        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)

        out += op.sum(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]), axis=0)

        return out


class Llama4TextDecoderLayer(nn.Module):
    def __init__(self, config: Llama4Config, layer_idx):
        rms_norm_eps = config.text_config.rms_norm_eps
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.is_moe_layer = layer_idx in config.text_config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(config)

        self.input_layernorm = nn.RMSNorm(
            config.text_config.hidden_size, -1, rms_norm_eps, bias=False
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.text_config.hidden_size, -1, rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                if hasattr(layer, "weight"):
                    layer.weight.attrs["shard_strategy"] = hint
                else:
                    layer.attrs["shard_strategy"] = hint

            _set(self.self_attn.q_proj, tp.ShardSingleDim("_shard_q", dim=0))
            _set(self.self_attn.k_proj, tp.ShardSingleDim("_shard_k", dim=0))
            _set(self.self_attn.v_proj, tp.ShardSingleDim("_shard_v", dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))

            if isinstance(self.feed_forward, Llama4TextMLP):
                i = self.feed_forward.intermediate_size
                _set(
                    self.feed_forward.gate_up_proj,
                    tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0),
                )
                _set(
                    self.feed_forward.down_proj,
                    tp.ShardSingleDim("_shard_mlp_down", dim=1),
                )
            else:
                assert isinstance(self.feed_forward, Llama4TextMoe)
                i = self.feed_forward.shared_expert.intermediate_size
                _set(
                    self.feed_forward.shared_expert.gate_up_proj,
                    tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0),
                )
                _set(
                    self.feed_forward.shared_expert.down_proj,
                    tp.ShardSingleDim("_shard_mlp_down", dim=1),
                )

                j = self.feed_forward.experts.intermediate_size
                _set(
                    self.feed_forward.experts.gate_up_proj,
                    tp.ShardSingleDim("_shard_expert_mlp_up", segs=[j, j], dim=2),
                )
                _set(
                    self.feed_forward.experts.down_proj,
                    tp.ShardSingleDim("_shard_expert_mlp_down", dim=1),
                )

                _set(
                    self.feed_forward.router.router,
                    tp.ShardSingleDim("_shard_router", dim=0),
                )

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        cache_position,
    ):
        out = self.self_attn(
            self.input_layernorm(hidden_states),
            paged_kv_cache,
            layer_id,
            cache_position,
        )
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.feed_forward(self.post_attention_layernorm(hidden_states))

        hidden_states = self._apply_residual(
            op.reshape(out, hidden_states.shape), residual=hidden_states
        )

        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Llama4TextModel(nn.Module):
    def __init__(self, config: Llama4Config):
        assert config.text_config.hidden_size % config.text_config.num_attention_heads == 0
        self.embed_tokens = LlamaEmbedding("vocab_size", config.text_config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Llama4TextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.text_config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.text_config.hidden_size,
            -1,
            config.text_config.rms_norm_eps,
            bias=False,
        )

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        cache_position = paged_kv_cache.get_query_positions(
            input_embed.shape[0] * input_embed.shape[1]
        )

        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id, cache_position)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Llama4ForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Llama4Config):
        self.text_config = config.text_config
        self.model = Llama4TextModel(config)
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        if not self.text_config.tie_word_embeddings:
            self.lm_head = nn.Linear(self.text_config.hidden_size, "vocab_size", bias=False)
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.head_dim = self.text_config.head_dim
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        self.rope_scaling = self.text_config.rope_scaling
        self.rope_theta = self.text_config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.disaggregation = config.disaggregation
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            if self.tensor_parallel_shards > 1:
                logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def batch_forward_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        op_ext.configure()

        hidden_states = self.model(input_embeds, paged_kv_cache)
        return hidden_states

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def get_logits(self, hidden_states: Tensor):
        op_ext.configure()
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def batch_select_last_hidden_states(self, hidden_states: Tensor, logit_positions: Tensor):
        op_ext.configure()
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        hidden_states = op.take(hidden_states, logit_positions, axis=0)
        return hidden_states

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def prefill_to_last_hidden_states(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def decode_to_last_hidden_states(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_prefill_to_last_hidden_states(
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = self.batch_forward_to_last_hidden_states(input_embeds, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_decode_to_last_hidden_states(
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = self.batch_forward_to_last_hidden_states(input_embeds, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_verify_to_last_hidden_states(
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = self.batch_forward_to_last_hidden_states(input_embeds, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            enable_disaggregation=self.disaggregation,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "get_logits": {
                "hidden_states": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_select_last_hidden_states": {
                "hidden_states": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill_to_last_hidden_states": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode_to_last_hidden_states": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
