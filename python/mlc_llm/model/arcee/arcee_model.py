"""
Implementation for Arcee architecture.
"""

import dataclasses
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ArceeConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Arcee model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool = False
    position_embedding_base: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    tensor_parallel_shards: int = 1
    pipeline_parallel_stages: int = 1
    max_batch_size: int = 1
    disaggregation: bool = False
    hidden_act: str = "relu2"
    mlp_bias: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):  # pylint: disable=too-many-branches
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        
        # Handle YARN rope scaling
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", ""))
            if rope_type != "yarn":
                logger.warning(
                    "Arcee model expects YARN rope scaling, got %s. Continuing anyway.",
                    rope_type
                )

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if (
            self.pipeline_parallel_stages <= 0
            or self.pipeline_parallel_stages > self.num_hidden_layers
        ):
            raise ValueError(
                f'Invalid "pipeline_parallel_stages" value ({self.pipeline_parallel_stages}). '
            )
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
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


def relu2(x: Tensor) -> Tensor:
    """ReLU^2 activation function: relu(x)^2"""
    relu_x = op.relu(x)
    return relu_x * relu_x


class ArceeMLP(nn.Module):
    def __init__(self, config: ArceeConfig):
        super().__init__()
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, 
            config.hidden_size, 
            bias=config.mlp_bias
        )
        self.hidden_act = config.hidden_act

    def forward(self, x: Tensor):
        if self.hidden_act == "relu2":
            return self.down_proj(relu2(self.up_proj(x)))
        else:
            # Fallback to standard activations if needed
            raise ValueError(f"Unsupported activation function: {self.hidden_act}")


class ArceeEmbedding(nn.Embedding):
    """The embedding module that can be shared with the final lm_head."""

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class ArceeAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: ArceeConfig):
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        assert (
            config.num_key_value_heads % config.tensor_parallel_shards == 0
        ), f"num_kv_heads({config.num_key_value_heads}) must be divisible by tensor_parallel_shards"
        assert (
            config.num_key_value_heads >= config.tensor_parallel_shards
        ), f"Too large tensor_parallel_shards, must be smaller than {config.num_key_value_heads}"
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim, 
            config.hidden_size, 
            bias=config.attention_bias
        )

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        # QKV Projection
        qkv = self.qkv_proj(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_q_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class ArceeDecoderLayer(nn.Module):
    def __init__(self, config: ArceeConfig):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = ArceeAttention(config)
        self.mlp = ArceeMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))
            # For Arcee, up_proj doesn't have gate, so shard differently
            _set(self.mlp.up_proj, tp.ShardSingleDim("_shard_mlp_up", dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class ArceeModel(nn.Module):
    def __init__(self, config: ArceeConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = ArceeEmbedding("vocab_size", config.hidden_size)
        self.layers = nn.ModuleList(
            [ArceeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.num_layers_per_stage = (
            config.num_hidden_layers + config.pipeline_parallel_stages - 1
        ) // config.pipeline_parallel_stages

        # Compute pipeline layer partition.
        layers_per_stage = (
            config.num_hidden_layers + config.pipeline_parallel_stages - 1
        ) // config.pipeline_parallel_stages
        self.layer_partition = [
            i * layers_per_stage for i in range(config.pipeline_parallel_stages)
        ] + [config.num_hidden_layers]

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            if layer_id != 0 and layer_id in self.layer_partition:
                hidden_states = op_ext.pipeline_stage_boundary(hidden_states)
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class ArceeForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: ArceeConfig):
        self.model = ArceeModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, "vocab_size", bias=False)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.rope_scaling = config.rope_scaling
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        if self.tie_word_embeddings:
            self.model.embed_tokens.weight.to(dtype=dtype)

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def batch_select_last_hidden_states(self, hidden_states: Tensor, logit_positions: Tensor):
        op_ext.configure()
        return op.take(hidden_states, logit_positions, axis=1)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        op_ext.configure()

        # Determine RoPE mode and scaling
        rope_mode = RopeMode.NORMAL
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", ""))
            if rope_type == "yarn":
                rope_mode = RopeMode.NORMAL  # YARN uses normal mode with scaling factors
            else:
                logger.warning("Unsupported RoPE scaling type for Arcee: %s", rope_type)

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
            rope_mode=rope_mode,
            rope_scale=1,
            rope_theta=self.rope_theta,
            layer_partition=self.model.layer_partition,
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
            "prefill": {
                "input_embed": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor(
                    [1, 1, self.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor(
                    ["batch_size", "seq_len", self.hidden_size], self.dtype
                ),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(
                    ["batch_size", 1, self.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_select_last_hidden_states": {
                "hidden_states": nn.spec.Tensor(
                    ["batch_size", "seq_len", self.hidden_size], self.dtype
                ),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
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