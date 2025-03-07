"""
Implementation for OLMo architecture.
TODO: add docstring
"""

import dataclasses
from functools import partial
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
class OLMoConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the OLMo model."""

    vocab_size: int = None
    hidden_size: int = None
    num_attention_heads: int = None
    num_key_value_heads: int = 0
    head_dim: int = 0
    position_embedding_base: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    intermediate_size: int = None
    hidden_act: str = None
    num_hidden_layers: int = None
    tie_word_embeddings: bool = False
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    pipeline_parallel_stages: int = 1
    max_batch_size: int = 1
    clip_qkv: float = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):  # pylint: disable=too-many-branches
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0

        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000

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

        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d", bold("prefill_chunk_size"), min(self.context_window_size, 8192)
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

        if (
            self.pipeline_parallel_stages <= 0
            or self.pipeline_parallel_stages > self.num_hidden_layers
        ):
            raise ValueError(
                f'Invalid "pipeline_parallel_stages" value({self.pipeline_parallel_stages}). '
            )

        if self.clip_qkv is not None:
            if self.clip_qkv <= 0:
                raise ValueError(f"'clip_qkv'({self.clip_qkv}) should be non-negative")


class OLMoEmbedding(nn.Embedding):
    """The embedding module that can be shared with the final lm_head. From Qwen2Embedding."""

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class OLMoAttention(nn.Module):  # pylint: disable=missing-class-docstring
    def __init__(self, config: OLMoConfig):
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        assert (
            config.num_key_value_heads >= config.tensor_parallel_shards
        ), f"Too large tensor_parallel_shards, must be smaller than {config.num_key_value_heads}"
        assert (
            config.num_key_value_heads % config.tensor_parallel_shards == 0
        ), f"num_kv_heads({config.num_key_value_heads}) must be divisible by tensor_parallel_shards"
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.clip_qkv = config.clip_qkv
        self.o_proj = nn.Linear(
            in_features=self.num_q_heads * self.head_dim,
            out_features=config.hidden_size,
            bias=False,
        )

    def forward(  # pylint: disable=missing-function-docstring
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape

        # QKV Projection
        qkv = self.qkv_proj(hidden_states)

        # Clamp after qkv projection if needed
        if self.clip_qkv is not None:
            qkv = qkv.maximum(-self.clip_qkv).minimum(self.clip_qkv)

        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_q_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


# Copied from qwen2_model.ACT2FN
ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class OLMoFFN(nn.Module):  # pylint: disable=missing-class-docstring
    def __init__(self, config: OLMoConfig):
        super().__init__()
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.down_proj = nn.Linear(
            in_features=self.intermediate_size,
            out_features=config.hidden_size,
            bias=False,
        )

    def forward(self, x: Tensor):  # pylint: disable=missing-function-docstring
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


# pylint: disable=trailing-whitespace


class OLMoDecoderLayer(nn.Module):  # pylint: disable=missing-class-docstring
    def __init__(self, config: OLMoConfig):
        self.input_layernorm = nn.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=1e-5,
            elementwise_affine=False,
        )
        self.self_attn = OLMoAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=1e-5,
            elementwise_affine=False,
        )
        self.mlp = OLMoFFN(config)

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
            _set(self.mlp.gate_up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual

    def forward(  # pylint: disable=missing-function-docstring
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ):
        out = self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states


class OLMoModel(nn.Module):  # pylint: disable=missing-class-docstring
    def __init__(self, config: OLMoConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = OLMoEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [OLMoDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=1e-5,
            elementwise_affine=False,
        )

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

    def forward(  # pylint: disable=missing-function-docstring
        self, inputs: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            if layer_id != 0 and layer_id in self.layer_partition:
                hidden_states = op_ext.pipeline_stage_boundary(hidden_states)
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class OLMoForCausalLM(  # pylint: disable=missing-class-docstring,too-many-instance-attributes
    nn.Module
):
    def __init__(self, config: OLMoConfig):
        self.model = OLMoModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.position_embedding_base
        self.rope_scaling = config.rope_scaling
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"

        def _set_pp():
            # hidden layers
            for layer_id in range(config.num_hidden_layers):
                stage = layer_id // (config.num_hidden_layers // config.pipeline_parallel_stages)
                for _, param in self.model.layers[layer_id].named_parameters():
                    param.attrs["pipeline_stages"] = [stage]

            # embedding table and lm_head is required by all stages
            all_stages = list(range(config.pipeline_parallel_stages))
            self.model.embed_tokens.weight.attrs["pipeline_stages"] = all_stages
            if not config.tie_word_embeddings:
                self.lm_head.weight.attrs["pipeline_stages"] = all_stages

        _set_pp()

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(  # pylint: disable=missing-function-docstring
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

    def batch_forward_to_last_hidden_states(  # pylint: disable=missing-function-docstring
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        op_ext.configure()
        hidden_states = self.model(input_embeds, paged_kv_cache)
        return hidden_states

    def embed(self, input_ids: Tensor):  # pylint: disable=missing-function-docstring
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def get_logits(self, hidden_states: Tensor):  # pylint: disable=missing-function-docstring
        op_ext.configure()
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def batch_select_last_hidden_states(  # pylint: disable=missing-function-docstring
        self, hidden_states: Tensor, logit_positions: Tensor
    ):
        op_ext.configure()
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        hidden_states = op.take(hidden_states, logit_positions, axis=0)
        return hidden_states

    def prefill(  # pylint: disable=missing-function-docstring
        self, input_embed: Tensor, paged_kv_cache: PagedKVCache
    ):
        op_ext.configure()

        def _index(x: te.Tensor):  # get tensor of the last sequence
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k])

        # pylint: disable=trailing-whitespace
        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    # pylint: disable=trailing-whitespace
    def decode(  # pylint: disable=missing-function-docstring
        self, input_embed: Tensor, paged_kv_cache: PagedKVCache
    ):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def prefill_to_last_hidden_states(  # pylint: disable=missing-function-docstring
        self, input_embed: Tensor, paged_kv_cache: PagedKVCache
    ):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def decode_to_last_hidden_states(  # pylint: disable=missing-function-docstring
        self, input_embed: Tensor, paged_kv_cache: PagedKVCache
    ):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_prefill(  # pylint: disable=missing-function-docstring
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(  # pylint: disable=missing-function-docstring
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(  # pylint: disable=missing-function-docstring
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_prefill_to_last_hidden_states(  # pylint: disable=missing-function-docstring
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = self.batch_forward_to_last_hidden_states(input_embeds, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_decode_to_last_hidden_states(  # pylint: disable=missing-function-docstring
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = self.batch_forward_to_last_hidden_states(input_embeds, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_verify_to_last_hidden_states(  # pylint: disable=missing-function-docstring
        self, input_embeds: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = self.batch_forward_to_last_hidden_states(input_embeds, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=missing-function-docstring,too-many-arguments
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
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            layer_partition=self.model.layer_partition,
            dtype=self.dtype,
        )

    def get_default_spec(self):  # pylint: disable=missing-function-docstring
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
