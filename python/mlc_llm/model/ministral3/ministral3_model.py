"""
Implementation for Ministral 3 architecture.
"""

import dataclasses
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

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
class Ministral3Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Ministral 3 model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    attention_sink_size: int = 0
    context_window_size: int = 0
    dtype: str = "float32"
    head_dim: int = 0
    hidden_act: str = "silu"
    max_batch_size: int = 1
    num_key_value_heads: int = 0
    position_embedding_base: int = 0
    prefill_chunk_size: int = 0
    rope_parameters: Optional[Dict[str, Any]] = None
    sliding_window_size: int = 0
    tensor_parallel_shards: int = 1
    tie_word_embeddings: bool = False
    weight_block_size: Optional[Tuple[int, int]] = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    modules_to_not_convert: Tuple[str, ...] = dataclasses.field(default_factory=tuple)

    @classmethod
    def from_dict(  # type: ignore[override]
        cls,
        source: Dict[str, Any],
    ) -> "Ministral3Config":
        if "text_config" in source and isinstance(source["text_config"], dict):
            top_level = dict(source)
            text_cfg = top_level.pop("text_config")
            merged: Dict[str, Any] = dict(top_level)
            merged.update(text_cfg)
            if "tie_word_embeddings" in source:
                merged["tie_word_embeddings"] = source["tie_word_embeddings"]
            if "dtype" in source:
                merged["dtype"] = source["dtype"]
            return super().from_dict(merged)
        return super().from_dict(source)

    def __post_init__(self):  # pylint: disable=too-many-branches,too-many-statements
        if "quantization_config" in self.kwargs:
            quantization_config = self.kwargs.pop("quantization_config")
            if isinstance(quantization_config, dict):
                activation_scheme = quantization_config.get("activation_scheme", "")
                quant_method = quantization_config.get("quant_method", "")
                weight_block_size = quantization_config.get("weight_block_size")
                modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])
                if isinstance(modules_to_not_convert, list):
                    self.modules_to_not_convert = tuple(modules_to_not_convert)
                if quant_method == "fp8" and activation_scheme == "static":
                    if weight_block_size is not None:
                        self.weight_block_size = weight_block_size
                        if (
                            not isinstance(self.weight_block_size, (tuple, list))
                            or len(self.weight_block_size) != 2
                        ):
                            raise ValueError(
                                "Invalid Ministral3 quantization config: "
                                "weight_block_size must be a list or tuple of two integers, "
                                f"got {self.weight_block_size} "
                                f"of type {type(self.weight_block_size)}"
                            )
                    else:
                        # Set default block size if not provided.
                        self.weight_block_size = (128, 128)
                        logger.info(
                            "Setting default weight_block_size=%s since "
                            "quantization_config does not provide "
                            "FP8 block-scale details required by MLC (activation_scheme=%s, quant_method=%s)",
                            self.weight_block_size,
                            activation_scheme,
                            quant_method,
                        )
                else:
                    raise ValueError(
                        "Invalid Ministral 3 model quantization config: "
                        "only FP8 static quantization is supported, "
                        f"got activation_scheme={activation_scheme}, quant_method={quant_method}"
                    )
            else:
                raise ValueError(
                    "Invalid Ministral 3 model quantization config: unrecognized quantization config: "
                    f"{quantization_config}"
                )

        if self.position_embedding_base == 0:
            if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
                self.position_embedding_base = self.rope_parameters.pop("rope_theta")
            elif "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.sliding_window_size == 0:
            self.sliding_window_size = self.kwargs.pop("sliding_window", -1)
        if self.sliding_window_size is None:
            # Sliding window is disabled.
            self.sliding_window_size = -1
        if self.context_window_size == 0:
            if self.sliding_window_size == -1:
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
                        "`context_window_size`, `max_position_embeddings` or "
                        "`max_sequence_length` is provided in `config.json`."
                    )
            else:
                self.context_window_size = -1

        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.attention_sink_size >= 0
        if self.prefill_chunk_size == 0:
            prefill_chunk_size_candidates = []
            if self.sliding_window_size != -1:
                prefill_chunk_size_candidates.append(self.sliding_window_size)
            if self.context_window_size != -1:
                prefill_chunk_size_candidates.append(self.context_window_size)
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(*prefill_chunk_size_candidates, 8192),
            )
            self.prefill_chunk_size = min(*prefill_chunk_size_candidates, 8192)


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class Ministral3Embedding(nn.Embedding):
    """The embedding module specialized for Ministral3 so that
    it can be shared with the final lm_head.
    """

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


# pylint: disable=invalid-name,missing-docstring


class Ministral3MLP(nn.Module):
    """Same as in Llama architecture (LlamaFFN)."""

    def __init__(self, config: Ministral3Config):
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
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


def yarn_get_sm_scale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class Ministral3Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Same as LlamaAttention, but with sliding window attention using a rolling buffer cache."""

    def __init__(self, config: Ministral3Config):
        self.head_dim = config.head_dim
        if config.num_key_value_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_key_value_heads} key-value attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

        self.softmax_scale = self.head_dim ** (-0.5)
        if config.rope_parameters is not None:
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = config.rope_parameters["factor"]
            if mscale_all_dim:
                sm_scale = yarn_get_sm_scale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * sm_scale * sm_scale

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        # QKV Projection
        qkv = self.qkv_proj(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_q_heads, sm_scale=self.softmax_scale
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class Ministral3DecoderLayer(nn.Module):
    """Exact same as LlamaDecoderLayer."""

    def __init__(self, config: Ministral3Config):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = Ministral3Attention(config)
        self.mlp = Ministral3MLP(config)
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
            _set(self.mlp.gate_up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
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


class Ministral3Model(nn.Module):
    """Exact same as LlamaModel."""

    def __init__(self, config: Ministral3Config):
        assert config.hidden_size % config.num_attention_heads == 0
        # self.embed_tokens = nn.Embedding("vocab_size", config.hidden_size)
        self.embed_tokens = Ministral3Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Ministral3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Mistral3ForConditionalGeneration(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Ministral3Config):
        self.model = Ministral3Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )  # "vocab_size"
        self._mark_modules_no_quant(config.modules_to_not_convert)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.position_embedding_base
        self.rope_parameters = config.rope_parameters
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.sliding_window_size = config.sliding_window_size
        self.dtype = config.dtype
        self.weight_block_size = config.weight_block_size

    def _mark_modules_no_quant(self, modules: Tuple[str, ...]):
        for path in modules:
            if not path:
                continue
            parts = path.split(".")
            target = self
            for part in parts:
                if not hasattr(target, part):
                    target = None
                    break
                target = getattr(target, part)
            if target is not None:
                setattr(target, "no_quantization", True)

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
            hidden_states = op.take(hidden_states, logit_positions, axis=1)

        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

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

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
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
            rope_scaling=self.rope_parameters,
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
