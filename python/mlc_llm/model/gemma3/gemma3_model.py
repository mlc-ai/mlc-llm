"""Implementation for Gemma3 architecture."""

import dataclasses
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.gemma.gemma_model import GemmaEmbedding
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Gemma3TextConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the text model inside Gemma3"""

    # NOTE More fields have defaults due to Huggingface Gemma3 configs missing fields
    # The defaults for these fields can be found in the transformers library
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    attention_bias: bool = False
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    hidden_activation: Optional[str] = "gelu_pytorch_tanh"
    position_embedding_base: int = 0
    context_window_size: int = 131_072
    prefill_chunk_size: int = 0

    query_pre_attn_scalar: int = 256
    sliding_window: int = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.hidden_activation is None:
            self.hidden_activation = self.kwargs.get("hidden_act", None)
        if self.hidden_activation not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError("Only GeLU is supported as the activation for gemma.")
        if self.attention_bias:
            raise ValueError('Only "False" attention_bias is supported for gemma')
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
        # NOTE: override the context window size with the Gemma2 sliding window size,
        # as the sliding window attention every other layer is yet to be supported.
        self.context_window_size = max(self.sliding_window, 8192)


@dataclasses.dataclass
class Gemma3Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Gemma3 model"""

    text_config: Gemma3TextConfig = None
    vocab_size: int = 262_208
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    sliding_window_size: int = -1
    prefill_chunk_size: int = -1
    is_text_model: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.text_config is None:
            self.is_text_model = True
            self.text_config = Gemma3TextConfig.from_dict(self.kwargs)

        text_config_dict: Dict[str, Any]  # type: ignore
        if isinstance(self.text_config, Gemma3TextConfig):
            text_config_dict = dataclasses.asdict(self.text_config)
        else:
            text_config_dict = dict(self.text_config)

        for k, v in text_config_dict.pop("kwargs", {}).items():
            text_config_dict[k] = v

        self.text_config = Gemma3TextConfig.from_dict(text_config_dict)

        for k in ["context_window_size", "prefill_chunk_size"]:
            if getattr(self, k) <= 0:
                if hasattr(self.text_config, k):
                    setattr(self, k, getattr(self.text_config, k))


# pylint: disable=invalid-name,missing-docstring


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3Config):
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
        return self.down_proj(op.gelu(x1, approximate="tanh") * x2)


class Gemma3Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Gemma3Config):
        self.head_dim = config.text_config.head_dim
        self.num_q_heads = config.text_config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.text_config.num_key_value_heads
        assert (
            self.num_kv_heads % config.tensor_parallel_shards == 0
        ), f"num_kv_heads({self.num_kv_heads}) must be divisible by tensor_parallel_shards"
        assert (
            self.num_kv_heads >= config.tensor_parallel_shards
        ), f"Too large tensor_parallel_shards, must be smaller than {self.num_kv_heads}"
        self.num_kv_heads = self.num_kv_heads // config.tensor_parallel_shards
        self.q_proj = nn.Linear(
            in_features=config.text_config.hidden_size,
            out_features=self.num_q_heads * self.head_dim,
            bias=config.text_config.attention_bias,
        )
        self.k_proj = nn.Linear(
            in_features=config.text_config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            bias=config.text_config.attention_bias,
        )
        self.v_proj = nn.Linear(
            in_features=config.text_config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            bias=config.text_config.attention_bias,
        )
        self.o_proj = nn.Linear(
            in_features=self.num_q_heads * self.head_dim,
            out_features=config.text_config.hidden_size,
            bias=config.text_config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(
            config.text_config.head_dim, -1, config.text_config.rms_norm_eps, bias=False
        )
        self.k_norm = nn.RMSNorm(
            config.text_config.head_dim, -1, config.text_config.rms_norm_eps, bias=False
        )
        # self.scaling_factor = (self.head_dim / config.text_config.query_pre_attn_scalar) ** 0.5
        self.scaling = config.text_config.query_pre_attn_scalar**-0.5

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q = self.head_dim, self.num_q_heads
        b, s, _ = hidden_states.shape
        # QKV Projection
        q_proj = op.reshape(self.q_proj(hidden_states), (b, s, -1, d))
        k_proj = op.reshape(self.k_proj(hidden_states), (b, s, -1, d))
        v_proj = op.reshape(self.v_proj(hidden_states), (b, s, -1, d))

        q_norm = self.q_norm(q_proj)
        k_norm = self.k_norm(k_proj)

        qkv = op.concat([q_norm, k_norm, v_proj], dim=2)

        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_q_heads, sm_scale=self.scaling
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3Config):
        rms_norm_eps = config.text_config.rms_norm_eps
        self.self_attn = Gemma3Attention(config)
        self.mlp = Gemma3MLP(config)
        # Gemma RMSNorm adds 1 to the weights. It is already fused in the loader
        self.input_layernorm = nn.RMSNorm(
            config.text_config.hidden_size, -1, rms_norm_eps, bias=False
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.text_config.hidden_size, -1, rms_norm_eps, bias=False
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.text_config.hidden_size, -1, rms_norm_eps, bias=False
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.text_config.hidden_size, -1, rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            i = self.mlp.intermediate_size
            _set(self.self_attn.q_proj, tp.ShardSingleDim("_shard_q", dim=0))
            _set(self.self_attn.k_proj, tp.ShardSingleDim("_shard_k", dim=0))
            _set(self.self_attn.v_proj, tp.ShardSingleDim("_shard_v", dim=0))
            _set(self.self_attn.q_norm, tp.ShardSingleDim("_shard_q_norm", dim=0))
            _set(self.self_attn.k_norm, tp.ShardSingleDim("_shard_k_norm", dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.mlp.gate_up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        out = self._apply_post_matmul_norm(out, norm=self.post_attention_layernorm)
        hidden_states = out + hidden_states

        out = self.pre_feedforward_layernorm(hidden_states)
        out = self.mlp(out)
        out = self._apply_post_matmul_norm(out, norm=self.post_feedforward_layernorm)
        hidden_states = out + hidden_states

        return hidden_states

    def _apply_post_matmul_norm(self, out: Tensor, norm: nn.Tensor):
        if self.tensor_parallel_shards > 1:
            return norm(op.ccl_allreduce(out, "sum"))
        return norm(out)


class Gemma3TextModel(nn.Module):
    def __init__(self, config: Gemma3Config):
        self.hidden_size = config.text_config.hidden_size
        assert config.text_config.hidden_size % config.text_config.num_attention_heads == 0
        self.embed_tokens = GemmaEmbedding("vocab_size", config.text_config.hidden_size)
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config) for _ in range(config.text_config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(
            config.text_config.hidden_size, -1, config.text_config.rms_norm_eps, bias=False
        )

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        hidden_states = hidden_states * (self.hidden_size**0.5)
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3LanguageModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Gemma3Config):
        self.model = Gemma3TextModel(config)
        self.num_hidden_layers = config.text_config.num_hidden_layers
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_key_value_heads = config.text_config.num_key_value_heads
        self.head_dim = config.text_config.head_dim
        self.hidden_size = config.text_config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.text_config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_logits(self, hidden_states: Tensor):
        logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

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
        logits = self.get_logits(hidden_states)
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
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
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


class Gemma3ForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.language_model = Gemma3LanguageModel(config)
        self.vocab_size = config.vocab_size
        self.dtype = "float32"
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        self.language_model.to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_logits(self, hidden_states: Tensor):
        logits = self.language_model.model.embed_tokens.lm_head_forward(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.language_model.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.get_logits(hidden_states)
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.language_model.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
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
            num_hidden_layers=self.language_model.num_hidden_layers,
            num_attention_heads=self.language_model.num_attention_heads
            // self.tensor_parallel_shards,
            num_key_value_heads=self.language_model.num_key_value_heads
            // self.tensor_parallel_shards,
            qk_head_dim=self.language_model.head_dim,
            v_head_dim=self.language_model.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.language_model.rope_theta,
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
                    [1, "seq_len", self.language_model.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.language_model.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.language_model.hidden_size], self.dtype
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
                    ["batch_size", 1, self.language_model.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.language_model.hidden_size], self.dtype
                ),
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
