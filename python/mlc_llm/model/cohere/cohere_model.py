"""
Implementation for Aya23 architecture
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
class CohereConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Cohere Aya-23 model"""

    model_type: str  # cohere
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    layer_norm_eps: float
    position_embedding_base: int = 0
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    head_dim: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs["rope_theta"]
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

            if self.num_key_value_heads == 0 or self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_attention_heads
            if self.head_dim == 0:
                self.head_dim = self.hidden_size // self.num_attention_heads
            assert (
                self.head_dim * self.num_attention_heads == self.hidden_size
            ), "head_dim * num_attention_heads != hidden_size"
            assert (
                self.num_attention_heads % self.num_key_value_heads == 0
            ), "num_attention_heads % num_key_value_heads != 0"


# pylint: disable=invalid-name,missing-docstring


class CohereMLP(nn.Module):
    def __init__(self, config: CohereConfig):
        super().__init__()
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )

        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(op.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# pylint: disable=invalid-name,missing-docstring


class CohereAttention(nn.Module):
    def __init__(self, config: CohereConfig):
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        assert config.num_attention_heads % config.tensor_parallel_shards == 0, (
            f"num_attention_heads({config.num_attention_heads}) "
            "must be divisible by tensor_parallel_shards"
        )
        self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        assert config.num_key_value_heads % config.tensor_parallel_shards == 0, (
            f"num_attention_heads({config.num_key_value_heads}) "
            "must be divisible by tensor_parallel_shards"
        )
        self.head_dim = config.head_dim

        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_key_value_heads
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
        return self.out_proj(output)


class CohereDecoderLayer(nn.Module):
    def __init__(self, config: CohereConfig):
        super().__init__()
        self.self_attn = CohereAttention(config)
        self.mlp = CohereMLP(config)
        self.input_layernorm = CohereNorm(config.hidden_size, eps=config.layer_norm_eps)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_key_value_heads * hd
            v = self.self_attn.num_key_value_heads * hd
            i = self.mlp.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.self_attn.out_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.mlp.gate_proj, tp.ShardSingleDim("_shard_mlp_gate", segs=[i, i], dim=0))
            _set(self.mlp.up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        hidden_ln = self.input_layernorm(hidden_states)
        attn = self.self_attn(hidden_ln, paged_kv_cache, layer_id)
        mlp = self.mlp(hidden_ln)
        hidden_states = self._apply_parallel_residual(attn, residual=hidden_states)  # type: ignore
        hidden_states = self._apply_parallel_residual(mlp, residual=hidden_states)  # type: ignore
        return hidden_states

    def _apply_parallel_residual(self, mlp_out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(mlp_out + residual / self.tensor_parallel_shards, "sum")
        return mlp_out + residual


class CohereNorm(nn.Module):
    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, dtype: Optional[str] = None
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter((normalized_shape,), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return op.layer_norm(
            x,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=None,
            eps=self.eps,
        )


class CohereEmbedding(nn.Embedding):
    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class CohereModel(nn.Module):
    def __init__(self, config: CohereConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = CohereEmbedding("vocab_size", config.hidden_size)
        self.layers = nn.ModuleList(
            [CohereDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CohereNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class CohereForCausalLM(nn.Module):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, config: CohereConfig) -> None:
        super().__init__()
        self.model = CohereModel(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
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
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        lm_logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        if lm_logits.dtype != "float32":
            lm_logits = lm_logits.astype("float32")
        return lm_logits

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape  # type: ignore
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        # logits = self.lm_head(hidden_states)
        logits = self.model.embed_tokens.lm_head_forward(hidden_states)  # type: ignore

        if logits.dtype != "float32":
            logits = logits.astype("float32")

        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)  # type: ignore
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)  # type: ignore
        embeds = self.model.embed_tokens(input_ids)
        return embeds

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
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)  # type: ignore
