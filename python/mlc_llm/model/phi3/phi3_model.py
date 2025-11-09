"""
Implementation for Phi-3 architecture.
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
class Phi3Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Phi-3 model."""

    model_type: str  # "phi", "phi-msft", "mixformer-sequential"
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_eps: float
    num_key_value_heads: int
    max_position_embeddings: int
    position_embedding_base: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    original_max_position_embeddings: int = 0
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    head_dim: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.rope_scaling is not None:
            if "type" not in self.rope_scaling:
                self.rope_scaling = None
            else:
                if self.rope_scaling["type"] == "su":
                    self.rope_scaling["type"] = "longrope"

                assert (
                    self.rope_scaling["type"] == "longrope"
                ), f'Unsupported RoPE scaling type {self.rope_scaling["rope_type"]} for Phi3'
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
                (
                    self.rope_scaling["max_position_embeddings"],
                    self.rope_scaling["original_max_position_embeddings"],
                ) = (self.max_position_embeddings, self.original_max_position_embeddings)

        if self.context_window_size == 0:
            self.context_window_size = self.max_position_embeddings

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
        assert self.head_dim * self.num_attention_heads == self.hidden_size
        assert self.num_attention_heads % self.num_key_value_heads == 0


# pylint: disable=invalid-name,missing-docstring


class Phi3Embedding(nn.Embedding):
    """The embedding module that can be shared with the final lm_head."""

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class Phi3MLP(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor):
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = nn.op.split(up_states, 2, axis=-1)
        up_states = up_states * op.silu(gate)
        return self.down_proj(up_states)


class PhiMHA(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Phi3Config):
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


class Phi3ParallelBlock(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()

        self.ln = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.mixer = PhiMHA(config)
        self.mlp = Phi3MLP(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.mixer.num_q_heads * hd
            k = self.mixer.num_key_value_heads * hd
            v = self.mixer.num_key_value_heads * hd
            i = self.mlp.intermediate_size

            _set(self.mixer.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.mixer.out_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.mlp.gate_up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        attn_outputs = self.mixer(self.ln(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_parallel_residual(attn_outputs, hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_parallel_residual(out, hidden_states)
        return hidden_states

    def _apply_parallel_residual(self, mlp_out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(mlp_out + residual / self.tensor_parallel_shards, "sum")
        return mlp_out + residual


class Phi3Model(nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.embd = Phi3Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([Phi3ParallelBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.h):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Phi3ForCausalLM(nn.Module):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()

        self.transformer = Phi3Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, "vocab_size", bias=False)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_scaling = config.rope_scaling
        self.rope_theta = config.position_embedding_base
        self.rope_ext_factors = (
            (config.rope_scaling["long_factor"] + config.rope_scaling["short_factor"])
            if config.rope_scaling is not None
            else None
        )
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.partial_rotary_factor = config.partial_rotary_factor
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_logits(self, hidden_states: Tensor):
        op_ext.configure()
        if self.tie_word_embeddings:
            logits = self.transformer.embd.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
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

        hidden_states = self.transformer(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.transformer(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.transformer(input_embed, paged_kv_cache)
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

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        embeds = self.transformer.embd(input_ids)
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
            rope_scaling=self.rope_scaling,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_ext_factors=self.rope_ext_factors,
            rotary_dim=int(self.head_dim * self.partial_rotary_factor),
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
