"""
Implementation for GPTBigCode architecture.
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
class GPTBigCodeConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPTBigCode model."""

    n_embd: int
    n_inner: int
    n_head: int
    n_layer: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.context_window_size == 0:
            if self.n_positions > 0:
                self.context_window_size = self.n_positions
                logger.info(
                    "%s not found in config.json. Falling back to %s (%d)",
                    bold("context_window_size"),
                    bold("n_positions"),
                    self.context_window_size,
                )
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
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


class GPTBigCodeMLP(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        super().__init__()
        self.n_inner = config.n_inner // config.tensor_parallel_shards
        self.c_fc = nn.Linear(in_features=config.n_embd, out_features=self.n_inner, bias=True)
        self.c_proj = nn.Linear(in_features=self.n_inner, out_features=config.n_embd, bias=True)

    def forward(self, x: Tensor):
        hidden_states = self.c_fc(x)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPTBigCodeAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPTBigCodeConfig):
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.num_q_heads = config.n_head // config.tensor_parallel_shards
        self.num_kv_heads = 1
        assert (
            config.tensor_parallel_shards == 1
        ), "GPT bigcode only support tensor parallel shards = 1"
        self.c_attn = nn.Linear(
            in_features=self.n_embd,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=True,
        )
        self.c_proj = nn.Linear(
            in_features=self.num_q_heads * self.head_dim,
            out_features=config.n_embd,
            bias=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
    ):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape

        # QKV Projection
        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, h_q, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        return self.c_proj(output)


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        self.attn = GPTBigCodeAttention(config)
        self.mlp = GPTBigCodeMLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.n_embd // config.n_head
            q = config.n_head * hd
            k = 1 * hd
            v = 1 * hd
            _set(self.attn.c_attn, tp.ShardSingleDim("_shard_c_attn", dim=0, segs=[q, k, v]))
            _set(self.attn.c_proj, tp.ShardSingleDim("_shard_c_proj", dim=1))
            _set(self.mlp.c_fc, tp.ShardSingleDim("_shard_mlp_c_fc", dim=0))
            _set(self.mlp.c_proj, tp.ShardSingleDim("_shard_mlp_c_proj", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.attn(self.ln_1(hidden_states), paged_kv_cache, layer_id)
        hidden_states = out + hidden_states
        out = self.mlp(self.ln_2(hidden_states))
        hidden_states = out + hidden_states
        return hidden_states


class GPTBigCodeModel(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        assert config.n_embd % config.n_head == 0
        self.wte = nn.Embedding("vocab_size", config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([GPTBigCodeBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        # Position Embeddings
        # shape[1] indicates the total query length in the batch
        input_positions = paged_kv_cache.get_query_positions(input_embed.shape[1])
        pos_embd = self.wpe(input_positions)

        # apply position embeddings
        hidden_states = input_embed + pos_embd
        for layer_id, layer in enumerate(self.h):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class GPTBigCodeForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPTBigCodeConfig):
        self.transformer = GPTBigCodeModel(config)
        self.lm_head = nn.Linear(config.n_embd, "vocab_size", bias=False)
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.num_q_heads = config.n_head // config.tensor_parallel_shards
        self.num_kv_heads = 1
        self.head_dim = config.n_embd // config.n_head
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.transformer(input_embed, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.transformer.wte(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.transformer(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.transformer(input_embed, paged_kv_cache)
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
            num_hidden_layers=self.n_layer,
            num_attention_heads=self.num_q_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_kv_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=-1,
            rope_theta=-1,
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
                "input_embed": nn.spec.Tensor([1, "seq_len", self.n_embd], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.n_embd], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.n_embd], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.n_embd], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.n_embd], self.dtype),
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
