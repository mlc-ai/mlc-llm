"""
Implementation for GPT-2 architecture.
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
class GPT2Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPT-2 model."""

    vocab_size: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float
    n_inner: int = -1
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    scale_attn_by_inverse_layer_idx: bool = False
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.n_inner is None or self.n_inner == -1:
            self.n_inner = 4 * self.n_embd
        if self.context_window_size == 0:
            for name in ["n_positions", "max_sequence_length"]:
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
                    "`context_window_size`, `n_positions` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.head_dim == 0:
            self.head_dim = self.n_embd // self.n_head
        assert self.head_dim * self.n_head == self.n_embd
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


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class GPT2Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPT2Config):
        self.embed_dim = config.n_embd
        if config.n_head % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.n_head} attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_heads = config.n_head // config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx

        self.c_attn = nn.Linear(
            in_features=self.embed_dim,
            out_features=3 * self.num_heads * self.head_dim,
            bias=True,
        )
        self.c_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, 3 * h, d))

        if self.scale_attn_by_inverse_layer_idx:
            attn_score_scaling_factor = 1.0 / float(layer_id + 1)
        else:
            attn_score_scaling_factor = 1.0

        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id,
                qkv,
                self.num_heads,
                sm_scale=attn_score_scaling_factor * (self.head_dim**-0.5),
            ),
            (b, s, h * d),
        )
        return self.c_proj(output)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        embed_dim = config.n_embd
        if config.n_inner % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.n_inner} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        intermediate_size = config.n_inner // config.tensor_parallel_shards
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = op.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        hidden_size = config.n_embd
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

        def _set_tp():
            def _set(param, hint):
                param.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = k = v = self.attn.num_heads * hd
            _set(
                self.attn.c_attn.weight,
                tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
            )
            _set(
                self.attn.c_attn.bias,
                tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
            )
            _set(self.attn.c_proj.weight, tp.ShardSingleDim("_shard_attn_c_proj", dim=1))
            _set(
                self.mlp.c_fc.weight,
                tp.ShardSingleDim("_shard_c_fc_weight", dim=0),
            )
            _set(self.mlp.c_fc.bias, tp.ShardSingleDim("_shard_c_fc_bias", dim=0))
            _set(self.mlp.c_proj.weight, tp.ShardSingleDim("_shard_mlp_c_proj", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        with tp.shard_bias(self.attn.c_proj, self.tensor_parallel_shards), tp.shard_bias(
            self.mlp.c_proj, self.tensor_parallel_shards
        ):
            hidden_states = self._apply_residual(
                self.attn(self.ln_1(hidden_states), paged_kv_cache, layer_id), hidden_states
            )
            hidden_states = self._apply_residual(self.mlp(self.ln_2(hidden_states)), hidden_states)

        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out + residual / self.tensor_parallel_shards, "sum")
        return out + residual


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        assert config.n_embd % config.n_head == 0
        self.wte = nn.Embedding("vocab_size", config.n_embd)
        self.wpe = nn.Embedding(config.context_window_size, config.n_embd)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        # Position Embeddings
        # Generate np.arange(offset, offset+seq_len)
        # shape[1] indicates the total query length in the batch
        input_positions = paged_kv_cache.get_query_positions(inputs.shape[1])
        pos_embd = self.wpe(input_positions)

        # Pass through GPT2Block
        hidden_states = inputs + pos_embd
        for layer_id, layer in enumerate(self.h):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPT2Config):
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, "vocab_size", bias=False)
        self.n_layer = config.n_layer
        self.n_embed = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim
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

        hidden_states = self.transformer(input_embeds, paged_kv_cache)
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
            num_attention_heads=self.n_head // self.tensor_parallel_shards,
            num_key_value_heads=self.n_head // self.tensor_parallel_shards,
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
                "input_embed": nn.spec.Tensor([1, "seq_len", self.n_embed], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.n_embed], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.n_embed], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.n_embed], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.n_embed], self.dtype),
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
