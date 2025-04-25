"""
Implementation for QWEN architecture.
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
class QWenConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the QWen model."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_epsilon: float
    scale_attn_weights: bool
    kv_channels: int
    rotary_emb_base: int
    intermediate_size: int
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    head_dim: int = 0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
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
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * self.num_attention_heads == self.hidden_size
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


class QWenAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: QWenConfig):
        self.hidden_size = config.hidden_size
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.head_dim = config.head_dim

        self.c_attn = nn.Linear(config.hidden_size, 3 * self.num_heads * self.head_dim, bias=True)

        self.c_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(  # pylint: disable=too-many-locals
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
    ):
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, 3 * h, d))
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h * d),
        )
        return self.c_proj(output)


class QWenMLP(nn.Module):
    def __init__(self, config: QWenConfig):
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
        )
        self.c_proj = nn.Linear(self.intermediate_size // 2, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.c_proj(x1 * op.silu(x2))


class QWenBlock(nn.Module):
    def __init__(self, config: QWenConfig):
        rms_norm_eps = config.layer_norm_epsilon
        self.attn = QWenAttention(config)
        self.mlp = QWenMLP(config)
        self.ln_1 = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.ln_2 = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.attn.num_heads * hd
            k = self.attn.num_heads * hd
            v = self.attn.num_heads * hd
            i = self.mlp.intermediate_size // 2
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
                self.mlp.gate_up_proj.weight,
                tp.ShardSingleDim("_shard_mlp_gate_up_proj", segs=[i, i], dim=0),
            )
            _set(self.mlp.c_proj.weight, tp.ShardSingleDim("_shard_mlp_c_proj", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.attn(self.ln_1(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.ln_2(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class QWenModel(nn.Module):
    def __init__(self, config: QWenConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([QWenBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.RMSNorm(config.hidden_size, -1, config.layer_norm_epsilon, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.h):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class QWenLMHeadModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: QWenConfig):
        self.transformer = QWenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype="float32")
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.rotary_emb_base = config.rotary_emb_base
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states = self.transformer(inputs, paged_kv_cache)
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

    def prefill(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.transformer(inputs, paged_kv_cache)
        hidden_states = op.tensor_expr_op(
            _index,
            name_hint="index",
            args=[hidden_states],
        )
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.transformer(inputs, paged_kv_cache)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(self, inputs: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(inputs, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(inputs, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(inputs, paged_kv_cache)
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
            num_key_value_heads=self.num_attention_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rotary_emb_base,
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
                "inputs": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "inputs": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "inputs": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "inputs": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
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
