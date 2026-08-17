"""
Implementation for GptOss MoE architecture.
"""

import dataclasses
import math
from typing import Any, Dict, Optional

import tvm
from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import GptOssMxfp4Experts
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GptOssConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GptOss model."""

    num_hidden_layers: int = 36
    num_local_experts: int = 128
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: float = 150000.0
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    rope_scaling: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
        }
    )
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 4
    router_aux_loss_coef: float = 0.9
    output_router_logits: bool = False
    attention_bias: bool = False
    pad_token_id: int = 199999
    use_cache: bool = True
    layer_types: list = None
    dtype: str = "float32"
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    sliding_window_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    disaggregation: bool = False
    perplexity: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):  # pylint: disable=too-many-branches
        if self.rope_scaling is not None:
            if "rope_type" not in self.rope_scaling:
                self.rope_scaling = None
            else:
                assert (
                    self.rope_scaling["rope_type"] == "yarn"
                ), f'Unsupported RoPE scaling type {self.rope_scaling["rope_type"]} for GptOss'
            if self.rope_scaling is not None and "inv_theta_log_scale" not in self.rope_scaling:
                self.rope_scaling["inv_theta_log_scale"] = 1.0 / (2 * math.log(self.rope_theta))
        if self.sliding_window_size == 0:
            self.sliding_window_size = self.kwargs.pop("sliding_window", -1)
        if self.sliding_window_size is None:
            # Sliding window is disabled.
            self.sliding_window_size = -1
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


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class GptOssAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GptOssConfig):
        self.head_dim = config.head_dim
        if config.num_key_value_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_key_value_heads} key-value attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.rope_theta = config.rope_theta
        self.attention_bias = config.attention_bias
        self.sinks = nn.Parameter((config.num_attention_heads,), dtype="float16")
        self.c_attn = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            # out_features=(2 * self.num_key_value_heads + self.num_attention_heads) * self.head_dim,
            bias=self.attention_bias,
            dtype="float16",
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=self.attention_bias,
            dtype="float16",
        )

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        query_positions: Tensor,
        layer_id: int,
    ):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        # qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        qkv = op.reshape(qkv, (b, s, h_q + 2 * h_kv, d))
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id,
                qkv,
                self.num_attention_heads,
                sm_scale=self.head_dim**-0.5,
                sinks=self.sinks,
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class GptOssMLP(nn.Module):  # pylint: disable=too-many-instance-attributes
    """MoE block for GptOss model."""

    # GptOss activation hyper-parameters
    _ALPHA: float = 1.702
    _LIMIT: float = 7.0

    def __init__(self, config: GptOssConfig):
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.intermediate_size
        self.dtype = "float16"
        self.mxfp4_group_size = 32  # GptOss checkpoint group size

        # Router: maps hidden states to expert logits
        self.gate = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_experts,
            bias=True,
            dtype="float16",
        )

        # Gate+Up projection experts (output dim = 2 * intermediate_size)
        self.moe_gate_up_proj = GptOssMxfp4Experts(
            num_local_experts=self.num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.moe_intermediate_size,
            group_size=self.mxfp4_group_size,
        )

        # Down projection experts (output dim = hidden_size)
        self.moe_down_proj = GptOssMxfp4Experts(
            num_local_experts=self.num_experts,
            in_features=self.moe_intermediate_size,
            out_features=self.hidden_size,
            group_size=self.mxfp4_group_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        def _expert_forward(x: Tensor, indptr: Tensor) -> Tensor:
            """Run gate-up MXFP4 projection, GptOss activation, then down MXFP4 projection."""
            # --- gate-up projection (MXFP4, bias included inside forward) ---
            x1_x2 = self.moe_gate_up_proj(x, indptr)
            x1_x2 = x1_x2.reshape(*x1_x2.shape[:-1], self.moe_intermediate_size, 2)
            x1, x2 = op.split(x1_x2, indices_or_sections=2, axis=-1)
            x1 = x1.reshape(*x1.shape[:-1])  # drop trailing size-1 dim
            x2 = x2.reshape(*x2.shape[:-1])  # drop trailing size-1 dim
            x1 = x1.minimum(self._LIMIT)
            x2 = x2.minimum(self._LIMIT).maximum(-self._LIMIT)
            x1 = x1 * op.sigmoid(x1 * self._ALPHA)
            x = (x2 + 1) * x1
            # --- down projection (MXFP4, bias included inside forward) ---
            return self.moe_down_proj(x, indptr)

        experts_per_tok = self.num_experts_per_tok
        num_experts = self.num_experts
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len

        # --- routing ---
        gate_logits = self.gate(hidden_states)
        gate_logits = gate_logits.reshape(num_tokens, -1)
        expert_weights, expert_indices = op_ext.moe_misc.gating_topk(gate_logits, experts_per_tok)
        expert_weights = op.softmax(expert_weights, axis=-1)

        use_cutlass = op_ext.get_store().cutlass_group_gemm and self.dtype in [
            "float16",
            "bfloat16",
        ]

        if seq_len == 1:
            moe_hidden_states = _expert_forward(
                hidden_states,
                expert_indices,  # (batch, experts_per_tok), ndim==2
            )
        else:
            x = hidden_states.reshape(num_tokens, hidden_size)

            # cumsum: [num_tokens * num_experts] -- token-to-expert assignment
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, num_experts)

            # reverse_indices: scatter-back permutation
            # token_indices:   gather permutation (tokens ordered by expert)
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)

            # indptr: segment boundaries for grouped GEMM (ndim==1)
            if use_cutlass:
                # inclusive indptr: [num_experts], dtype int64
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, num_experts, num_tokens, inclusive=True, out_dtype="int64"
                )
            else:
                # exclusive indptr: [num_experts + 1], dtype int32
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, num_experts, num_tokens, inclusive=False, out_dtype="int32"
                )

            # Gather tokens in expert order, run experts, scatter back
            # moe_hidden_states: [num_tokens * experts_per_tok, hidden_size]
            moe_hidden_states = op.take(x, token_indices, axis=0)
            moe_hidden_states = _expert_forward(moe_hidden_states, indptr)
            moe_hidden_states = op_ext.moe_misc.scatter_output(moe_hidden_states, reverse_indices)

        # --- weighted reduction ---
        # moe_hidden_states: [num_tokens, experts_per_tok, hidden_size]
        expert_weights = expert_weights.reshape(num_tokens, experts_per_tok, 1)
        moe_hidden_states = (
            moe_hidden_states.reshape(num_tokens, experts_per_tok, hidden_size) * expert_weights
        )
        # moe_hidden_states: [num_tokens, hidden_size]
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)

        return moe_hidden_states.reshape(batch_size, seq_len, hidden_size)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GptOssAttention(config)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False, dtype="float16"
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False, dtype="float16"
        )
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        query_positions: Tensor,
        layer_id: int,
    ):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, paged_kv_cache, query_positions, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype="float16")
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False, dtype="float16"
        )

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        query_positions = paged_kv_cache.get_query_positions(inputs.shape[0] * inputs.shape[1])
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, query_positions, layer_id)
        return self.norm(hidden_states)


class GptOssForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GptOssConfig):
        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype="float16")
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.rope_scaling = config.rope_scaling
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.sliding_window_size = config.sliding_window_size
        self.attn_kinds = [
            "mha_sliding" if (ty == "sliding_attention") else "mha" for ty in config.layer_types
        ]

    def to(self, dtype: Optional[str] = None):
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
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
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
            attn_kind=self.attn_kinds,
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
            dtype=self.dtype,
            is_sinks=True,
            sliding_window_size=self.sliding_window_size,
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
