"""
Implementation for GLM-4.5-Air MoE architecture.
"""

import dataclasses
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Glm4MoeConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GLM-4.5-Air MoE model."""

    hidden_size: int = 4096
    num_hidden_layers: int = 46
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 10944  # Dense MLP size (for first_k_dense_replace layers)
    moe_intermediate_size: int = 1408  # Expert MLP size
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 1  # First N layers use dense MLP
    partial_rotary_factor: float = 0.5
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-5
    vocab_size: int = 151552
    hidden_act: str = "silu"
    attention_bias: bool = True
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "seq_length"]:
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
                    "`context_window_size`, `max_position_embeddings` or `seq_length` is "
                    "provided in `config.json`."
                )
        # GLM-4.5 uses independent head_dim (128) that doesn't follow head_dim = hidden_size / num_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
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


ACT2FN = {
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu": nn.gelu,
    "relu": nn.relu,
}


class Glm4MoeAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Multi-head attention for GLM-4.5 with partial RoPE support."""

    def __init__(self, config: Glm4MoeConfig):
        self.hidden_size = config.hidden_size
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        if config.num_key_value_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_key_value_heads} key-value heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.head_dim = config.head_dim
        # QKV combined projection (with bias for GLM-4.5)
        self.c_attn = nn.Linear(
            config.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.attention_bias,
        )
        # Output projection (no bias)
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        # Compute QKV
        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        # Apply attention with fused QKV (RoPE applied inside KV cache)
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, h_q, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class Glm4MoeDenseMLP(nn.Module):
    """Dense MLP for the first k layers (non-MoE layers)."""

    def __init__(self, config: Glm4MoeConfig):
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        # Combined gate and up projection
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class Glm4MoeSparseMoeBlock(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Sparse MoE layer for GLM-4.5 with 128 routed experts + 1 shared expert."""

    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor

        if config.moe_intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE intermediate size {config.moe_intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.moe_intermediate_size = config.moe_intermediate_size // config.tensor_parallel_shards

        # Router gate
        self.gate = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.n_routed_experts,
            bias=False,
        )
        # Router bias correction (for noaux_tc routing)
        self.e_score_correction_bias = nn.Parameter((config.n_routed_experts,), dtype="float32")

        # Routed experts (128 experts)
        self.moe_gate_up_proj = MixtralExperts(
            self.num_experts,
            in_features=config.hidden_size,
            out_features=2 * self.moe_intermediate_size,
        )
        self.moe_down_proj = MixtralExperts(
            self.num_experts,
            in_features=self.moe_intermediate_size,
            out_features=config.hidden_size,
        )

        # Shared expert (1 expert, added directly to output)
        self.shared_expert_gate_up = nn.Linear(
            config.hidden_size,
            2 * self.moe_intermediate_size,
            bias=False,
        )
        self.shared_expert_down = nn.Linear(
            self.moe_intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        def _expert_forward(x: Tensor, indptr: Tensor):
            x1_x2 = self.moe_gate_up_proj(x, indptr)
            x1, x2 = op.split(x1_x2, indices_or_sections=2, axis=-1)
            x = self.moe_down_proj(self.act_fn(x1) * x2, indptr)
            return x

        experts_per_tok = self.num_experts_per_tok
        num_experts = self.num_experts
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        x = x.reshape(num_tokens, hidden_size)

        # Compute router logits
        gate_logits = self.gate(x)

        # Use group_limited_greedy_topk for routing (noaux_tc method with bias correction)
        bias_f32 = self.e_score_correction_bias.astype("float32")
        if self.n_group > 1:
            expert_weights, expert_indices = op_ext.moe_misc.group_limited_greedy_topk(
                scores=gate_logits.astype("float32"),
                top_k=experts_per_tok,
                num_routed_experts=num_experts,
                n_group=self.n_group,
                topk_group=self.topk_group,
                topk_method="noaux_tc",
                num_tokens=num_tokens,
                e_score_correction_bias=bias_f32,
            )
        else:
            # Standard softmax top-k routing with bias correction
            gate_logits_corrected = gate_logits.astype("float32") + bias_f32
            expert_weights, expert_indices = op_ext.moe_misc.gating_softmax_topk(
                gate_logits_corrected.astype(x.dtype),
                experts_per_tok,
                norm_topk_prob=self.norm_topk_prob,
            )

        # Apply scaling factor
        if self.routed_scaling_factor != 1.0:
            expert_weights = expert_weights * self.routed_scaling_factor

        if num_tokens == 1:
            # Single token: direct expert computation
            moe_hidden_states = _expert_forward(x, expert_indices)
        else:
            # Multiple tokens: shuffle, compute, scatter
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, num_experts)
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)
            indptr = op_ext.moe_misc.get_indptr(
                cumsum, num_experts, num_tokens, inclusive=False, out_dtype="int32"
            )
            moe_hidden_states = op.take(x, token_indices, axis=0)
            moe_hidden_states = _expert_forward(moe_hidden_states, indptr)
            moe_hidden_states = op_ext.moe_misc.scatter_output(moe_hidden_states, reverse_indices)

        # Weight and sum expert outputs
        expert_weights = expert_weights.reshape(num_tokens, experts_per_tok, 1)
        moe_hidden_states = (
            moe_hidden_states.reshape(num_tokens, experts_per_tok, hidden_size) * expert_weights
        )
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)

        # Compute shared expert output (no gating, directly added)
        shared_x1_x2 = self.shared_expert_gate_up(x)
        shared_x1, shared_x2 = op.split(shared_x1_x2, 2, axis=-1)
        shared_expert_output = self.shared_expert_down(self.act_fn(shared_x1) * shared_x2)

        # Combine routed + shared expert outputs
        final_hidden_states = moe_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_size)
        return final_hidden_states


class Glm4MoeDecoderLayer(nn.Module):
    """Decoder layer with hybrid dense/MoE support."""

    def __init__(self, config: Glm4MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Glm4MoeAttention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        # Layer 0 to first_k_dense_replace-1 use dense MLP, rest use MoE
        if layer_idx < config.first_k_dense_replace:
            self.mlp = Glm4MoeDenseMLP(config)
            self.is_moe = False
        else:
            self.mlp = Glm4MoeSparseMoeBlock(config)
            self.is_moe = True

        self.tensor_parallel_shards = config.tensor_parallel_shards
        self._set_tp(config)

    def _set_tp(self, config: Glm4MoeConfig):
        def _set(layer, hint):
            layer.attrs["shard_strategy"] = hint

        hd = config.head_dim
        q = self.self_attn.num_attention_heads * hd
        k = self.self_attn.num_key_value_heads * hd
        v = self.self_attn.num_key_value_heads * hd

        _set(
            self.self_attn.c_attn.weight,
            tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
        )
        if config.attention_bias:
            _set(
                self.self_attn.c_attn.bias,
                tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
            )
        _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))

        if self.is_moe:
            mi = self.mlp.moe_intermediate_size
            _set(
                self.mlp.shared_expert_gate_up.weight,
                tp.ShardSingleDim("_shard_shared_mlp_up", segs=[mi, mi], dim=0),
            )
            _set(
                self.mlp.shared_expert_down.weight,
                tp.ShardSingleDim("_shard_shared_mlp_down", dim=1),
            )
            _set(
                self.mlp.moe_gate_up_proj.weight,
                tp.ShardSingleDim("_shard_moe_mlp_up", segs=[mi, mi], dim=1),
            )
            _set(self.mlp.moe_down_proj.weight, tp.ShardSingleDim("_shard_moe_mlp_down", dim=2))
        else:
            di = self.mlp.intermediate_size
            _set(
                self.mlp.gate_up_proj.weight,
                tp.ShardSingleDim("_shard_dense_mlp_up", segs=[di, di], dim=0),
            )
            _set(self.mlp.down_proj.weight, tp.ShardSingleDim("_shard_dense_mlp_down", dim=1))

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Glm4MoeModel(nn.Module):
    """GLM-4.5-Air MoE Transformer model."""

    def __init__(self, config: Glm4MoeConfig):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Glm4MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Glm4MoeForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    """GLM-4.5-Air MoE model for causal language modeling."""

    def __init__(self, config: Glm4MoeConfig):
        self.model = Glm4MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = "float32"
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.head_dim = config.head_dim
        # Partial RoPE: only apply to first 50% of head dimensions
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)

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
            rotary_dim=self.rotary_dim,  # Partial RoPE: 64 out of 128 dims
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
