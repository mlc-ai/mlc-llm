"""
Implementation for Sarvam MoE architecture.
"""
import dataclasses
from typing import Any, Dict, Literal, Optional, Tuple

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T

from mlc_llm import op as op_ext
from mlc_llm.model.qwen3.qwen3_model import ACT2FN, Qwen3Attention
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SarvamMoeConfig(ConfigBase):
    """Configuration of the SarvamMoE model."""

    hidden_act: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: int
    vocab_size: int
    use_qk_norm: bool = True
    use_qkv_bias: bool = False
    use_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    dtype: str = "float32"
    max_batch_size: int = 1
    weight_block_size: Optional[Tuple[int, int]] = None
    moe_intermediate_size: int = 0
    num_experts_per_tok: int = 0
    num_experts: int = 0
    num_shared_experts: int = 0
    moe_shared_expert_intermediate_size: int = 0
    moe_router_enable_expert_bias: bool = False
    routed_scaling_factor: float = 1.0
    score_function: str = "sigmoid"
    first_k_dense_replace: int = 0
    n_group: int = 1
    topk_group: int = 1
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
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        self.attention_bias = self.use_qkv_bias or self.use_bias


class SarvamMoeMLP(nn.Module):
    """Dense MLP used in early dense layers and for shared expert branch."""

    def __init__(self, config: SarvamMoeConfig, intermediate_size: Optional[int] = None):
        intermediate_size = intermediate_size or config.intermediate_size
        if intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class SarvamMoeSparseMoeBlock(nn.Module):
    """MoE block for SarvamMoE"""

    def __init__(self, config: SarvamMoeConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.score_function = config.score_function
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        if config.moe_intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE intermediate size {config.moe_intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.moe_intermediate_size = config.moe_intermediate_size // config.tensor_parallel_shards
        self.gate = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
        )
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
        self.expert_bias = None
        if config.moe_router_enable_expert_bias:
            self.expert_bias = nn.Parameter((config.num_experts,), dtype="float32")
        self.shared_expert = None
        if config.num_shared_experts and config.num_shared_experts > 0:
            shared_intermediate = (
                config.moe_shared_expert_intermediate_size * config.num_shared_experts
            )
            self.shared_expert = SarvamMoeMLP(
                config,
                intermediate_size=shared_intermediate,
            )
        self.act_fn = ACT2FN[config.hidden_act]
        self.dtype = config.dtype

    def _expert_forward(self, x: Tensor, indptr: Tensor):
        x1_x2 = self.moe_gate_up_proj(x, indptr)
        x1, x2 = op.split(x1_x2, indices_or_sections=2, axis=-1)
        x = self.moe_down_proj(self.act_fn(x1) * x2, indptr)
        return x

    def _route_topk(self, gate_logits: Tensor):
        if self.score_function != "sigmoid":
            raise ValueError(
                f"Only sigmoid routing is currently supported, got {self.score_function}"
            )
        scores = op.sigmoid(gate_logits.astype("float32"))
        num_tokens = gate_logits.shape[0]
        if self.n_group == 1 and self.topk_group == 1:
            routing_scores = scores
            if self.expert_bias is not None:
                routing_scores = routing_scores + op.astype(self.expert_bias, routing_scores.dtype)
            expert_weights, expert_indices = op_ext.moe_misc.gating_topk(
                routing_scores, self.num_experts_per_tok
            )
            if self.expert_bias is not None:
                num_experts = self.num_experts
                num_experts_per_tok = self.num_experts_per_tok
                TX = 1024

                @T.prim_func(private=True)
                def gather_scores(
                    var_scores: T.handle,
                    var_expert_indices: T.handle,
                    var_output: T.handle,
                ):
                    T.func_attr({"tir.noalias": True, "tir.is_scheduled": True})
                    num_tokens_var = T.int64()
                    scores_buf = T.match_buffer(
                        var_scores,
                        (num_tokens_var, num_experts),
                        dtype=scores.dtype,
                    )
                    expert_indices_buf = T.match_buffer(
                        var_expert_indices,
                        (num_tokens_var, num_experts_per_tok),
                        dtype=expert_indices.dtype,
                    )
                    output = T.match_buffer(
                        var_output,
                        (num_tokens_var, num_experts_per_tok),
                        dtype=scores.dtype,
                    )
                    for io in T.thread_binding(0, T.ceildiv(num_tokens_var, TX), "blockIdx.x"):
                        for ii in T.thread_binding(0, TX, "threadIdx.x"):
                            with T.sblock("gather_scores"):
                                vi = T.axis.spatial(num_tokens_var, io * TX + ii)
                                T.where(io * TX + ii < num_tokens_var)
                                for j in T.unroll(num_experts_per_tok):
                                    with T.sblock("gather_inner"):
                                        vj = T.axis.remap("S", [j])
                                        output[vi, vj] = scores_buf[vi, expert_indices_buf[vi, vj]]

                expert_weights = op.tensor_ir_op(
                    gather_scores,
                    "sarvam_gather_scores",
                    args=[scores, expert_indices],
                    out=Tensor.placeholder((num_tokens, num_experts_per_tok), scores.dtype),
                )
        else:
            # General grouped-routing
            expert_weights, expert_indices = op_ext.moe_misc.group_limited_greedy_topk(
                scores=scores,
                top_k=self.num_experts_per_tok,
                num_routed_experts=self.num_experts,
                n_group=self.n_group,
                topk_group=self.topk_group,
                topk_method="noaux_tc",
                num_tokens=num_tokens,
                e_score_correction_bias=self.expert_bias,
            )
        denom = op.sum(expert_weights, axis=-1, keepdims=True)
        expert_weights = expert_weights / (denom + op.full(denom.shape, 1e-20, denom.dtype))
        if self.routed_scaling_factor != 1.0:
            expert_weights = expert_weights * self.routed_scaling_factor
        return expert_weights.astype(gate_logits.dtype), expert_indices

    def forward(self, x: Tensor):
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        identity = x
        x = x.reshape(num_tokens, hidden_size)
        gate_logits = self.gate(x)
        expert_weights, expert_indices = self._route_topk(gate_logits)
        use_cutlass = op_ext.get_store().cutlass_group_gemm and self.dtype in [
            "float16",
            "bfloat16",
        ]
        if num_tokens == 1:
            moe_hidden_states = self._expert_forward(x, expert_indices)
        else:
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, self.num_experts)
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)
            if use_cutlass:
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, self.num_experts, num_tokens, inclusive=True, out_dtype="int64"
                )
            else:
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, self.num_experts, num_tokens, inclusive=False, out_dtype="int32"
                )
            moe_hidden_states = op.take(x, token_indices, axis=0)
            moe_hidden_states = self._expert_forward(moe_hidden_states, indptr)
            moe_hidden_states = op_ext.moe_misc.scatter_output(moe_hidden_states, reverse_indices)
        expert_weights = expert_weights.reshape(num_tokens, self.num_experts_per_tok, 1)
        moe_hidden_states = (
            moe_hidden_states.reshape(num_tokens, self.num_experts_per_tok, hidden_size)
            * expert_weights
        )
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)
        moe_hidden_states = moe_hidden_states.reshape(batch_size, seq_len, hidden_size)
        if self.shared_expert is not None:
            moe_hidden_states = moe_hidden_states + self.shared_expert(identity)
        return moe_hidden_states


class SarvamMoeDecoderLayer(nn.Module):
    def __init__(self, config: SarvamMoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        if config.num_experts > 0 and layer_idx >= config.first_k_dense_replace:
            self.mlp = SarvamMoeSparseMoeBlock(config)
            self.is_moe_layer = True
        else:
            self.mlp = SarvamMoeMLP(config, intermediate_size=config.intermediate_size)
            self.is_moe_layer = False
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        def _set_tp():
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
            if self.is_moe_layer:
                mi = self.mlp.moe_intermediate_size
                _set(
                    self.mlp.moe_gate_up_proj.weight,
                    tp.ShardSingleDim("_shard_moe_mlp_up", segs=[mi, mi], dim=1),
                )
                _set(
                    self.mlp.moe_down_proj.weight,
                    tp.ShardSingleDim("_shard_moe_mlp_down", dim=2),
                )
            else:
                mi = self.mlp.intermediate_size
                _set(
                    self.mlp.gate_up_proj.weight,
                    tp.ShardSingleDim("_shard_mlp_up", segs=[mi, mi], dim=0),
                )
                _set(
                    self.mlp.down_proj.weight,
                    tp.ShardSingleDim("_shard_mlp_down", dim=1),
                )

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

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


class SarvamMoeModel(nn.Module):
    def __init__(self, config: SarvamMoeConfig):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                SarvamMoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SarvamMoeForCausalLM(nn.Module):
    def __init__(self, config: SarvamMoeConfig):
        self.model = SarvamMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Precision loss upon Quantization
        self.lm_head.no_quantization = True
        self.dtype = config.dtype
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
        self.weight_block_size = config.weight_block_size

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

        def _index(x: te.Tensor):
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
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
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

    def create_paged_kv_cache(
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
