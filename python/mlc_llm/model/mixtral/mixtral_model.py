"""Implementation for Mistral architecture."""

import dataclasses

from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.llama.llama_model import (
    LlamaAttention,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from mlc_llm.nn import PagedKVCache
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MixtralConfig(LlamaConfig):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Mixtral model."""

    num_local_experts: int = 0
    num_experts_per_tok: int = 0


# pylint: disable=invalid-name,missing-docstring,too-many-locals,fixme


class MixtralMoE(nn.Module):
    """Mixture of experts"""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_local_experts = config.num_local_experts
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.num_local_experts,
            bias=False,
        )
        self.e1_e3 = MixtralExperts(
            self.num_local_experts,
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
            tensor_parallel_shards=config.tensor_parallel_shards,
        )
        self.e2 = MixtralExperts(
            self.num_local_experts,
            in_features=self.intermediate_size,
            out_features=config.hidden_size,
            tensor_parallel_shards=config.tensor_parallel_shards,
        )
        self.dtype = "float32"

    def forward(self, x: Tensor):
        def _expert_forward(x: Tensor, indptr: Tensor):
            x1_x3 = self.e1_e3(x, indptr)
            x1, x3 = op.split(x1_x3, indices_or_sections=2, axis=-1)
            x = self.e2(op.silu(x1) * x3, indptr)
            return x

        experts_per_tok = self.num_experts_per_tok  # activated experts per token
        local_experts = self.num_local_experts  # total number of experts
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        x = x.reshape(num_tokens, hidden_size)
        # gate: [num_tokens, local_experts]
        gate: Tensor = self.gate(x)
        # expert_weights: [num_tokens, experts_per_tok]
        # expert_indices: [num_tokens, experts_per_tok]
        expert_weights, expert_indices = op_ext.moe_misc.gating_softmax_topk(gate, experts_per_tok)
        use_ft = (
            op_ext.get_store().cutlass_group_gemm or op_ext.get_store().faster_transformer
        ) and self.dtype == "float16"
        if num_tokens == 1:
            # x: [num_tokens * experts_per_tok, hidden_size]
            x = _expert_forward(x, expert_indices)
        else:
            # cumsum: [num_tokens * local_experts]
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, local_experts)
            # indices: [num_tokens * experts_per_tok]
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)
            if use_ft:
                # indptr: [num_local_experts]
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, local_experts, num_tokens, inclusive=True, out_dtype="int64"
                )
            else:
                # indptr: [num_local_experts + 1]
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, local_experts, num_tokens, inclusive=False, out_dtype="int32"
                )
            # x: [num_tokens * experts_per_tok, hidden_size]
            x = op.take(x, token_indices, axis=0)
            x = _expert_forward(x, indptr)
            x = op_ext.moe_misc.scatter_output(x, reverse_indices)
        # x: [num_tokens, experts_per_tok, hidden_size]
        x = x.reshape(  # pylint: disable=too-many-function-args
            num_tokens, experts_per_tok, hidden_size
        ) * expert_weights.reshape(  # pylint: disable=too-many-function-args
            num_tokens, experts_per_tok, 1
        )
        # x: [num_tokens, hidden_size]
        x = op_ext.moe_misc.moe_sum(x, dim=1)
        x = x.reshape(batch_size, seq_len, hidden_size)  # pylint: disable=too-many-function-args
        return x


class MixtralDecoderLayer(nn.Module):
    """Mixtral decoder layer"""

    def __init__(self, config: MixtralConfig):
        eps = config.rms_norm_eps
        self.self_attn = LlamaAttention(config)
        self.moe = MixtralMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, eps, bias=False)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.moe.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.moe.e1_e3, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=1))
            _set(self.moe.e2, tp.ShardSingleDim("_shard_mlp_down", dim=2))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        """Forward pass of a decoder layer; calculate attention, and add an residual connection."""
        out = self.self_attn(self.input_layernorm(hidden_states), attention_mask, total_seq_len)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.moe(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def batch_forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.moe(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class MixtralModel(LlamaModel):
    """Exact same as LlamaModel."""

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class MixtralForCasualLM(LlamaForCausalLM):
    """Same as LlamaForCausalLM."""

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.model = MixtralModel(config)
