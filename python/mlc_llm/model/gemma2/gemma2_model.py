"""Implementation for Gemma2 architecture."""

import dataclasses

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.model.gemma.gemma_model import (
    GemmaAttention,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
)
from mlc_llm.nn import PagedKVCache
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Gemma2Config(GemmaConfig):
    """Configuration of the Gemma2 model, in addition to the Gemma model"""

    # NOTE: We ignore attn_logit_softcapping in the gemma2 implementation for now.
    # The Gemma 2 team observed minor differences when soft-capping is removed during inference,
    # according to https://huggingface.co/blog/gemma2.
    # The soft-capping is also not supported by HuggingFace transformers `Gemma2SdpaAttention`.
    attn_logit_softcapping: float = None
    final_logit_softcapping: float = None
    query_pre_attn_scalar: int = None
    sliding_window: int = None

    def __post_init__(self):
        super().__post_init__()
        # NOTE: override the context window size with the Gemma2 sliding window size,
        # as the sliding window attention every other layer is yet to be supported.
        self.context_window_size = self.sliding_window


# pylint: disable=invalid-name,missing-docstring


class Gemma2Attention(GemmaAttention):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.scaling_factor = (config.head_dim / config.query_pre_attn_scalar) ** 0.5


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Gemma2Config):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = Gemma2Attention(config)
        self.mlp = GemmaMLP(config)
        # Gemma RMSNorm adds 1 to the weights. It is already fused in the loader
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, -1, rms_norm_eps, bias=False
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, -1, rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
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


class Gemma2Model(GemmaModel):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class Gemma2ForCausalLM(GemmaForCausalLM):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.model = Gemma2Model(config)
        self.final_logit_softcapping = config.final_logit_softcapping

    def get_logits(self, hidden_states: Tensor):
        logits = super().get_logits(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = op.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits
