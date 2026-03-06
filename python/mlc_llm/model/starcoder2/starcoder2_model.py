"""
Implementation for Starcoder2 architecture.
"""

import dataclasses
from typing import Any, Dict, Optional

from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.nn import BaseForCausalLM, PagedKVCache
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Starcoder2Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Starcoder2 model."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_act: str
    norm_epsilon: float
    intermediate_size: int
    rope_theta: int
    use_bias: bool
    use_cache: bool
    bos_token_id: int
    eos_token_id: int
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


class Starcoder2Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Starcoder2Config):
        super().__init__()  # Make sure to call the parent class constructor
        self.hidden_size = config.hidden_size
        self.rope_theta = config.rope_theta
        self.tensor_parallel_shards = config.tensor_parallel_shards
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )

        self.num_heads = config.num_attention_heads // self.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads // self.tensor_parallel_shards
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.context_window_size
        self.use_bias = config.use_bias

        self.wqkv_pack = nn.Linear(
            in_features=self.hidden_size,
            out_features=(self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=self.use_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=self.use_bias
        )

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        qkv = self.wqkv_pack(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        attn_output = self.o_proj(output)
        return attn_output


class Starcoder2MLP(nn.Module):
    def __init__(self, config: Starcoder2Config):
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        embed_dim = config.hidden_size

        self.c_fc = nn.Linear(
            in_features=embed_dim,
            out_features=self.intermediate_size,
            bias=config.use_bias,
        )
        self.c_proj = nn.Linear(self.intermediate_size, embed_dim, bias=config.use_bias)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = op.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Starcoder2DecoderLayer(nn.Module):
    def __init__(self, config: Starcoder2Config):
        self.hidden_size = config.hidden_size
        self.self_attn = Starcoder2Attention(config)
        self.mlp = Starcoder2MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_heads * hd
            k = self.self_attn.num_key_value_heads * hd
            v = self.self_attn.num_key_value_heads * hd
            _set(
                self.self_attn.wqkv_pack.weight,
                tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
            )
            if config.use_bias:
                _set(
                    self.self_attn.wqkv_pack.bias,
                    tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
                )

            _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))

            _set(
                self.mlp.c_fc.weight,
                tp.ShardSingleDim("_shard_c_fc_weight", dim=0),
            )
            if config.use_bias:
                _set(self.mlp.c_fc.bias, tp.ShardSingleDim("_shard_c_fc_bias", dim=0))

            _set(self.mlp.c_proj.weight, tp.ShardSingleDim("_shard_mlp_c_proj", dim=1))

            if config.use_bias:
                _set(
                    self.mlp.c_proj.bias,
                    tp.ShardSingleDim("_shard_mlp_c_proj_bias", dim=0),
                )

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Starcoder2Model(nn.Module):
    def __init__(self, config: Starcoder2Config):
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Starcoder2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_epsilon)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Starcoder2ForCausalLM(BaseForCausalLM):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Starcoder2Config):
        super().__init__()
        self.model = Starcoder2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.tensor_parallel_shards = config.tensor_parallel_shards
