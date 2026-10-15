"""
Implementation for the SmolLM3 architecture: Llama with RoPE skipped in some layers (NoPE).
"""

import dataclasses
from typing import List, Optional  # noqa: UP035

from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm import position_embedding

from mlc_llm import op as op_ext
from mlc_llm.model.llama.llama_model import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import tensor_parallel as tp


@dataclasses.dataclass
class SmolLM3Config(LlamaConfig):
    """Configuration of the SmolLM3 model."""

    no_rope_layers: Optional[List[int]] = None  # noqa: UP006
    no_rope_layer_interval: int = 4

    def __post_init__(self):
        # Checked before LlamaConfig.__post_init__, which silently drops some RoPE settings.
        # _apply_rope implements the default RoPE frequencies only.
        if self.rope_scaling is not None:
            raise ValueError("RoPE scaling is not supported for SmolLM3")
        rope_parameters = self.kwargs.pop("rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        if rope_type != "default":
            raise ValueError(f"SmolLM3 RoPE type {rope_type!r} is not supported")
        if "rope_theta" in rope_parameters:
            self.kwargs["rope_theta"] = rope_parameters["rope_theta"]
        if self.position_embedding_base == 0 and "rope_theta" not in self.kwargs:
            raise ValueError("SmolLM3 config must set rope_theta")
        super().__post_init__()
        if self.pipeline_parallel_stages > 1:
            # SmolLM3ForCausalLM replaces the decoder after LlamaForCausalLM has assigned the
            # pipeline stages of its parameters.
            raise ValueError("SmolLM3 does not support pipeline_parallel_stages > 1")
        if self.kwargs.get("use_sliding_window", False):
            raise ValueError("SmolLM3 with sliding window attention is not supported")
        if not self.no_rope_layers:
            # 1 means the layer applies RoPE, as in the HF config.
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0)
                for i in range(self.num_hidden_layers)
            ]
        if len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError(
                f"no_rope_layers has {len(self.no_rope_layers)} entries, "
                f"expected num_hidden_layers = {self.num_hidden_layers}"
            )


def _apply_rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Llama RoPE (rotate-half) on x of shape (tokens, heads, head_dim) at the given positions,
    with the frequencies the KV cache uses for RopeMode.NORMAL.

    As in Gemma 4 (Gemma4TextRotaryEmbedding), RoPE is a TE kernel in the model and the KV cache
    runs with RopeMode.NONE. position_embedding.llama_rope_with_position_map cannot be used
    through op.tensor_ir_op here: call_tir rejects its trailing scalar `apply_rope` parameter.
    """
    head_dim = x.shape[2]

    def _te_rope(x_in: te.Tensor, pos: te.Tensor):
        def _rope(i, h, d):
            cos, sin, var_map = position_embedding.rope_freq_default(
                pos[i].astype("float32"), d, head_dim, theta, "float32"
            )
            rotated = tirx.if_then_else(
                d < head_dim // 2, -x_in[i, h, d + head_dim // 2], x_in[i, h, d - head_dim // 2]
            )
            expr = (cos * x_in[i, h, d].astype("float32") + sin * rotated.astype("float32")).astype(
                x_in.dtype
            )
            for var, value in var_map.items():
                expr = tirx.Let(var, value, expr)
            return expr

        return te.compute(x_in.shape, _rope, name="rope")

    return op.tensor_expr_op(_te_rope, "rope", [x, positions])


class SmolLM3Attention(LlamaAttention):
    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__(config)
        self.use_rope = bool(config.no_rope_layers[layer_idx])
        self.rope_theta = config.position_embedding_base

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        query_positions: Tensor,
    ):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        qkv = op.reshape(self.qkv_proj(hidden_states), (b, s, h_q + 2 * h_kv, d))
        if self.use_rope:
            # The KV cache applies no RoPE because some layers skip it, so apply it here.
            qk, v = op.split(op.reshape(qkv, (b * s, h_q + 2 * h_kv, d)), [h_q + h_kv], axis=1)
            qk = _apply_rope(qk, query_positions, self.rope_theta)
            qkv = op.reshape(op.concat([qk, v], dim=1), (b, s, h_q + 2 * h_kv, d))
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, h_q, sm_scale=d**-0.5),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class SmolLM3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__(config)
        self.self_attn = SmolLM3Attention(config, layer_idx)
        q = self.self_attn.num_q_heads * config.head_dim
        kv = self.self_attn.num_kv_heads * config.head_dim
        self.self_attn.qkv_proj.weight.attrs["shard_strategy"] = tp.ShardSingleDim(
            "_shard_qkv", segs=[q, kv, kv], dim=0
        )
        self.self_attn.o_proj.weight.attrs["shard_strategy"] = tp.ShardSingleDim("_shard_o", dim=1)

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        query_positions: Tensor,
    ):
        out = self.self_attn(
            self.input_layernorm(hidden_states), paged_kv_cache, layer_id, query_positions
        )
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states


class SmolLM3Model(LlamaModel):
    def __init__(self, config: SmolLM3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [SmolLM3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        b, s, _ = input_embed.shape
        query_positions = paged_kv_cache.get_query_positions(b * s)
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            if layer_id != 0 and layer_id in self.layer_partition:
                hidden_states = op_ext.pipeline_stage_boundary(hidden_states)
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id, query_positions)
        return self.norm(hidden_states)


class SmolLM3ForCausalLM(LlamaForCausalLM):
    def __init__(self, config: SmolLM3Config):
        super().__init__(config)
        self.model = SmolLM3Model(config)

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
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
            # RoPE is applied inside the RoPE layers (see SmolLM3Attention).
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=self.rope_theta,
            layer_partition=self.model.layer_partition,
            enable_disaggregation=self.disaggregation,
            dtype=self.dtype,
        )
