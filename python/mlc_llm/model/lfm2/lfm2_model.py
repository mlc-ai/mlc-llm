"""
Implementation for the LFM2 hybrid architecture: gated short convolutions interleaved with
GQA attention layers.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
from tvm import relax as R
from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Lfm2Config(ConfigBase):
    """Configuration of the LFM2 model."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int = 12288
    norm_eps: float = 1e-5
    conv_L_cache: int = 3
    conv_bias: bool = False
    block_auto_adjust_ff_dim: bool = True
    block_ffn_dim_multiplier: Optional[float] = 1.0
    block_multiple_of: int = 256
    layer_types: Optional[List[str]] = None  # noqa: UP006
    full_attn_idxs: Optional[List[int]] = None  # noqa: UP006
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    head_dim: int = 0
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        if self.tensor_parallel_shards != 1:
            raise ValueError("LFM2 declares no sharding strategies, so it runs on one device")
        # LFM2 and LFM2.5 checkpoints spell these fields differently.
        if "block_ff_dim" in self.kwargs:
            self.intermediate_size = self.kwargs.pop("block_ff_dim")
        if "tie_embedding" in self.kwargs:
            self.tie_word_embeddings = self.kwargs.pop("tie_embedding")
        rope_scaling = self.kwargs.pop("rope_scaling", None)
        rope_parameters = self.kwargs.pop("rope_parameters", None)
        rope = rope_scaling if rope_scaling is not None else (rope_parameters or {})
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise ValueError(f"LFM2 RoPE type {rope_type!r} is not supported")
        if "rope_theta" in rope:
            self.rope_theta = rope["rope_theta"]
        if self.conv_bias:
            raise ValueError("LFM2 with conv_bias is not supported")
        if self.max_batch_size != 1:
            # The conv state cannot be carried through a packed multi-sequence prefill.
            logger.info(
                "Overriding %s from %d to %d", bold("max_batch_size"), self.max_batch_size, 1
            )
            self.max_batch_size = 1
        if self.layer_types is None:
            full_attn_idxs = (
                self.full_attn_idxs
                if self.full_attn_idxs is not None
                else list(range(self.num_hidden_layers))
            )
            self.layer_types = [
                "full_attention" if i in full_attn_idxs else "conv"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, "
                f"expected num_hidden_layers = {self.num_hidden_layers}"
            )
        if self.num_conv_layers == 0:
            # The model compiles as a hybrid, whose RNN state holds the conv layers.
            raise ValueError("LFM2 without a conv layer is not supported")
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
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

    @property
    def ffn_dim(self) -> int:
        """The width of the feed-forward layer, adjusted the way the HF implementation does."""
        ffn_dim = self.intermediate_size
        if self.block_auto_adjust_ff_dim:
            ffn_dim = int(2 * ffn_dim / 3)
            if self.block_ffn_dim_multiplier is not None:
                ffn_dim = int(self.block_ffn_dim_multiplier * ffn_dim)
                multiple = self.block_multiple_of
                ffn_dim = multiple * ((ffn_dim + multiple - 1) // multiple)
        return ffn_dim

    @property
    def num_attention_layers(self) -> int:
        """Number of GQA attention layers."""
        return sum(layer_type == "full_attention" for layer_type in self.layer_types)

    @property
    def num_conv_layers(self) -> int:
        """Number of short convolution layers."""
        return self.num_hidden_layers - self.num_attention_layers


class Lfm2Embedding(nn.Embedding):
    def lm_head_forward(self, x: nn.Tensor):
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class Lfm2MLP(nn.Module):
    def __init__(self, config: Lfm2Config):
        self.ffn_dim = config.ffn_dim
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        gate, up = op.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(op.silu(gate) * up)


class Lfm2Attention(nn.Module):
    def __init__(self, config: Lfm2Config):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, -1, config.norm_eps, bias=False)
        self.k_layernorm = nn.RMSNorm(self.head_dim, -1, config.norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        qkv = op.reshape(self.qkv_proj(hidden_states), (b, s, h_q + 2 * h_kv, d))
        q, k, v = op.split(qkv, [h_q, h_q + h_kv], axis=2)
        # LFM2 normalizes each query and key head before RoPE, which the KV cache applies.
        qkv = op.concat([self.q_layernorm(q), self.k_layernorm(k), v], dim=2)
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, h_q, sm_scale=d**-0.5),
            (b, s, h_q * d),
        )
        return self.out_proj(output)


class Lfm2Conv1d(nn.Module):
    """Holds the depthwise kernel, named as in HF so the weights map one to one."""

    def __init__(self, channels: int, kernel_size: int):
        self.weight = nn.Parameter((channels, 1, kernel_size))


def _causal_depthwise_conv1d(
    x: Tensor, conv_state: Tensor, weight: Tensor
) -> Tuple[Tensor, Tensor]:  # noqa: UP006
    """Causal depthwise conv over x of shape (b, s, d), continuing from conv_state of shape
    (b, kernel_size - 1, d), which holds the last inputs of the previous call.

    Returns the conv output (b, s, d) and the updated state. The kernels are those of
    Qwen3.5's `_causal_conv1d_with_state`. That one is a Qwen35GatedDeltaNet method that reads
    the module's own config and weights, so it cannot be shared without refactoring Qwen3.5.
    """
    kernel_size = weight.shape[2]

    def _te_update_conv_state(old_state: te.Tensor, x_in: te.Tensor):
        history = old_state.shape[1]
        seq = x_in.shape[1]
        return te.compute(
            old_state.shape,
            lambda bi, ti, di: tirx.if_then_else(
                seq + ti < history,
                old_state[bi, seq + ti, di],
                x_in[bi, seq + ti - history, di],
            ),
            name="update_conv_state",
        )

    def _te_depthwise_conv(state: te.Tensor, x_in: te.Tensor, w: te.Tensor):
        history = state.shape[1]
        kk = te.reduce_axis((0, kernel_size), name="kk")
        return te.compute(
            x_in.shape,
            lambda bi, si, di: te.sum(
                tirx.if_then_else(
                    si + kk < history,
                    state[bi, si + kk, di],
                    x_in[bi, si + kk - history, di],
                )
                * w[di, 0, kk],
                axis=kk,
            ),
            name="depthwise_conv1d",
        )

    new_conv_state = op.tensor_expr_op(_te_update_conv_state, "update_conv_state", [conv_state, x])
    output = op.tensor_expr_op(
        _te_depthwise_conv,
        "depthwise_conv1d",
        [conv_state, x, weight],
        attrs={"op_pattern": 8},
    )
    return output, new_conv_state


class Lfm2ShortConv(nn.Module):
    def __init__(self, config: Lfm2Config, conv_layer_id: int):
        self.conv_layer_id = conv_layer_id
        self.kernel_size = config.conv_L_cache
        self.hidden_size = config.hidden_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.conv = Lfm2Conv1d(config.hidden_size, self.kernel_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dtype = "float32"

    def forward(self, hidden_states: Tensor, state: RNNState) -> Tuple[Tensor, RNNState]:  # noqa: UP006
        b, _, d = hidden_states.shape
        # The input projection yields the two gates B and C and the convolved input x.
        gate_b, gate_c, x = op.split(self.in_proj(hidden_states), 3, axis=-1)
        conv_state = state.get(self.conv_layer_id, 0, (b, self.kernel_size - 1, d), self.dtype)
        conv_out, conv_state = _causal_depthwise_conv1d(gate_b * x, conv_state, self.conv.weight)
        state = state.set(self.conv_layer_id, 0, conv_state)
        return self.out_proj(gate_c * conv_out), state


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: Lfm2Config, layer_id: int, category_id: int):
        """category_id is the index of the layer among the layers of its own type, which is the
        index it uses in the KV cache (attention layers) or in the RNN state (conv layers).
        """
        self.is_attention_layer = config.layer_types[layer_id] == "full_attention"
        self.category_id = category_id
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config)
        else:
            self.conv = Lfm2ShortConv(config, category_id)
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, state: RNNState):
        out = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            out = self.self_attn(out, paged_kv_cache, self.category_id)
        else:
            out, state = self.conv(out, state)
        hidden_states = hidden_states + out
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, state


class Lfm2Model(nn.Module):
    def __init__(self, config: Lfm2Config):
        self.embed_tokens = Lfm2Embedding(config.vocab_size, config.hidden_size)
        layers = []
        num_layers_by_type = {"full_attention": 0, "conv": 0}
        for i, layer_type in enumerate(config.layer_types):
            layers.append(Lfm2DecoderLayer(config, i, num_layers_by_type[layer_type]))
            num_layers_by_type[layer_type] += 1
        self.layers = nn.ModuleList(layers)
        # Applied after the last layer despite its name.
        self.embedding_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache, state: RNNState):
        hidden_states = inputs
        for layer in self.layers:
            hidden_states, state = layer(hidden_states, paged_kv_cache, state)
        return self.embedding_norm(hidden_states), state


class Lfm2ForCausalLM(nn.Module):
    def __init__(self, config: Lfm2Config):
        self.model = Lfm2Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_layers = config.num_attention_layers
        self.num_conv_layers = config.num_conv_layers
        self.conv_kernel_size = config.conv_L_cache
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def _forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states, rnn_state = self.model(input_embeds, paged_kv_cache, rnn_state)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache, rnn_state

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state, logit_positions)

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache, rnn_state: RNNState):
        return self._forward(input_embeds, paged_kv_cache, rnn_state)

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache, rnn_state: RNNState):
        return self._forward(input_embeds, paged_kv_cache, rnn_state)

    def create_rnn_state(self, max_batch_size: tirx.Var, max_history: tirx.Var) -> RNNState:
        # Each conv layer keeps the last kernel_size - 1 inputs of its convolution.
        init_values = [
            R.const(np.zeros((self.conv_kernel_size - 1, self.hidden_size), self.dtype)),
        ]
        return RNNState.create(
            max_batch_size=max_batch_size,
            num_hidden_layers=self.num_conv_layers,
            max_history=max_history,
            init_values=init_values,
        )

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
            # Only the attention layers use the KV cache.
            num_hidden_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
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
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
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
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
