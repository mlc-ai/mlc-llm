"""
Implementation for QWEN2 architecture.
"""

import dataclasses
from functools import partial
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_chat import op as op_ext
from mlc_chat.support import logging
from mlc_chat.support.config import ConfigBase
from mlc_chat.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class QWen2Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the QWen model."""

    hidden_act: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: int
    vocab_size: int

    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    dtype: str = "float32"
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
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %s (%d)",
                bold("prefill_chunk_size"),
                bold("context_window_size"),
                self.context_window_size,
            )
            self.prefill_chunk_size = self.context_window_size
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d (%s)",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                self.context_window_size,
                bold("context_window_size"),
            )
            self.prefill_chunk_size = self.context_window_size
        assert self.tensor_parallel_shards == 1, "QWEN currently does not support sharding."


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class QWen2Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: QWen2Config):
        head_dim = config.hidden_size // config.num_attention_heads

        self.c_attn = nn.Linear(
            in_features=config.hidden_size,
            out_features=(2 * config.num_key_value_heads + config.num_attention_heads) * head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * head_dim, config.hidden_size, bias=False
        )
        # KV cache for single sequence
        self.k_cache = nn.KVCache(
            config.context_window_size, [config.num_key_value_heads, head_dim]
        )
        self.v_cache = nn.KVCache(
            config.context_window_size, [config.num_attention_heads, head_dim]
        )

        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        bsz, sl, _ = hidden_states.shape
        assert bsz == 1, "Only support batch size 1 at this moment."
        # Step 1. QKV Projection
        qkv = self.c_attn(hidden_states)
        num_heads = 2 * self.num_key_value_heads + self.num_attention_heads
        qkv = op.reshape(qkv, (bsz, sl, num_heads, self.head_dim))
        # Step 2. Apply QK rotary embedding
        q, k, v = op_ext.llama_rope(
            qkv, total_seq_len, self.rope_theta, self.num_attention_heads, self.num_key_value_heads
        )
        # Step 3. Query and update KVCache
        self.k_cache.append(op.squeeze(k, axis=0))
        self.v_cache.append(op.squeeze(v, axis=0))
        k = self.k_cache.view(total_seq_len)
        v = self.v_cache.view(total_seq_len)
        # Step 4. Compute softmax(Q @ K^T / sqrt(d)) @ V
        output = op_ext.attention(q, k, v, casual_mask=attention_mask)
        # Step 5. Apply output projection
        return self.o_proj(output)


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class QWen2MLP(nn.Module):
    def __init__(self, config: QWen2Config):
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class QWen2DecoderLayer(nn.Module):
    def __init__(self, config: QWen2Config):
        self.self_attn = QWen2Attention(config)
        self.mlp = QWen2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, attention_mask, total_seq_len)
        hidden_states = out + hidden_states

        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = out + hidden_states
        return hidden_states


class QWen2Model(nn.Module):
    def __init__(self, config: QWen2Config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QWen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class QWen2LMHeadModel(nn.Module):
    def __init__(self, config: QWen2Config):
        self.model = QWen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(inputs, attention_mask, total_seq_len)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(self, inputs: Tensor, total_seq_len: tir.Var):
        def _attention_mask(batch_size, seq_len, total_seq_len):
            return te.compute(
                (batch_size, 1, seq_len, total_seq_len),
                lambda b, _, i, j: tir.if_then_else(
                    i < j - (total_seq_len - seq_len),
                    tir.min_value(self.dtype),
                    tir.max_value(self.dtype),
                ),
                name="attention_mask_prefill",
            )

        batch_size, seq_len = inputs.shape
        attention_mask = op.tensor_expr_op(
            _attention_mask,
            name_hint="attention_mask_prefill",
            args=[batch_size, seq_len, total_seq_len],
        )
        return self.forward(inputs, attention_mask, total_seq_len)

    def decode(self, inputs: Tensor, total_seq_len: tir.Var):
        batch_size, seq_len = inputs.shape
        attention_mask = op.full(
            shape=[batch_size, 1, seq_len, total_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        return self.forward(inputs, attention_mask, total_seq_len)

    @staticmethod
    def softmax_with_temperature(logits: Tensor, temperature: Tensor):
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        batch_size = 1
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "total_seq_len": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "total_seq_len": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
