"""
Implementation for Phi architecture.
TODO: add docstring
"""
import dataclasses
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
class PhiConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Phi model."""

    vocab_size: int = 51200
    n_positions: int = 2048
    n_embd: int = 2560
    n_layer: int = 32
    n_inner: int = 0
    n_head: int = 32
    rotary_dim: int = 32
    position_embedding_base: int = 0
    layer_norm_epsilon: float = 1e-5
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    n_head_kv: int = 0
    head_dim: int = 0
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
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
                self.context_window_size = self.n_positions
                logger.info(
                    "%s not found in config.json. Falling back to %s (%d)",
                    bold("context_window_size"),
                    "n_positions",
                    self.context_window_size,
                )
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = self.context_window_size
        if self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = self.context_window_size
        if self.n_head_kv == 0 or self.n_head_kv is None:
            self.n_head_kv = self.n_head
        if self.n_inner == 0 or self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        if self.head_dim == 0:
            self.head_dim = self.n_embd // self.n_head
        assert self.head_dim * self.n_head == self.n_embd
        assert self.n_head % self.n_head_kv == 0


# pylint: disable=invalid-name,missing-docstring


class PhiMLP(nn.Module):
    def __init__(self, config: PhiConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.n_embd, config.n_inner)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = op.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class PhiCrossAttention(nn.Module):
    def __init__(self, config: PhiConfig):  # pylint: disable=unused-argument
        super().__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor):
        output = op_ext.attention(q, k, v, casual_mask=attention_mask)
        return output


class PhiMHA(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: PhiConfig):
        self.rope_theta = config.position_embedding_base
        self.rotary_dim = config.rotary_dim
        self.n_head = config.n_head
        self.n_head_kv = config.n_head_kv
        self.head_dim = config.head_dim
        op_size = self.head_dim * (self.n_head + 2 * self.n_head_kv)
        hidden_size = config.n_embd

        self.Wqkv = nn.Linear(hidden_size, op_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.inner_cross_attn = PhiCrossAttention(config)
        self.k_cache = nn.KVCache(config.context_window_size, [self.n_head_kv, self.head_dim])
        self.v_cache = nn.KVCache(config.context_window_size, [self.n_head_kv, self.head_dim])

    def forward(self, x: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        d, h_q, h_kv, t = self.head_dim, self.n_head, self.n_head_kv, total_seq_len
        b, s, _ = x.shape
        assert b == 1, "Only support batch size 1 at this moment."
        # Step 1. QKV Projection
        qkv = self.Wqkv(x)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        # Step 2. Apply QK rotary embedding
        q, k, v = op_ext.llama_rope(qkv, t, self.rope_theta, h_q, h_kv, rotary_dim=self.rotary_dim)
        # Step 3. Query and update KVCache
        self.k_cache.append(op.squeeze(k, axis=0))
        self.v_cache.append(op.squeeze(v, axis=0))
        k = self.k_cache.view(t)
        v = self.v_cache.view(t)
        # Step 4. Compute softmax(Q @ K^T / sqrt(d)) @ V
        output = self.inner_cross_attn(q, k, v, attention_mask)
        # Step 5. Apply output projection
        return self.out_proj(output)


class PhiParallelBlock(nn.Module):
    def __init__(self, config: PhiConfig):
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mixer = PhiMHA(config)
        self.mlp = PhiMLP(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs = self.mixer(
            hidden_states,
            attention_mask,
            total_seq_len,
        )

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states


class PhiCausalLMHead(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.ln(hidden_states)
        logits = self.linear(hidden_states)

        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits


class PhiModel(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([PhiParallelBlock(config) for i in range(config.n_layer)])

    def forward(self, input_ids: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        hidden_states = self.embd(input_ids)
        for layer in self.h:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)

        return hidden_states


class PhiForCausalLM(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()

        self.transformer = PhiModel(config)
        self.lm_head = PhiCausalLMHead(config)
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, input_ids: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.transformer(input_ids, total_seq_len, attention_mask)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        lm_logits = self.lm_head(hidden_states)

        return lm_logits

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
        return self.forward(inputs, total_seq_len, attention_mask)

    def decode(self, inputs: Tensor, total_seq_len: tir.Var):
        batch_size, seq_len = inputs.shape
        attention_mask = op.full(
            shape=[batch_size, 1, seq_len, total_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        return self.forward(inputs, total_seq_len, attention_mask)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
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
