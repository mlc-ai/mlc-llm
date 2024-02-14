"""
Implementation for GPTJ architecture.
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
class GPTJConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPTJ model."""

    vocab_size: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: int
    rotary_dim: int
    n_inner: int = -1
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.context_window_size == 0:
            for name in ["n_positions", "max_sequence_length"]:
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
                    "Unable to determine the maxmimum sequence length, because none of "
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
        assert self.tensor_parallel_shards == 1, "GPTJ currently does not support sharding."


# pylint: disable=invalid-name,missing-docstring

class GPTJAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPTJConfig):
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head // config.tensor_parallel_shards
        self.head_dim = self.embed_dim // self.num_heads
        self.rope_theta = config.rotary_dim

        self.c_attn = nn.Linear(
            in_features=self.embed_dim,
            out_features=3 * self.num_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

        self.k_cache = nn.KVCache(config.context_window_size, [self.num_heads, self.head_dim])
        self.v_cache = nn.KVCache(config.context_window_size, [self.num_heads, self.head_dim])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,
    ):
        d, h, t = self.head_dim, self.num_heads, total_seq_len
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."
        # Step 1. QKV Projection
        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, 3 * h, d))
        # Step 2. Apply QK rotary embedding
        q, k, v = op_ext.llama_rope(qkv, t, self.rope_theta, h, h)
        # Step 3. Query and update KVCache
        self.k_cache.append(op.squeeze(k, axis=0))
        self.v_cache.append(op.squeeze(v, axis=0))
        k = self.k_cache.view(t)
        v = self.v_cache.view(t)
        # Step 4. Compute softmax(Q @ K^T / sqrt(d)) @ V
        output = op_ext.attention(q, k, v, attention_mask)
        # Step 5. Apply output projection
        return self.out_proj(output)


class GPTJMLP(nn.Module):
    def __init__(self, config: GPTJConfig):  # in MLP: intermediate_size= 4 * embed_dim

        embed_dim = config.n_embd
        intermediate_size = 4 * embed_dim // config.tensor_parallel_shards
        self.fc_in = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.fc_out = nn.Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):
    def __init__(self, config: GPTJConfig):
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask, total_seq_len)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual
        return hidden_states

class GPTJModel(nn.Module):
    def __init__(self, config: GPTJConfig):
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.h = nn.ModuleList([GPTJBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        hidden_states = self.wte(inputs)
        for layer in self.h:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTJForCausalLM(nn.Module):
    def __init__(self, config: GPTJConfig):
        self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, dtype="float32")
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        def _index(x: te.Tensor):  # x[:, -1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.transformer(inputs, total_seq_len, attention_mask)
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
