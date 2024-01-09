"""
Implementation for GPT-2 architecture.
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
class GPT2Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPT-2 model."""

    vocab_size: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: int
    n_inner: int = -1
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    scale_attn_by_inverse_layer_idx: bool = False
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.n_inner is None or self.n_inner == -1:
            self.n_inner = 4 * self.n_embd
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
                    "`context_window_size`, `n_positions` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        assert self.tensor_parallel_shards == 1, "GPT2 currently does not support sharding."
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


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class GPT2Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPT2Config, layer_idx: int = None):
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.c_attn = nn.Linear(
            in_features=self.embed_dim,
            out_features=3 * self.num_heads * self.head_dim,
            bias=True,
        )
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

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

        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, 3 * h, d))
        q, k, v = op.split(qkv, 3, axis=2)

        self.k_cache.append(op.squeeze(k, axis=0))
        self.v_cache.append(op.squeeze(v, axis=0))
        k = self.k_cache.view(t)
        v = self.v_cache.view(t)

        if self.scale_attn_by_inverse_layer_idx:
            attn_score_scaling_factor = 1.0 / float(self.layer_idx + 1)
        else:
            attn_score_scaling_factor = 1.0
        output = op_ext.attention(q, k, v, attention_mask, attn_score_scaling_factor)
        return self.c_proj(output)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        embed_dim = config.n_embd
        intermediate_size = config.n_inner
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = op.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int = None):
        hidden_size = config.n_embd
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        hidden_states = (
            self.attn(self.ln_1(hidden_states), attention_mask, total_seq_len) + hidden_states
        )

        hidden_states = self.mlp(self.ln_2(hidden_states)) + hidden_states
        return hidden_states


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        assert config.n_embd % config.n_head == 0
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.context_window_size, config.n_embd)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        # Token Embeddings
        t_embd = self.wte(inputs)

        # Position Embeddings
        # Generate np.arange(offset, offset+seq_len)
        def _input_positions(inputs: te.Tensor, total_seq_len: tir.Var):
            b, s = inputs.shape
            offset = total_seq_len - s
            return te.compute(
                (b, s), lambda _, j: (offset + j).astype("int32"), name="input_positions"
            )

        input_positions = op.tensor_expr_op(
            _input_positions,
            name_hint="input_positions",
            args=[inputs, total_seq_len],
        )
        pos_embd = self.wpe(input_positions)

        # Pass through GPT2Block
        hidden_states = t_embd + pos_embd
        for layer in self.h:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        def _index(x: te.Tensor):  # x[:-1,:]
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
