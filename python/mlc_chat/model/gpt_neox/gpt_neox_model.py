"""
Implementation for GPTNeoX architecture.
TODO: add docstring
"""
import dataclasses
import logging
import math
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_chat.support.config import ConfigBase
from mlc_chat.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPTNeoXConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPTNeoX model."""

    use_parallel_residual: bool
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    layer_norm_eps: float
    vocab_size: int
    rotary_pct: float
    position_embedding_base: int = 0
    context_window_size: int = 0
    head_dim: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    ffn_out_dtype: str = "float32"
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
                    "Unable to determine the maxmimum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * self.num_attention_heads == self.hidden_size

        if self.prefill_chunk_size == 0:
            # chunk size same as context window size by default
            self.prefill_chunk_size = self.context_window_size

        assert self.tensor_parallel_shards == 1, "GPTNeoX currently does not support sharding."


# pylint: disable=invalid-name,missing-docstring


class RotaryEmbedding(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.position_embedding_base = config.position_embedding_base

    def forward(self, q: Tensor, k: Tensor, offset: tir.Var):
        def te_op(x: te.Tensor, offset: tir.Var):
            dtype = x.dtype

            def compute(b: tir.Var, s: tir.Var, h: tir.Var, d: tir.Var):
                head_dim = tir.const(self.head_dim, "int32")
                position_embedding_base = tir.const(self.position_embedding_base, "float32")
                freq = tir.power(
                    position_embedding_base,
                    (d * 2 % head_dim).astype("float32") / head_dim,
                )
                freq = (offset + s) / freq
                cos = tir.cos(freq).astype(dtype) * x[b, s, h, d]
                sin = tir.sin(freq).astype(dtype) * tir.if_then_else(
                    d < head_dim // 2,
                    -x[b, s, h, d + head_dim // 2],
                    x[b, s, h, d - head_dim // 2],
                )
                return cos + sin

            return te.compute(x.shape, compute, name="rotary")

        q_embed = op.tensor_expr_op(
            te_op,
            "rotary_embedding",
            args=[q, offset],
            attrs={"mlc.rotary_embedding_to_all_dims": True},
        )
        k_embed = op.tensor_expr_op(
            te_op,
            "rotary_embedding",
            args=[k, offset],
            attrs={"mlc.rotary_embedding_to_all_dims": True},
        )
        return q_embed, k_embed


class GPTNeoXAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GPTNeoXConfig, rotary_embedding: RotaryEmbedding):
        self.rotary_embedding = rotary_embedding
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.query_key_value = nn.Linear(
            in_features=self.hidden_size,
            out_features=3 * self.hidden_size,
            bias=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_cache = nn.KVCache(
            config.context_window_size, [self.num_attention_heads, self.head_dim]
        )
        self.v_cache = nn.KVCache(
            config.context_window_size, [self.num_attention_heads, self.head_dim]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,
    ):
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "Only support batch size 1 at this moment."

        # q/k/v states: [batch_size, seq_len, hidden_size]
        qkv = self.query_key_value(hidden_states)
        qkv = op.reshape(qkv, (batch_size, seq_len, 3 * self.num_attention_heads, self.head_dim))
        q, k, v = op.split(
            qkv,
            indices_or_sections=[self.num_attention_heads, 2 * self.num_attention_heads],
            axis=2,
        )
        # q/k/v states: [batch_size, seq_len, num_attention_heads, head_size]
        q, k = self.rotary_embedding(q, k, total_seq_len - seq_len)
        self.k_cache.append(op.squeeze(k, axis=0))
        self.v_cache.append(op.squeeze(v, axis=0))

        # k/v states: [batch_size, total_seq_len, num_attention_heads, head_size]
        k = op.reshape(
            self.k_cache.view(total_seq_len),
            (batch_size, total_seq_len, self.num_attention_heads, self.head_dim),
        )
        v = op.reshape(
            self.v_cache.view(total_seq_len),
            (batch_size, total_seq_len, self.num_attention_heads, self.head_dim),
        )

        # k/v states: [batch_size, num_attention_heads, seq_len, head_size]
        q = q.permute_dims([0, 2, 1, 3])
        # k/v states: [batch_size, num_attention_heads, total_seq_len, head_size]
        k = k.permute_dims([0, 2, 1, 3])
        v = v.permute_dims([0, 2, 1, 3])

        # Calculate QK
        # [batch_size, num_attention_heads, seq_len, head_size]
        # matmul
        # [batch_size, num_attention_heads, head_size, total_seq_len]
        # [batch_size, num_attention_heads, seq_len, totla_seq_len]
        attn_weights = op.matmul(q, k.permute_dims([0, 1, 3, 2])) / math.sqrt(self.head_dim)
        # Apply attention mask
        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype)).minimum(attention_mask)

        # Calculate Softmax(QK)
        if dtype == "float32":
            attn_weights = op.softmax(attn_weights, axis=-1)
        else:
            attn_weights = op.softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)

        # Calculate Softmax(QK)V
        # [batch_size, num_attention_heads, seq_len, totla_seq_len]
        # matmul
        # [batch_size, num_attention_heads, total_seq_len, head_size]
        # [batch_size, num_attention_heads, seq_len, head_size]
        attn_output = op.matmul(attn_weights, v)

        # Apply output projection
        # [batch_size, num_attention_heads, seq_len, head_size]
        # =>
        # [batch_size, seq_len, num_attention_heads, head_size]
        # =>
        # [batch_size, seq_len, hidden_size]
        attn_output = self.dense(
            attn_output.permute_dims([0, 2, 1, 3]).reshape(
                (batch_size, seq_len, self.num_attention_heads * self.head_dim)
            )
        )
        return attn_output


class GPTNeoXMLP(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__()
        out_dtype = config.ffn_out_dtype
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            out_dtype=out_dtype,
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            out_dtype=out_dtype,
        )

    def forward(self, hidden_states: Tensor):
        # dtype = "float16"
        dtype = hidden_states.dtype
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.astype(dtype)
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = op.gelu(hidden_states)
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.astype(dtype)
        hidden_states = self.dense_4h_to_h(hidden_states)
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.astype(dtype)
        return hidden_states


class GPTNeoXLayer(nn.Module):
    def __init__(self, config: GPTNeoXConfig, rotary_embedding: RotaryEmbedding):
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config, rotary_embedding)
        self.mlp = GPTNeoXMLP(config)
        self.use_parallel_residual = config.use_parallel_residual

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,
    ):
        # dtype = "float16"
        dtype = hidden_states.dtype
        attn_input = self.input_layernorm(hidden_states)
        attn_output = self.attention(
            attn_input,
            attention_mask,
            total_seq_len,
        )
        if self.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            attn_output = attn_output + hidden_states
            mlp_input = self.post_attention_layernorm(attn_output)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output.astype(dtype) + attn_output
        return hidden_states


class GPTNeoXModel(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        rotart_embedding = RotaryEmbedding(config)
        self.embed_in = nn.Embedding(num=config.vocab_size, dim=config.hidden_size)
        self.layers = nn.ModuleList(
            [GPTNeoXLayer(config, rotart_embedding) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        hidden_states = self.embed_in(inputs)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class GPTNeoXForCausalLM(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype="float32",
        )
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        def _index(x: te.Tensor):  # x[:, -1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.gpt_neox(inputs, total_seq_len, attention_mask)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.embed_out(hidden_states)
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
