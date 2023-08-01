"""Implementation for Llama2 architecture"""
import dataclasses
import math
from typing import Optional

import numpy as np
from tvm import te, tir
from tvm.relax.frontend.nn import (
    Embedding,
    KVCache,
    Linear,
    Module,
    ModuleList,
    RMSNorm,
    Tensor,
    full,
    matmul,
    reshape,
    silu,
    softmax,
    squeeze,
    tensor_expr_op,
)

# pylint: disable=invalid-name,missing-docstring


@dataclasses.dataclass
class LlamaConfig:  # pylint: disable=too-many-instance-attributes
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    max_sequence_length: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    position_embedding_base: int

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    initializer_range: float
    model_type: str
    torch_dtype: str
    num_key_value_heads: int = 0

    def __post_init__(self):
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @classmethod
    def from_dict(cls, d: dict) -> "LlamaConfig":
        field_names = (field.name for field in dataclasses.fields(cls))
        return cls(**{k: v for k, v in d.items() if k in field_names})


class RotaryEmbedding(Module):
    def __init__(self, position_embedding_base: int, max_seq_len: int, rotary_dim: int):
        super().__init__()
        self.position_embedding_base = position_embedding_base
        self.max_seq_len = max_seq_len
        self.rotary_dim = rotary_dim

    def forward(self, q: Tensor, k: Tensor, offset: tir.Var):
        def te_op(x: te.Tensor, cos: te.Tensor, sin: te.Tensor, offset: tir.Var):
            def compute(b: tir.Var, s: tir.Var, h: tir.Var, d: tir.Var):
                result = cos[offset + s, d] * x[b, s, h, d] + sin[offset + s, d] * tir.if_then_else(
                    d < self.rotary_dim // 2,
                    -x[b, s, h, d + self.rotary_dim // 2],
                    x[b, s, h, d - self.rotary_dim // 2],
                )
                return tir.if_then_else(d < self.rotary_dim, result, x[b, s, h, d])

            return te.compute(x.shape, compute, name="rotary")

        r0 = np.arange(0, self.rotary_dim, 2, dtype="float32")
        r1 = np.arange(0, self.max_seq_len, dtype="float32")
        inv_freq = 1.0 / (self.position_embedding_base ** (r0 / self.rotary_dim))
        freq = np.einsum("i,j->ij", r1, inv_freq)
        emb = np.concatenate((freq, freq), axis=-1)
        cos = Tensor.from_const(np.cos(emb).astype(q.dtype))
        sin = Tensor.from_const(np.sin(emb).astype(q.dtype))
        q_embed = tensor_expr_op(te_op, "rotary_embedding", args=[q, cos, sin, offset])
        k_embed = tensor_expr_op(te_op, "rotary_embedding", args=[k, cos, sin, offset])
        return q_embed, k_embed


class LlamaFFN(Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        return self.down_proj(silu(x1) * x2)


class LlamaAttention(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: LlamaConfig, rotary_embedding: RotaryEmbedding):
        head_dim = config.hidden_size // config.num_attention_heads

        self.rotary_embedding = rotary_embedding
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim

        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.q_proj = Linear(config.hidden_size, self.num_q_heads * head_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_cache = KVCache(config.max_sequence_length, [self.num_kv_heads, head_dim])
        self.v_cache = KVCache(config.max_sequence_length, [self.num_kv_heads, head_dim])

    def forward(  # pylint: disable=too-many-locals
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,
    ):
        d, h_q, h_kv, t = self.head_dim, self.num_q_heads, self.num_kv_heads, total_seq_len
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."
        q = reshape(self.q_proj(hidden_states), (b, s, h_q, d))
        k = reshape(self.k_proj(hidden_states), (b, s, h_kv, d))
        v = reshape(self.v_proj(hidden_states), (b, s, h_kv, d))
        q, k = self.rotary_embedding(q, k, t - s)

        self.k_cache.append(squeeze(k, axis=0))
        self.v_cache.append(squeeze(v, axis=0))
        k = reshape(self.k_cache.view(total_seq_len), (t, b, h_kv, d))
        v = reshape(self.v_cache.view(total_seq_len), (t, b, h_kv, d))
        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=2)
            v = v.repeat(h_q // h_kv, axis=2)
        attn_weights = matmul(  # [b, h, s, t]
            q.permute_dims([0, 2, 1, 3]),  # [b, h, s, d]
            k.permute_dims([1, 2, 3, 0]),  # [b, h, d, t]
        ) / math.sqrt(d)
        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype)).minimum(attention_mask)
        if dtype == "float32":
            attn_weights = softmax(attn_weights, axis=-1)
        else:
            attn_weights = softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
        return self.o_proj(
            matmul(  # [b, h, s, d]
                attn_weights,  # [b, h, s, t]
                v.permute_dims([1, 2, 0, 3]),  # [b, h, t, d]
            )
            .permute_dims([0, 2, 1, 3])  # [b, s, h, d]
            .reshape((b, s, h_q * d))
        )


class LlamaDecoderLayer(Module):
    def __init__(self, config: LlamaConfig, rotary_embedding: RotaryEmbedding):
        self.attn = LlamaAttention(config, rotary_embedding)
        self.ffn = LlamaFFN(config)
        self.input_norm = RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_norm = RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        hidden_states = (
            self.attn(self.input_norm(hidden_states), attention_mask, total_seq_len) + hidden_states
        )
        hidden_states = self.ffn(self.post_attention_norm(hidden_states)) + hidden_states
        return hidden_states


class LlamaModel(Module):
    def __init__(self, config: LlamaConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_embedding = RotaryEmbedding(
            position_embedding_base=config.position_embedding_base,
            max_seq_len=config.max_sequence_length,
            rotary_dim=head_dim,
        )
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList(
            [LlamaDecoderLayer(config, rotary_embedding) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        hidden_states = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCasualLM(Module):
    def __init__(self, config: LlamaConfig, dtype: str = "float32"):
        self.model = LlamaModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.dtype = dtype

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(inputs, total_seq_len, attention_mask)
        hidden_states = tensor_expr_op(_index, name_hint="index", args=[hidden_states])
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
        attention_mask = tensor_expr_op(
            _attention_mask,
            name_hint="attention_mask_prefill",
            args=[batch_size, seq_len, total_seq_len],
        )
        return self.forward(inputs, total_seq_len, attention_mask)

    def decode(self, inputs: Tensor, total_seq_len: tir.Var):
        batch_size, seq_len = inputs.shape
        attention_mask = full(
            shape=[batch_size, 1, seq_len, total_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        return self.forward(inputs, total_seq_len, attention_mask)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        return softmax(logits / temperature, axis=-1)
