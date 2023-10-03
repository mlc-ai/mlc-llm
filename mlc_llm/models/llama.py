"""Implementation for Llama2 architecture"""
import dataclasses
import math
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .model_config_base import ModelConfig

# pylint: disable=invalid-name,missing-docstring


@dataclasses.dataclass
class LlamaConfig(ModelConfig):  # pylint: disable=too-many-instance-attributes
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    max_sequence_length: int = 2048
    position_embedding_base: int = 10000
    num_key_value_heads: int = 0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    head_dim: int = 0

    def __post_init__(self):
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size


class RotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.position_embedding_base = config.position_embedding_base

    def forward(self, q: Tensor, k: Tensor, offset: tir.Var):
        def te_op(x: te.Tensor, offset: tir.Var):
            def compute(b: tir.Var, s: tir.Var, h: tir.Var, d: tir.Var):
                head_dim = tir.const(self.head_dim, "int32")
                position_embedding_base = tir.const(self.position_embedding_base, "float32")
                freq = tir.power(
                    position_embedding_base,
                    (d * 2 % head_dim).astype("float32") / head_dim,
                )
                freq = (offset + s) / freq
                return tir.cos(freq) * x[b, s, h, d] + tir.sin(freq) * tir.if_then_else(
                    d < self.head_dim // 2,
                    -x[b, s, h, d + self.head_dim // 2],
                    x[b, s, h, d - self.head_dim // 2],
                )

            return te.compute(x.shape, compute, name="rotary")

        q_embed = op.tensor_expr_op(te_op, "rotary_embedding", args=[q, offset])
        k_embed = op.tensor_expr_op(te_op, "rotary_embedding", args=[k, offset])
        return q_embed, k_embed


class LlamaFFN(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_up_proj = nn.MultiLinear(
            in_features=config.hidden_size,
            out_features=[config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        x1, x2 = self.gate_up_proj(x)
        return self.down_proj(op.silu(x1) * x2)


class LlamaAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: LlamaConfig, rotary_embedding: RotaryEmbedding):
        self.rotary_embedding = rotary_embedding
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.qkv_proj = nn.MultiLinear(
            in_features=config.hidden_size,
            out_features=[
                self.num_q_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_cache = nn.KVCache(config.max_sequence_length, [self.num_kv_heads, self.head_dim])
        self.v_cache = nn.KVCache(config.max_sequence_length, [self.num_kv_heads, self.head_dim])

    def forward(  # pylint: disable=too-many-locals
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,
    ):
        d, h_q, h_kv, t = self.head_dim, self.num_q_heads, self.num_kv_heads, total_seq_len
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."
        q, k, v = self.qkv_proj(hidden_states)
        q = op.reshape(q, (b, s, h_q, d))
        k = op.reshape(k, (b, s, h_kv, d))
        v = op.reshape(v, (b, s, h_kv, d))
        q, k = self.rotary_embedding(q, k, t - s)

        self.k_cache.append(op.squeeze(k, axis=0))
        self.v_cache.append(op.squeeze(v, axis=0))
        k = op.reshape(self.k_cache.view(total_seq_len), (t, b, h_kv, d))
        v = op.reshape(self.v_cache.view(total_seq_len), (t, b, h_kv, d))
        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=2)
            v = v.repeat(h_q // h_kv, axis=2)
        attn_weights = op.matmul(  # [b, h, s, t]
            q.permute_dims([0, 2, 1, 3]),  # [b, h, s, d]
            k.permute_dims([1, 2, 3, 0]),  # [b, h, d, t]
        ) / math.sqrt(d)
        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype)).minimum(attention_mask)
        if dtype == "float32":
            attn_weights = op.softmax(attn_weights, axis=-1)
        else:
            attn_weights = op.softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
        return self.o_proj(
            op.matmul(  # [b, h, s, d]
                attn_weights,  # [b, h, s, t]
                v.permute_dims([1, 2, 0, 3]),  # [b, h, t, d]
            )
            .permute_dims([0, 2, 1, 3])  # [b, s, h, d]
            .reshape((b, s, h_q * d))
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, rotary_embedding: RotaryEmbedding):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = LlamaAttention(config, rotary_embedding)
        self.mlp = LlamaFFN(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        hidden_states = (
            self.self_attn(self.input_layernorm(hidden_states), attention_mask, total_seq_len)
            + hidden_states
        )
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states)) + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        rotary_embedding = RotaryEmbedding(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, rotary_embedding) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        hidden_states = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCasualLM(nn.Module):
    def __init__(self, config: LlamaConfig, dtype: str = "float32"):
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "total_seq_len": int,
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
