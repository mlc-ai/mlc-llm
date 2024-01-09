"""
Implementation for GPTBigCode architecture.
TODO: add docstring
"""
import dataclasses
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_chat import op as op_ext
from mlc_chat.support import logging
from mlc_chat.support import tensor_parallel as tp
from mlc_chat.support.config import ConfigBase
from mlc_chat.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPTBigCodeConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPTBigCode model."""

    n_embd: int
    n_inner: int
    n_head: int
    n_layer: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.context_window_size == 0:
            if self.n_positions > 0:
                self.context_window_size = self.n_positions
                logger.info(
                    "%s not found in config.json. Falling back to %s (%d)",
                    bold("context_window_size"),
                    bold("n_positions"),
                    self.context_window_size,
                )
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


# pylint: disable=invalid-name,missing-docstring


class GPTBigCodeMLP(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        super().__init__()
        self.n_inner = config.n_inner // config.tensor_parallel_shards
        self.c_fc = nn.Linear(in_features=config.n_embd, out_features=self.n_inner, bias=True)
        self.c_proj = nn.Linear(in_features=self.n_inner, out_features=config.n_embd, bias=True)

    def forward(self, x: Tensor):
        hidden_states = self.c_fc(x)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPTBigCodeAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPTBigCodeConfig):
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.num_q_heads = config.n_head // config.tensor_parallel_shards
        self.num_kv_heads = 1
        assert (
            config.tensor_parallel_shards == 1
        ), "GPT bigcode only support tensor parallel shards = 1"
        self.c_attn = nn.Linear(
            in_features=self.n_embd,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=True,
        )
        self.c_proj = nn.Linear(
            in_features=self.num_q_heads * self.head_dim,
            out_features=config.n_embd,
            bias=True,
        )

        self.k_cache = nn.KVCache(config.context_window_size, [self.num_kv_heads, self.head_dim])
        self.v_cache = nn.KVCache(config.context_window_size, [self.num_kv_heads, self.head_dim])

    def forward(  # pylint: disable=too-many-locals
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,
    ):
        d, h_q, h_kv, t = self.head_dim, self.num_q_heads, self.num_kv_heads, total_seq_len
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."

        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + 2 * h_kv, d))
        q, k, v = op.split(qkv, indices_or_sections=[h_q, h_q + h_kv], axis=2)

        self.k_cache.append(op.squeeze(k, axis=0))
        self.v_cache.append(op.squeeze(v, axis=0))
        k = self.k_cache.view(t)
        v = self.v_cache.view(t)
        output = op_ext.attention(q, k, v, casual_mask=attention_mask)
        return self.c_proj(output)


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPTBigCodeMLP(config)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.n_embd // config.n_head
            q = config.n_head * hd
            k = 1 * hd
            v = 1 * hd
            _set(self.attn.c_attn, tp.ShardSingleDim("_shard_c_attn", dim=0, segs=[q, k, v]))
            _set(self.attn.c_proj, tp.ShardSingleDim("_shard_c_proj", dim=1))
            _set(self.mlp.c_fc, tp.ShardSingleDim("_shard_mlp_c_fc", dim=0))
            _set(self.mlp.c_proj, tp.ShardSingleDim("_shard_mlp_c_proj", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, total_seq_len: tir.Var):
        hidden_states = (
            self.attn(self.ln_1(hidden_states), attention_mask, total_seq_len) + hidden_states
        )
        hidden_states = self.mlp(self.ln_2(hidden_states)) + hidden_states
        return hidden_states


class GPTBigCodeModel(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        assert config.n_embd % config.n_head == 0
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([GPTBigCodeBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        if self.tensor_parallel_shards > 1:
            inputs = op.ccl_broadcast_from_worker0(inputs)

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

        # apply position embeddings
        hidden_states = t_embd + pos_embd
        for layer in self.h:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class GPTBigCodeForCausalLM(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        self.transformer = GPTBigCodeModel(config)
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
