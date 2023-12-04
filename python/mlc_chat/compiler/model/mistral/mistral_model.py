"""
Implementation for Mistral architecture.
"""
import dataclasses
import math
from typing import Any, Dict, Optional

from tvm import relax as rx
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from ....support import logging
from ....support.config import ConfigBase
from ....support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MistralConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Mistral model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    position_embedding_base: int = 0
    context_window_size: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    sliding_window: int = 4096
    prefill_chunk_size: int = 0
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
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size

        if self.prefill_chunk_size == 0:
            # chunk size same as sliding window by default
            self.prefill_chunk_size = self.sliding_window
        self.context_window_size = -1
        logger.info(
            "Using sliding window attention, setting %s to -1",
            bold("context_window_size"),
        )


# pylint: disable=invalid-name,missing-docstring


class RotaryEmbedding(nn.Module):
    """Same as in Llama architecture."""

    def __init__(self, config: MistralConfig):
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

        q_embed = op.tensor_expr_op(te_op, "rotary_embedding", args=[q, offset])
        k_embed = op.tensor_expr_op(te_op, "rotary_embedding", args=[k, offset])
        return q_embed, k_embed


class MistralMLP(nn.Module):
    """Same as in Llama architecture (LlamaFFN)."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


class MistralAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Same as LlamaAttention, but with sliding window attention using a rolling buffer cache."""

    def __init__(self, config: MistralConfig, rotary_embedding: RotaryEmbedding):
        self.rotary_embedding = rotary_embedding
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_cache = RollingKVCache(self.sliding_window, [self.num_kv_heads, self.head_dim])
        self.v_cache = RollingKVCache(self.sliding_window, [self.num_kv_heads, self.head_dim])

    def interleave_kv(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        k_cur: Tensor,
        v_cur: Tensor,
        total_seq_len: tir.Var,
        kv_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
    ):
        """Unrotate and concatenate currunt and cached k and v"""
        d, _, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        t, kv_s, c = total_seq_len, kv_seq_len, rolling_cache_len
        b, s, _, _ = k_cur.shape
        cache_offset = (t - s) % self.sliding_window

        k_cached = op.reshape(self.k_cache.view(c), (b, c, h_kv, d))
        v_cached = op.reshape(self.v_cache.view(c), (b, c, h_kv, d))

        def _unrotate_concat(x_cur, x_cached, cache_offset, rolling_cache_len):
            return te.compute(
                (b, kv_s, h_kv, d),
                lambda xb, xs, xh, xd: te.if_then_else(
                    xs < rolling_cache_len - cache_offset,
                    x_cached[xb, cache_offset + xs, xh, xd],
                    te.if_then_else(
                        xs < rolling_cache_len,
                        x_cached[xb, xs + cache_offset - rolling_cache_len, xh, xd],
                        x_cur[xb, xs - rolling_cache_len, xh, xd],
                    ),
                ),
                name="unrotate_concat_te",
            )

        k = op.tensor_expr_op(
            _unrotate_concat,
            name_hint="te_unrotate_concat_key",
            args=[k_cur, k_cached, cache_offset, c],
        )
        v = op.tensor_expr_op(
            _unrotate_concat,
            name_hint="te_unrotate_concat_value",
            args=[v_cur, v_cached, cache_offset, c],
        )

        self.k_cache.override(op.squeeze(k_cur, axis=0), self.sliding_window)
        self.v_cache.override(op.squeeze(v_cur, axis=0), self.sliding_window)

        return k, v

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,  # Number of already-processed tokens plus ``seq_len``.
        rolling_cache_len: tir.Var,  # Number of elements currently in the cache.
        kv_seq_len: tir.Var,  # Equals to ``seq_len + rolling_cache_len``.
    ):
        """Forward pass of MistralAttention, performing QKV."""
        d, h_q, h_kv, t = self.head_dim, self.num_q_heads, self.num_kv_heads, total_seq_len
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."

        qkv_cur = self.qkv_proj(hidden_states)
        qkv_cur = op.reshape(qkv_cur, (b, s, h_q + 2 * h_kv, d))
        q, k_cur, v_cur = op.split(qkv_cur, [h_q, h_q + h_kv], axis=2)
        q, k_cur = self.rotary_embedding(q, k_cur, t - s)

        k, v = self.interleave_kv(k_cur, v_cur, total_seq_len, kv_seq_len, rolling_cache_len)

        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=2)
            v = v.repeat(h_q // h_kv, axis=2)
        q = q.permute_dims([0, 2, 1, 3])  # [b, h, s, d]
        k = k.permute_dims([0, 2, 1, 3])  # [b, h, t, d]
        v = v.permute_dims([0, 2, 1, 3])  # [b, h, t, d]
        attn_weights = op.matmul(
            q, k.permute_dims([0, 1, 3, 2])  # [b, h, s, d] x [b, h, d, t] = [b, h, s, t]
        ) / math.sqrt(d)
        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype)).minimum(attention_mask)
        if dtype == "float32":
            attn_weights = op.softmax(attn_weights, axis=-1)
        else:
            attn_weights = op.softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
        # [b, h, s, t] x [b, h, t, d] => [b, h, s, d] => [b, s, h, d]
        output = op.matmul(attn_weights, v)
        return self.o_proj(output.permute_dims([0, 2, 1, 3]).reshape((b, s, h_q * d)))


class RollingKVCache(nn.KVCache):
    """
    Rolling buffer cache implementation.
    """

    cache: Optional[rx.Var]

    def override(self, new_element: Tensor, max_cache_size: int) -> None:
        """
        Override cache elements in RollingKVCache.

        Parameters
        ----------
        new_element : Tensor
            The new tensor to append.

        max_cache_size : int
            Max size of the cache.
        """
        if new_element.dtype != self.dtype:
            raise TypeError(
                f'RollingKVCache has been set to use dtype "{self.dtype}", '
                f'but got "{new_element.dtype}"'
            )
        self.cache = rx.BlockBuilder.current().emit(
            rx.Call(
                rx.extern("vm.builtin.attention_kv_cache_window_override"),
                args=[
                    self.cache,
                    new_element._expr,  # pylint: disable=protected-access
                    rx.PrimValue(max_cache_size),
                ],
                sinfo_args=[rx.ObjectStructInfo()],
            )
        )


class MistralDecoderLayer(nn.Module):
    """Exact same as LlamaDecoderLayer."""

    def __init__(self, config: MistralConfig, rotary_embedding: RotaryEmbedding):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = MistralAttention(config, rotary_embedding)
        self.mlp = MistralMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

    def forward(  # pylint: disable=too-many-arguments
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        total_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
    ):
        """Forward pass of a decoder layer; calculate attention, and add an residual connection."""
        hidden_states = (
            self.self_attn(
                self.input_layernorm(hidden_states),
                attention_mask,
                total_seq_len,
                rolling_cache_len,
                kv_seq_len,
            )
            + hidden_states
        )
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states)) + hidden_states
        return hidden_states


class MistralModel(nn.Module):
    """Exact same as LlamaModel."""

    def __init__(self, config: MistralConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        rotary_embedding = RotaryEmbedding(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, rotary_embedding) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(  # pylint: disable=too-many-arguments
        self,
        inputs: Tensor,
        total_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        attention_mask: Tensor,
    ):
        """Forward pass of the model, passing through all decoder layers."""
        hidden_states = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, total_seq_len, rolling_cache_len, kv_seq_len
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MistralForCasualLM(nn.Module):
    """Same as LlamaForCausalLM, except for the use of sliding window attention."""

    def __init__(self, config: MistralConfig):
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.sliding_window = config.sliding_window
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(  # pylint: disable=too-many-arguments
        self,
        inputs: Tensor,
        total_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        attention_mask: Tensor,
    ):
        """Forward pass."""

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(
            inputs, total_seq_len, rolling_cache_len, kv_seq_len, attention_mask
        )
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(
        self,
        inputs: Tensor,
        total_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
    ):
        """
        Prefilling the prompt.

        Parameters
        ----------
        inputs: Tensor
            Input tokens, having ``seq_len`` number of tokens.

        total_seq_len: tir.Var
            Number of already-processed tokens plus ``seq_len``.

        rolling_cache_len: tir.Var
            Number of elements currently in the cache.

        kv_seq_len: tir.Var
            Equals to ``seq_len + rolling_cache_len``.
        """

        def _sliding_window_attention_mask(
            batch_size, seq_len, rolling_cache_len, kv_seq_len, sliding_window
        ):
            # See `tests/legacy-python/test_sliding_window_mask.py` for its behavior
            return te.compute(
                (batch_size, 1, seq_len, kv_seq_len),
                lambda b, _, i, j: tir.Select(
                    tir.all(i + rolling_cache_len >= j, i + rolling_cache_len - j < sliding_window),
                    tir.max_value(self.dtype),
                    tir.min_value(self.dtype),
                ),
                name="sliding_window_attention_mask_prefill",
            )

        batch_size, seq_len = inputs.shape
        attention_mask = op.tensor_expr_op(
            _sliding_window_attention_mask,
            name_hint="sliding_window_attention_mask_prefill",
            args=[
                batch_size,
                seq_len,
                rolling_cache_len,
                kv_seq_len,
                self.sliding_window,
            ],
        )
        return self.forward(inputs, total_seq_len, rolling_cache_len, kv_seq_len, attention_mask)

    def decode(
        self,
        inputs: Tensor,
        total_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
    ):
        """Decoding step."""
        batch_size, seq_len = inputs.shape
        attention_mask = op.full(
            shape=[batch_size, 1, seq_len, kv_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        return self.forward(inputs, total_seq_len, rolling_cache_len, kv_seq_len, attention_mask)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        """Softmax."""
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "total_seq_len": int,
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "total_seq_len": int,
                "rolling_cache_len": int,
                "kv_seq_len": int,
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
