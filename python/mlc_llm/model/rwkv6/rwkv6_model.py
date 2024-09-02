"""Implementation for RWKV6 architecture."""

import dataclasses
from typing import Any, Dict, Optional, Tuple

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Object, Tensor, op
from tvm.script import tir as T

from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StateID:
    """State ID for RWKV6."""

    ATT_X = 0
    ATT_KV = 1
    FFN_X = 2


@dataclasses.dataclass
class RWKV6Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the RWKV6 model."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    model_version: str = "6_0"
    tensor_parallel_shards: int = 1
    rescale_every: int = 0
    head_size: int = 64
    layer_norm_epsilon: float = 1e-5
    context_window_size: int = -1  # RWKV does not have context window limitation.
    prefill_chunk_size: int = 4096
    num_heads: int = 0
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.model_version != "6_0":
            raise ValueError(f"Only support RWKV v6_0, got {self.model_version}.")
        self.intermediate_size = self.intermediate_size or int((self.hidden_size * 3.5)) // 32 * 32
        self.num_heads = (
            self.hidden_size // self.head_size if self.num_heads == 0 else self.num_heads
        )
        if self.num_heads * self.head_size != self.hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible "
                f"by head_size ({self.head_size})"
            )
        if self.tensor_parallel_shards != 1:
            raise ValueError("Only support single device at this moment.")


# pylint: disable=invalid-name, missing-docstring
# pylint: disable=too-many-arguments, too-many-locals, redefined-argument-from-local
def create_wkv6_func(
    num_heads: int,
    head_size: int,
    dtype: str,
    out_dtype: str,
    state_dtype: str,
):
    @T.prim_func
    def wkv_func(
        r: T.handle,
        k: T.handle,
        v: T.handle,
        time_faaaa: T.handle,
        w: T.handle,
        state: T.handle,
        out: T.handle,
        out_state: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        # Inputs
        r_buf = T.match_buffer(r, (batch_size, seq_len, num_heads, head_size), dtype=dtype)
        k_buf = T.match_buffer(k, (batch_size, seq_len, num_heads, head_size), dtype=dtype)
        v_buf = T.match_buffer(v, (batch_size, seq_len, num_heads, head_size), dtype=dtype)
        time_faaaa_buf = T.match_buffer(time_faaaa, (num_heads, head_size), dtype="float32")
        w_buf = T.match_buffer(w, (batch_size, seq_len, num_heads, head_size), dtype="float32")
        state_buf = T.match_buffer(
            state, (batch_size, num_heads, head_size, head_size), dtype=state_dtype
        )
        # Outputs
        out_buf = T.match_buffer(out, (batch_size, seq_len, num_heads, head_size), dtype=out_dtype)
        out_state_buf = T.match_buffer(
            out_state, (batch_size, num_heads, head_size, head_size), dtype=state_dtype
        )
        for b in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h in T.thread_binding(num_heads, thread="blockIdx.x"):
                for i in T.thread_binding(head_size, thread="threadIdx.x"):
                    for j in range(head_size):
                        with T.block("init_state"):
                            vb, vh, vi, vj = T.axis.remap("SSSS", [b, h, i, j])
                            out_state_buf[vb, vh, vi, vj] = state_buf[vb, vh, vi, vj]

                    for t in range(seq_len):
                        with T.block("comput"):
                            vb = T.axis.spatial(batch_size, b)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_heads, h)
                            vi = T.axis.spatial(head_size, i)
                            out_buf[vb, vt, vh, vi] = 0

                            for k in range(head_size):
                                at = k_buf[vb, vt, vh, k] * v_buf[vb, vt, vh, vi]
                                out_buf[vb, vt, vh, vi] += T.cast(
                                    r_buf[vb, vt, vh, k], out_dtype
                                ) * T.cast(
                                    time_faaaa_buf[vh, k] * at + out_state_buf[vb, vh, vi, k],
                                    out_dtype,
                                )
                                out_state_buf[vb, vh, vi, k] = (
                                    at + w_buf[vb, vt, vh, k] * out_state_buf[vb, vh, vi, k]
                                )

    return wkv_func


def token_shift(state: Tensor, x: Tensor):
    def _te_token_shift(state: te.Tensor, x: te.Tensor):
        return te.compute(
            x.shape,
            lambda b, i, j: tir.if_then_else(i == 0, state[b, j], x[b, i - 1, j]),
        )

    return op.tensor_expr_op(_te_token_shift, "token_shift", [state, x])


def last_token(x: Tensor):
    batch, seq_len, hidden_size = x.shape

    def _te_last_token(x: te.Tensor):
        return te.compute((batch, 1, hidden_size), lambda b, _, j: x[b, x.shape[1] - 1, j])

    return x if seq_len == 1 else op.tensor_expr_op(_te_last_token, "last_token", [x])


def unbind_to_five(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert x.shape[0] == 5

    def _te_get_ith(x: te.Tensor, i: int):
        return te.compute((1, *x.shape[1:]), lambda _, j, k, l: x[i, j, k, l])

    return (
        op.reshape(op.tensor_expr_op(_te_get_ith, "unbind_to_five", [x, 0]), x.shape[1:]),
        op.reshape(op.tensor_expr_op(_te_get_ith, "unbind_to_five", [x, 1]), x.shape[1:]),
        op.reshape(op.tensor_expr_op(_te_get_ith, "unbind_to_five", [x, 2]), x.shape[1:]),
        op.reshape(op.tensor_expr_op(_te_get_ith, "unbind_to_five", [x, 3]), x.shape[1:]),
        op.reshape(op.tensor_expr_op(_te_get_ith, "unbind_to_five", [x, 4]), x.shape[1:]),
    )


class RWKV6_FNN(nn.Module):
    def __init__(self, config: RWKV6Config, layer_id: int):
        super().__init__()
        self.time_maa_k = nn.Parameter((1, 1, config.hidden_size))
        self.time_maa_r = nn.Parameter((1, 1, config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.hidden_size // 2 * 7, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size // 2 * 7, config.hidden_size, bias=False)
        self.layer_id = layer_id

    def forward(self, x: Tensor, state: RNNState):
        batch, _, hidden_size = x.shape
        state_x = state.get(self.layer_id, StateID.FFN_X, (batch, hidden_size), x.dtype)
        state_x = token_shift(state_x, x)

        state_x = state_x - x
        xk = x + state_x * self.time_maa_k
        xr = x + state_x * self.time_maa_r

        last_x = last_token(x).reshape(batch, hidden_size)
        state = state.set(self.layer_id, StateID.FFN_X, last_x)

        r = op.sigmoid(self.receptance(xr))
        xv = op.square(op.relu(self.key(xk)))
        return r * self.value(xv), state


class RWKV6_Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Attention layer for RWKV."""

    def __init__(self, config: RWKV6Config, layer_id: int):
        super().__init__()
        self.time_maa_x = nn.Parameter((1, 1, config.hidden_size))
        self.time_maa_w = nn.Parameter((1, 1, config.hidden_size))
        self.time_maa_k = nn.Parameter((1, 1, config.hidden_size))
        self.time_maa_v = nn.Parameter((1, 1, config.hidden_size))
        self.time_maa_r = nn.Parameter((1, 1, config.hidden_size))
        self.time_maa_g = nn.Parameter((1, 1, config.hidden_size))

        # RWKV v6 7B/14B
        if config.hidden_size == 4096:
            time_mix_extra_dim = 64
            time_decay_extra_dim = 128
        else:
            time_mix_extra_dim = 32
            time_decay_extra_dim = 64

        self.time_maa_w1 = nn.Parameter((config.hidden_size, 5 * time_mix_extra_dim))
        self.time_maa_w2 = nn.Parameter((5, time_mix_extra_dim, config.hidden_size))
        self.time_decay_w1 = nn.Parameter((config.hidden_size, time_decay_extra_dim))
        self.time_decay_w2 = nn.Parameter((time_decay_extra_dim, config.hidden_size))
        self.time_decay = nn.Parameter((1, 1, config.hidden_size))
        self.time_faaaa = nn.Parameter((config.num_heads, config.head_size))

        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(config.num_heads, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.layer_id = layer_id
        self.dtype = "float32"

    def forward(self, x: Tensor, state: RNNState):  # pylint: disable=too-many-locals
        batch, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size
        B, T, H, N = (  # pylint: disable=redefined-outer-name
            batch,
            seq_len,
            self.head_size,
            self.num_heads,
        )
        state_x = state.get(self.layer_id, StateID.ATT_X, (batch, self.hidden_size), x.dtype)
        state_x = token_shift(state_x, x)
        state_x = state_x - x
        xxx = x + state_x * self.time_maa_x
        xxx = op.permute(
            op.reshape(op.tanh(op.matmul(xxx, self.time_maa_w1)), (B, T, 5, -1)), [0, 2, 1, 3]
        )
        xxx = op.permute(
            op.matmul(xxx, self.time_maa_w2), axes=[1, 0, 2, 3]
        )  # it's a batch matrix-matrix multiplication
        mw, mk, mv, mr, mg = unbind_to_five(xxx)

        kv_state = state.get(
            self.layer_id,
            StateID.ATT_KV,
            (batch, self.num_heads, self.head_size, self.head_size),
            "float32",
        )

        xw = x + state_x * (self.time_maa_w + mw)
        xk = x + state_x * (self.time_maa_k + mk)
        xv = x + state_x * (self.time_maa_v + mv)
        xr = x + state_x * (self.time_maa_r + mr)
        xg = x + state_x * (self.time_maa_g + mg)

        r = op.reshape(self.receptance(xr), (B, T, N, H))
        k = op.reshape(self.key(xk), (B, T, N, H))
        v = op.reshape(self.value(xv), (B, T, N, H))
        g = op.silu(self.gate(xg))

        w = op.reshape(self.time_decay, (1, N, H)).astype("float32") + op.reshape(
            op.matmul(op.tanh(op.matmul(xw, self.time_decay_w1)), self.time_decay_w2),
            (B, T, N, H),
        ).astype("float32")
        w = op.exp(op.negative(op.exp(w)))
        # w = op.reshape(w, [B, T, N, H])

        out, kv_state = op.tensor_ir_op(
            create_wkv6_func(
                num_heads=self.num_heads,
                head_size=self.head_size,
                dtype=self.dtype,
                out_dtype="float32",
                state_dtype="float32",
            ),
            "wkv6",
            [r, k, v, self.time_faaaa, w, kv_state],
            [
                Tensor.placeholder([B, T, N, H], "float32"),
                Tensor.placeholder([B, N, H, H], "float32"),
            ],
        )

        last_x = last_token(x).reshape(batch, hidden_size)
        state = state.set(self.layer_id, StateID.ATT_X, last_x)
        state = state.set(self.layer_id, StateID.ATT_KV, kv_state)
        out = op.astype(self.ln_x(op.reshape(out, x.shape), channel_axis=-1, axes=[]), self.dtype)
        return self.output(out * g), state

    def to(self, dtype: Optional[str] = None):
        # RWKV uses special dtype, so we need to convert it.
        if dtype is not None:
            self.dtype = dtype

        self.time_maa_x.to(dtype)
        self.time_maa_w.to(dtype)
        self.time_maa_k.to(dtype)
        self.time_maa_v.to(dtype)
        self.time_maa_r.to(dtype)
        self.time_maa_g.to(dtype)
        self.time_maa_w1.to(dtype)
        self.time_maa_w2.to(dtype)
        self.time_decay_w1.to(dtype)
        self.time_decay_w2.to(dtype)
        self.key.to(dtype)
        self.value.to(dtype)
        self.receptance.to(dtype)
        self.gate.to(dtype)
        self.output.to(dtype)

        # These parameters are necessary to be converted to float32.
        self.time_decay.to("float32")
        self.time_faaaa.to("float32")
        self.ln_x.to("float32")


class RWKV6_Layer(nn.Module):
    def __init__(self, config: RWKV6Config, layer_id: int):
        super().__init__()
        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon,
            )
        self.ln1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.ln2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attention = RWKV6_Attention(config, layer_id)
        self.feed_forward = RWKV6_FNN(config, layer_id)
        self.layer_id = layer_id
        self.rescale_every = config.rescale_every

    def forward(self, x: Tensor, state: RNNState) -> Tensor:
        if self.layer_id == 0:
            x = self.pre_ln(x)
        att_x, state = self.attention(self.ln1(x), state)
        x += att_x
        ffn_x, state = self.feed_forward(self.ln2(x), state)
        x += ffn_x
        if self.rescale_every > 0 and (self.layer_id + 1) % self.rescale_every == 0:
            x = x / 2.0
        return x, state


class RWKV6_Model(nn.Module):
    """Exact same as LlamaModel."""

    def __init__(self, config: RWKV6Config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [RWKV6_Layer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.ln_out = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(self, input_embed: Tensor, state: RNNState):
        """Forward pass of the model, passing through all decoder layers."""
        hidden_states = input_embed
        for block in self.blocks:
            hidden_states, state = block(hidden_states, state)
        return self.ln_out(hidden_states), state


class RWKV6_ForCasualLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Same as LlamaForCausalLM, except for the use of sliding window attention."""

    def __init__(self, config: RWKV6Config):
        self.model = RWKV6_Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        return self.model.embeddings(input_ids)

    def forward(
        self, input_embed: Tensor, state: RNNState, logit_positions: Optional[Tensor] = None
    ):
        """Forward pass."""
        hidden_states, state = self.model(input_embed, state)
        hidden_states = last_token(hidden_states)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, state

    def prefill(self, input_embed: Tensor, state: RNNState):
        """Prefilling the prompt."""
        return self.forward(input_embed, state)

    def decode(self, input_embed: Tensor, state: RNNState):
        """Decoding step."""
        return self.forward(input_embed, state)

    def batch_prefill(self, input_embeds: Tensor, logit_positions: Tensor, state: RNNState):
        """Prefilling the prompt."""
        return self.forward(input_embeds, state, logit_positions=logit_positions)

    def batch_decode(self, input_embeds: Tensor, state: RNNState):
        """Decoding step."""
        return self.forward(input_embeds, state)

    def batch_verify(self, input_embeds: Tensor, state: RNNState):
        """Verify step."""
        return self.forward(input_embeds, state)

    def create_rnn_state(
        self,
        max_batch_size: tir.Var,
        max_history: tir.Var,
    ) -> Object:
        """Create RNN state."""
        init_values = [
            op.zeros((self.hidden_size,), dtype=self.dtype),  # ATT_X
            op.zeros((self.num_heads, self.head_size, self.head_size), dtype="float32"),  # ATT_KV
            op.zeros((self.hidden_size,), dtype=self.dtype),  # FFN_X
        ]
        return RNNState.create(
            max_batch_size=max_batch_size,
            num_hidden_layers=self.num_hidden_layers,
            max_history=max_history,
            init_values=init_values,
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
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
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
