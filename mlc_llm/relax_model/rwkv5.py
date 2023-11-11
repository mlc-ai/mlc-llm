# pylint: disable=missing-docstring,invalid-name
from dataclasses import dataclass
from typing import List, Literal, Tuple

from tvm import relax, te, tir, topi
from tvm.relax import Expr, op
from tvm.relax.op.nn import silu, group_norm
from tvm.relax.testing import nn
from tvm.script import relax as R
from tvm.script import tir as T

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList, Linear
from .param_manager import ParamManager

# Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model_run.py


@dataclass
class RWKV5Config:
    """The configuration class to store the configuration of a `RWKV5Model`."""

    num_hidden_layers: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    rescale_every: int = 0
    layer_norm_epsilon: float = 1e-5
    max_sequence_length: int = 1024
    dtype: str = "float32"

    def __init__(
        self,
        num_hidden_layers: int,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        head_size: int = 64,
        attention_hidden_size: int = None,
        rescale_every: int = 0,
        layer_norm_epsilon: float = 1e-5,
        context_length: int = 1024,
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_size = head_size
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.rescale_every = rescale_every
        self.layer_norm_epsilon = layer_norm_epsilon
        self.max_sequence_length = context_length
        self.dtype = dtype
        self.kwargs = kwargs


class State:
    ATT_X = 0
    ATT_KV = 1
    FFN_X = 2


def _load_state(state: Expr, hidden_size: int, num_attention_heads: int, dtype: str, kv: bool) -> Expr:
    # Reuse `attention_kv_cache_view`
    f_load_cache = relax.extern("vm.builtin.attention_kv_cache_view")
    if kv:
        cache = nn.emit(
            relax.Call(
                f_load_cache,
                [state, R.shape([1, num_attention_heads, hidden_size // num_attention_heads, hidden_size // num_attention_heads])],
                sinfo_args=[R.Tensor((1, num_attention_heads, hidden_size // num_attention_heads, hidden_size // num_attention_heads), dtype)],
            )
        )
    else:
        cache = nn.emit(
            relax.Call(
                f_load_cache,
                [state, R.shape([1, hidden_size])],
                sinfo_args=[R.Tensor((1, hidden_size), dtype)],
            )
        )
    return cache


def _store_state(state: Expr, value: Expr):
    # Reuse `attention_kv_cache_update`
    f_store_cache = relax.extern("vm.builtin.attention_kv_cache_update")

    return nn.emit(
        relax.Call(
            f_store_cache,
            [state, value],
            sinfo_args=[R.Object()],
        )
    )


def is_one(x: tir.PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def _te_concat_saved_x(saved_x: te.Tensor, x: te.Tensor):
    return te.compute(
        x.shape,
        lambda i, j: tir.if_then_else(i == 0, saved_x[0, j], x[i - 1, j]),
    )

def _te_get_last_x(x: te.Tensor):
    seq_len, hidden_size = x.shape
    return te.compute((1, hidden_size), lambda _, j: x[seq_len - 1, j])

def _te_get_receptance(x: te.Tensor, t):
    h, t, s = x.shape
    return te.compute((h, 1, s), lambda i, _, j: x[i, t, j])

def _te_get_key(x: te.Tensor, t):
    h, t, s = x.shape
    return te.compute((h, t, 1), lambda i, j, _: x[i, j, t])

def _te_get_value(x: te.Tensor, t):
    h, t, s = x.shape
    return te.compute((h, 1, s), lambda i, _, j: x[i, t, j])

# https://github.com/GiantPandaCV/mlc-llm/pull/1/files#diff-e39fd9584b9046e39f007d1c432b5c90703959d148de2e8eca29f08231c9fa57R127-R212
def create_wkv5_func(B: T.int32, T_dim: T.int32, C: T.int32, H: T.int32, dtype: str, out_dtype: str):
    @T.prim_func
    def wkv_func(
        r: T.handle,
        k: T.handle,
        v: T.handle,
        w: T.handle,
        u: T.handle,
        state: T.handle,
        out: T.handle
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        # Define buffer variables
        r_buf = T.match_buffer(r, (B, T_dim, H, C // H), dtype=dtype,)
        k_buf = T.match_buffer(k, (B, T_dim, H, C // H), dtype=dtype,)
        v_buf = T.match_buffer(v, (B, T_dim, H, C // H), dtype=dtype,)
        w_buf = T.match_buffer(w, (H, C // H), dtype=dtype,)
        u_buf = T.match_buffer(u, (H, C // H), dtype=dtype,)
        state_buf = T.match_buffer(state, (B, H, C // H, C // H), dtype=dtype,)
        out_buf = T.match_buffer(out, (B, T_dim, H, C // H), dtype=out_dtype,)

        # Initialize out_buf with zeros
        for i, j, k, l in T.grid(B, T_dim, H, C // H):
            out_buf[i, j, k, l] = 0

        # Define computation
        for b, h in T.grid(B, H):
            for t in T.serial(T_dim):
                for i, j in T.grid(C // H, C // H):
                    x = k_buf[b, t, h, j] * v_buf[b, t, h, i]
                    s = state_buf[b, h, i, j]
                    out_buf[b, t, h, i] += r_buf[b, t, h, j] * (u_buf[h, j] * x + s)
                    state_buf[b, h, i, j] = s * w_buf[h, j] + x

    return wkv_func

class RWKV_Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype=dtype, name="weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(op.reshape(x, shape=[-1]))
        return nn.emit(op.take(self.weight, x, axis=0))


class RWKV_LayerNorm(nn.Module):
    def __init__(self, intermediate_size, dtype, eps=1e-5, name_prefix=""):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            (intermediate_size,), dtype=dtype, name=f"{name_prefix}_ln_weight"
        )
        self.bias = nn.Parameter(
            (intermediate_size,), dtype=dtype, name=f"{name_prefix}_ln_bias"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(
            op.nn.layer_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )
        return x

class RWKV_GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.weight = nn.Parameter((num_channels,), dtype="float32", name="weight")
        self.bias = nn.Parameter((num_channels,), dtype="float32", name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
        if x.struct_info.dtype != "float32":
            x = nn.emit(relax.op.astype(x, "float32"))
        x = nn.emit(
            group_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                num_groups=self.num_groups,
                channel_axis=-2,
                axes=-1,
                epsilon=self.eps,
            )
        )
        return x


class RWKV_FFN(nn.Module):
    def __init__(self, config: RWKV5Config, index: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype
        self.head_size = config.head_size
        self.num_attention_heads = self.hidden_size // self.head_size
        self.index = index
        intermediate_size = (
                config.intermediate_size if config.intermediate_size is not None else int((config.hidden_size * 3.5) // 32 * 32)
            )
        self.time_mix_key = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"ffn_{index}_time_mix_k"
        )
        self.time_mix_receptance = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"ffn_{index}_time_mix_r"
        )
        self.key = Linear(
            self.hidden_size, intermediate_size, dtype=config.dtype, bias=False
        )
        self.receptance = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.value = Linear(
            intermediate_size, self.hidden_size, dtype=config.dtype, bias=False
        )

    def forward(self, x: Expr, state: Expr) -> Expr:
        offset = self.index * 3 + State.FFN_X
        context_length = x.struct_info.shape[0]
        hidden_size = self.hidden_size

        saved_x = _load_state(state[offset], hidden_size, self.num_attention_heads, self.dtype, kv=False)
        if not is_one(context_length):
            saved_x = nn.emit_te(_te_concat_saved_x, saved_x, x)
        ones = nn.emit(relax.op.ones((hidden_size,), self.dtype))
        xk = nn.emit(x * self.time_mix_key + saved_x * (ones - self.time_mix_key))
        xr = nn.emit(
            x * self.time_mix_receptance + saved_x * (ones - self.time_mix_receptance)
        )
        if not is_one(context_length):
            x = nn.emit_te(_te_get_last_x, x)
        assert is_one(x.struct_info.shape[0])
        saved_x = _store_state(state[offset], x)

        r = nn.emit(op.sigmoid(self.receptance(xr)))
        xv = nn.emit(op.square(op.nn.relu(self.key(xk))))

        return nn.emit(r * self.value(xv)), [saved_x]


class RWKV_Attention(nn.Module):
    def __init__(self, config: RWKV5Config, index: int) -> None:
        super().__init__()
        self.index = index
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.head_size = config.head_size
        self.num_attention_heads = self.hidden_size // self.head_size
        self.eps = config.layer_norm_epsilon
        self.time_decay = nn.Parameter(
            (self.num_attention_heads, self.head_size), dtype="float32", name=f"att_{index}_time_decay"
        )
        self.time_faaaa = nn.Parameter(
            (self.num_attention_heads, self.head_size), dtype="float32", name=f"att_{index}_time_faaaa"
        )
        self.time_mix_gate = nn.Parameter(
            (self.hidden_size, ), dtype=config.dtype, name=f"att_{index}_time_mix_gate"
        )
        self.time_mix_key = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"att_{index}_time_mix_k"
        )
        self.time_mix_value = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"att_{index}_time_mix_v"
        )
        self.time_mix_receptance = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"att_{index}_time_mix_r"
        )
        self.key = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.value = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.receptance = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.gate = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.output = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.ln_x = RWKV_GroupNorm(self.hidden_size // self.head_size, self.hidden_size)

    def forward(self, x: Expr, state: Expr) -> Expr:
        H = self.num_attention_heads
        S = x.struct_info.shape[-1] // H
        T = x.struct_info.shape[0]
        # Load current state
        index = self.index
        hidden_size = self.hidden_size
        C = hidden_size
        N = C // H
        ones = nn.emit(relax.op.ones((self.hidden_size,), self.dtype))
        context_length = x.struct_info.shape[0]
        bb = relax.BlockBuilder.current()

        saved_kv = _load_state(state[index * 3 + State.ATT_KV], hidden_size, self.num_attention_heads, "float32", kv=True)
        saved_x = _load_state(state[index * 3 + State.ATT_X], hidden_size, self.num_attention_heads, self.dtype, kv=False)
        if not is_one(context_length):
            saved_x = nn.emit_te(_te_concat_saved_x, saved_x, x)

        xk = nn.emit(x * self.time_mix_key + saved_x * (ones - self.time_mix_key))
        xv = nn.emit(x * self.time_mix_value + saved_x * (ones - self.time_mix_value))
        xr = nn.emit(
            x * self.time_mix_receptance + saved_x * (ones - self.time_mix_receptance)
        )
        xg = nn.emit(x * self.time_mix_gate + saved_x * (ones - self.time_mix_gate))
        g = nn.emit(silu(self.gate(xg)))

        if is_one(context_length):
            r = nn.emit(op.reshape(op.astype(self.receptance(xr), "float32"), shape=[H, 1, S]))
            k = nn.emit(op.reshape(op.astype(self.key(xk), "float32"), shape=[H, S, 1]))
            v = nn.emit(op.reshape(op.astype(self.value(xv), "float32"), shape=[H, 1, S]))
        else:
            r = nn.emit(op.reshape(op.astype(self.receptance(xr), "float32"), shape=[1, T, H, N]))
            k = nn.emit(op.reshape(op.astype(self.key(xk), "float32"), shape=[1, T, H, N]))
            v = nn.emit(op.reshape(op.astype(self.value(xv), "float32"), shape=[1, T, H, N]))

        if not is_one(context_length):
            # TODO: add rwkv5 tir here
            # out, s = self.RUN_RWKV_5(1, T, self.args.n_att, H, s.transpose(-1,-2).contiguous(), r, k, v, w=t_decay, u=t_first)
            # s = s.transpose(-1,-2)
            # s means saved_kv here
            w = nn.emit(op.reshape(self.time_decay, shape=([H, N])))
            u = nn.emit(op.reshape(self.time_faaaa, shape=([H, N])))
            gv = bb.add_func(create_wkv5_func(1, T, hidden_size, H, "float32", self.dtype), "wkv")
            ret = nn.emit(
                relax.call_tir(
                    gv,
                    [r, k, v, w, u, saved_kv],
                    [
                        R.Tensor((1, context_length, hidden_size), self.dtype),
                        R.Tensor((1, self.num_attention_heads, hidden_size // self.num_attention_heads, hidden_size // self.num_attention_heads), "float32"),
                    ],
                )
            )
            saved_kv = ret[1]
            out = nn.emit(op.reshape(ret[0], shape=([T, H*N])))
            out = nn.emit(op.squeeze(op.nn.group_norm(op.astype(op.expand_dims(out, 0), "float32"), self.ln_x.weight, self.ln_x.bias, self.ln_x.num_groups, channel_axis=-1, axes=[0, 1], epsilon=self.eps), 0))
            out = nn.emit(op.multiply(op.astype(out, self.dtype), g))
            out = nn.emit(self.output(out))
        else:
            # a = key @ value
            # out = receptance @ (time_first * a + state.squeeze(0))
            # state = a + time_decay * state
            # out = out.flatten()
            # out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lxw, bias=lxb).squeeze(0)
            # out = out.to(dtype=hidden.dtype) * gate
            # out = out @ ow\
            a = nn.emit(op.matmul(k, v))
            out = nn.emit(op.matmul(r, op.add(op.reshape(self.time_faaaa, shape=[H ,-1, 1]) * a, op.squeeze(saved_kv, 0))))
            saved_kv = nn.emit(a + op.reshape(self.time_decay, shape=[H ,-1, 1]) * saved_kv)
            out = nn.emit(op.flatten(out))
            out = nn.emit(op.squeeze(op.nn.group_norm(op.astype(op.expand_dims(out, 0), "float32"), self.ln_x.weight, self.ln_x.bias, self.ln_x.num_groups, channel_axis=-1, axes=[0], epsilon=self.eps), 0))
            out = nn.emit(op.multiply(op.astype(out, self.dtype), g))
            out = nn.emit(self.output(out))

        if not is_one(context_length):
            x = nn.emit_te(_te_get_last_x, x)

        assert is_one(x.struct_info.shape[0])
        saved_x = _store_state(state[self.index * 3 + State.ATT_X], x)
        saved_kv = _store_state(state[self.index * 3 + State.ATT_KV], saved_kv)

        return out, [
            saved_x,
            saved_kv,
        ]


class RWKVLayer(nn.Module):
    def __init__(self, config: RWKV5Config, index: int) -> None:
        super().__init__()
        if index == 0:
            self.pre_ln = RWKV_LayerNorm(
                config.hidden_size,
                config.dtype,
                eps=config.layer_norm_epsilon,
                name_prefix="pre_ln",
            )
        self.ln1 = RWKV_LayerNorm(
            config.hidden_size,
            config.dtype,
            eps=config.layer_norm_epsilon,
            name_prefix=f"att_{index}",
        )
        self.ln2 = RWKV_LayerNorm(
            config.hidden_size,
            config.dtype,
            eps=config.layer_norm_epsilon,
            name_prefix=f"ffn_{index}",
        )
        
        self.attention = RWKV_Attention(config, index)
        self.feed_forward = RWKV_FFN(config, index)
        self.rescale_every = config.rescale_every
        self.dtype = config.dtype
        self.index = index

    def forward(self, x: Expr, state: Expr) -> Tuple[Expr, List[Expr]]:
        if self.index == 0:
            x = self.pre_ln(x)
        att, att_state = self.attention(self.ln1(x), state)
        x = nn.emit(x + att)
        ffn, ffn_state = self.feed_forward(self.ln2(x), state)
        x = nn.emit(x + ffn)
        if self.rescale_every > 0 and (self.index + 1) % self.rescale_every == 0:
            x = nn.emit(x / relax.const(2, dtype=self.dtype))
        return x, att_state + ffn_state


class RWKVModel(nn.Module):
    def __init__(self, config: RWKV5Config) -> None:
        super().__init__()
        self.embeddings = RWKV_Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        self.blocks = ModuleList(
            [RWKVLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.ln_out = RWKV_LayerNorm(
            config.hidden_size,
            config.dtype,
            eps=config.layer_norm_epsilon,
            name_prefix="out_ln",
        )
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype

    def forward(self, input_ids: Expr, state: Expr) -> Tuple[Expr, List[Expr]]:
        hidden_states = self.embeddings(input_ids)
        states = []
        for _, layer in enumerate(self.blocks):
            hidden_states, layer_states = layer(hidden_states, state)
            states += layer_states
        context_length = hidden_states.struct_info.shape[0]
        if not is_one(context_length):
            hidden_states = nn.emit_te(_te_get_last_x, hidden_states)
        hidden_states = self.ln_out(hidden_states)
        return hidden_states, states


class RWKV5ForCausalLM(nn.Module):
    def __init__(self, config: RWKV5Config):
        self.rwkv = RWKVModel(config)
        self.head = Linear(
            config.hidden_size, config.vocab_size, dtype=config.dtype, bias=False
        )
        self.vocab_size = config.vocab_size
        ############ End ############

    def forward(
        self,
        input_ids: relax.Expr,
        state: relax.Expr,
    ):
        hidden_states, key_value_cache = self.rwkv(input_ids, state)
        logits = nn.emit(self.head(hidden_states))
        logits = nn.emit(op.reshape(logits, (1, 1, self.vocab_size)))
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, key_value_cache


def get_param_quant_kind(
    name: str, param_info: relax.TensorStructInfo
) -> ParamQuantKind:
    if name.endswith("embeddings.weight"):
        return ParamQuantKind.embedding_table
    elif name == "head.weight":
        return ParamQuantKind.final_fc_weight
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: RWKV5Config,
    quant_scheme: QuantizationScheme,
    func_name=Literal["prefill", "decode"],
):
    if func_name not in ["prefill", "decode"]:
        raise ValueError(f"func_name must be 'prefill' or 'decode', got {func_name}")
    seq_len = 1 if func_name == "decode" else tir.Var("n", "int64")

    with bb.function(func_name):
        model = RWKV5ForCausalLM(config)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        input_ids = nn.Placeholder((1, seq_len), dtype="int32", name="input_ids")
        # Placeholder for compatibility to RWKV
        all_seq_len_shape = relax.Var("place_holder", R.Object())
        state = relax.Var("state", R.Tuple([R.Object()] * config.num_hidden_layers * 3))
        with bb.dataflow():
            logits, states = model(input_ids, state)
            params = [
                input_ids,
                all_seq_len_shape,
                state,
            ] + model.parameters()

            gv = bb.emit_output((logits, relax.Tuple(states)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    f = mod[gv].with_attr("num_input", 3)
    if func_name == "prefill":
        f = f.with_attr("tir_var_upper_bound", {"n": config.max_sequence_length})
    bb.update_func(gv, f)


def create_kv_cache_func(bb: relax.BlockBuilder, config: RWKV5Config) -> None:
    """NOTE: It's not typical kv-cache, but try to reuse the logic for the quick hack."""
    init_shape = relax.ShapeExpr((1, config.hidden_size))
    num_attention_heads = config.hidden_size // config.head_size
    init_kv_shape = relax.ShapeExpr((1, num_attention_heads, config.hidden_size // num_attention_heads, config.hidden_size // num_attention_heads))
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            input_dtype_zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            fp32_zeros = bb.emit(relax.op.zeros(init_kv_shape, "float32"))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            conf = [
                ("att_x", input_dtype_zeros),
                ("att_kv", fp32_zeros),
                ("ffn_x", input_dtype_zeros),
            ]
            for i in range(config.num_hidden_layers):
                for name, init_value in conf:
                    if name == "att_kv":
                        caches.append(
                            bb.emit(
                                relax.Call(
                                    f_kv_cache_create,
                                    [init_value, init_kv_shape, relax.PrimValue(1)],
                                    sinfo_args=[R.Object()],
                                ),
                                name_hint=f"{name}_state_{i}",
                            )
                        )
                    else:
                        caches.append(
                            bb.emit(
                                relax.Call(
                                    f_kv_cache_create,
                                    [init_value, init_shape, relax.PrimValue(1)],
                                    sinfo_args=[R.Object()],
                                ),
                                name_hint=f"{name}_state_{i}",
                            )
                        )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_kv_cache_reset_func(bb: relax.BlockBuilder, config: RWKV5Config) -> None:
    state = relax.Var("state", R.Tuple([R.Object()] * config.num_hidden_layers * 3))
    init_shape = relax.ShapeExpr((1, config.hidden_size))
    num_attention_heads = config.hidden_size // config.head_size
    init_kv_shape = relax.ShapeExpr((1, num_attention_heads, config.hidden_size // num_attention_heads, config.hidden_size // num_attention_heads))
    with bb.function("reset_kv_cache", [state]):
        with bb.dataflow():
            input_dtype_zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            fp32_zeros = bb.emit(relax.op.zeros(init_kv_shape, "float32"))
            caches = []
            for i in range(config.num_hidden_layers):
                caches.append(
                    _store_state(state[i * 3 + State.ATT_X], input_dtype_zeros)
                )
                caches.append(_store_state(state[i * 3 + State.ATT_KV], fp32_zeros))
                caches.append(
                    _store_state(state[i * 3 + State.FFN_X], input_dtype_zeros)
                )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_softmax_func(bb: relax.BlockBuilder, config: RWKV5Config) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder(
            (1, 1, config.vocab_size), dtype="float32", name="logits"
        )
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def get_model(args, hf_config):
    model_name = args.model
    max_seq_len = args.max_seq_len
    dtype = args.quantization.model_dtype

    if not model_name.lower().startswith("rwkv-"):
        raise ValueError(f"Unsupported model name: {model_name}")

    config = RWKV5Config(**hf_config, dtype=dtype)
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    create_func(bb, param_manager, config, args.quantization, "prefill")
    create_func(bb, param_manager, config, args.quantization, "decode")
    create_kv_cache_func(bb, config)
    create_softmax_func(bb, config)
    create_metadata_func(
        bb,
        model_name=model_name,
        # RNN model do not have window size limit
        max_window_size=-1,
        stop_tokens=[0],
        add_prefix_space=False,
    )
    create_kv_cache_reset_func(bb, config)
    mod = bb.get()

    if args.build_model_only:
        return mod, param_manager, None, config

    def f_convert_pname_fwd(pname: str) -> List[str]:
        if (
            "key_weight" in pname
            or "value_weight" in pname
            or "receptance_weight" in pname
            or "output_weight" in pname
            or "head_weight" in pname
        ):
            return [pname.replace("_weight", ".weight")]
        else:
            return [pname]

    def f_convert_param_bkwd(torch_pname: str, torch_param):
        print(torch_pname, torch_param.shape, torch_param.dtype)
        # torch_param: numpy.ndarray
        import numpy as np  # pylint: disable=import-outside-toplevel

        # rescale_every
        if config.rescale_every > 0 and "blocks." in torch_pname:
            # based-on the assumption that the layer id is the second element in torch_pname
            layer_id = int(torch_pname.split(".")[2])
            if (
                "attention.output.weight" in torch_pname
                or "feed_forward.value.weight" in torch_pname
            ):
                torch_param = torch_param / (2 ** (layer_id // config.rescale_every))

        # reshape
        if "time_mix_" in torch_pname:
            torch_param = torch_param.squeeze().squeeze()
        
        if "ln_x" in torch_pname:
            return [(torch_pname, torch_param.astype("float32"))]

        # convert dtype
        # https://github.com/BBuf/RWKV-World-HF-Tokenizer/blob/main/rwkv_world_v5_model/modeling_rwkv5.py#L88
        # time_decay = torch.exp(-torch.exp(time_decay.float())).reshape(-1,1,1).reshape(n_head, -1, 1)
        # time_first = time_first.float().reshape(-1,1,1).reshape(n_head, -1, 1)
        if "time_decay" in torch_pname:  # need fp32 for this
            return [(torch_pname, np.exp(-np.exp(torch_param.astype("float32"))))]
        elif "time_faaaa" in torch_pname:
            return [(torch_pname, torch_param.astype("float32"))]
        else:
            return [(torch_pname, torch_param.astype(config.dtype))]

    param_manager.set_param_loading_func(
        args.model_path, args.use_safetensors, f_convert_pname_fwd, f_convert_param_bkwd
    )
    return mod, param_manager, [None] * len(param_manager.param_names), config
