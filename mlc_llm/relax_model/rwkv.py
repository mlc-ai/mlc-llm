# pylint: disable=missing-docstring,invalid-name
from dataclasses import dataclass
from typing import List, Literal, Tuple

from tvm import relax, te, tir
from tvm.relax import Expr, op
from tvm.relax.testing import nn
from tvm.script import relax as R
from tvm.script import tir as T

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList, Linear
from .param_manager import ParamManager

# Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model_run.py


@dataclass
class RWKVConfig:
    """The configuration class to store the configuration of a `RWKVModel`."""

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
        self.rescale_every = rescale_every
        self.layer_norm_epsilon = layer_norm_epsilon
        self.max_sequence_length = context_length
        self.dtype = dtype
        self.kwargs = kwargs


class State:
    ATT_X = 0
    ATT_A = 1
    ATT_B = 2
    ATT_P = 3
    FFN_X = 4


def _load_state(state: Expr, hidden_size: int, dtype: str) -> Expr:
    # Reuse `attention_kv_cache_view`
    f_load_cache = relax.extern("vm.builtin.attention_kv_cache_view")
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


def create_wkv_func(hidden_size: int, dtype: str, out_dtype: str):
    @T.prim_func
    def wkv_func(
        k: T.handle,
        v: T.handle,
        time_decay: T.handle,
        time_first: T.handle,
        saved_a: T.handle,
        saved_b: T.handle,
        saved_p: T.handle,
        wkv: T.handle,
        out_a: T.handle,
        out_b: T.handle,
        out_p: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        context_length = T.int64()
        K = T.match_buffer(k, (context_length, hidden_size), dtype=dtype)
        V = T.match_buffer(v, (context_length, hidden_size), dtype=dtype)
        TimeDecay = T.match_buffer(time_decay, (hidden_size,), dtype=dtype)
        TimeFirst = T.match_buffer(time_first, (hidden_size,), dtype=dtype)
        SavedA = T.match_buffer(saved_a, (1, hidden_size), dtype=dtype)
        SavedB = T.match_buffer(saved_b, (1, hidden_size), dtype=dtype)
        SavedP = T.match_buffer(saved_p, (1, hidden_size), dtype=dtype)
        WKV = T.match_buffer(wkv, (context_length, hidden_size), dtype=out_dtype)
        OutA = T.match_buffer(out_a, (1, hidden_size), dtype=dtype)
        OutB = T.match_buffer(out_b, (1, hidden_size), dtype=dtype)
        OutP = T.match_buffer(out_p, (1, hidden_size), dtype=dtype)

        P = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        E1 = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        E2 = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        A_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        B_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        P_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")

        for bx in T.thread_binding(hidden_size // 32, thread="blockIdx.x"):
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("init"):
                    vi = T.axis.S(hidden_size, bx * 32 + tx)
                    A_local[vi] = SavedA[0, vi]
                    B_local[vi] = SavedB[0, vi]
                    P_local[vi] = SavedP[0, vi]
                for j in range(context_length):
                    with T.block("main"):
                        vi = T.axis.S(hidden_size, bx * 32 + tx)
                        vj = T.axis.opaque(context_length, j)
                        P[vi] = T.max(P_local[vi], K[vj, vi] + TimeFirst[vi])
                        E1[vi] = T.exp(P_local[vi] - P[vi])
                        E2[vi] = T.exp(K[vj, vi] + TimeFirst[vi] - P[vi])
                        WKV[vj, vi] = T.cast(
                            (E1[vi] * A_local[vi] + E2[vi] * V[vj, vi])
                            / (E1[vi] * B_local[vi] + E2[vi]),
                            out_dtype,
                        )

                        P[vi] = T.max(P_local[vi] + TimeDecay[vi], K[vj, vi])
                        E1[vi] = T.exp(P_local[vi] + TimeDecay[vi] - P[vi])
                        E2[vi] = T.exp(K[vj, vi] - P[vi])
                        A_local[vi] = E1[vi] * A_local[vi] + E2[vi] * V[vj, vi]
                        B_local[vi] = E1[vi] * B_local[vi] + E2[vi]
                        P_local[vi] = P[vi]

                with T.block("write_back"):
                    vi = T.axis.S(hidden_size, bx * 32 + tx)
                    OutA[0, vi] = A_local[vi]
                    OutB[0, vi] = B_local[vi]
                    OutP[0, vi] = P_local[vi]

    return wkv_func


def _te_concat_saved_x(saved_x: te.Tensor, x: te.Tensor):
    return te.compute(
        x.shape,
        lambda i, j: tir.if_then_else(i == 0, saved_x[0, j], x[i - 1, j]),
    )


def _te_get_last_x(x: te.Tensor):
    seq_len, hidden_size = x.shape
    return te.compute((1, hidden_size), lambda _, j: x[seq_len - 1, j])


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


class RWKV_FFN(nn.Module):
    def __init__(self, config: RWKVConfig, index: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype
        self.index = index
        self.time_mix_key = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"ffn_{index}_time_mix_k"
        )
        self.time_mix_receptance = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"ffn_{index}_time_mix_r"
        )
        self.key = Linear(
            self.hidden_size, config.intermediate_size, dtype=config.dtype, bias=False
        )
        self.receptance = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.value = Linear(
            config.intermediate_size, self.hidden_size, dtype=config.dtype, bias=False
        )

    def forward(self, x: Expr, state: Expr) -> Expr:
        offset = self.index * 5 + State.FFN_X
        context_length = x.struct_info.shape[0]
        hidden_size = self.hidden_size

        saved_x = _load_state(state[offset], hidden_size, self.dtype)
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
    def __init__(self, config: RWKVConfig, index: int) -> None:
        super().__init__()
        self.index = index
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.time_decay = nn.Parameter(
            (self.hidden_size,), dtype="float32", name=f"att_{index}_time_decay"
        )
        self.time_first = nn.Parameter(
            (self.hidden_size,), dtype="float32", name=f"att_{index}_time_first"
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
        self.output = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )

    def forward(self, x: Expr, state: Expr) -> Expr:
        # Load current state
        ones = nn.emit(relax.op.ones((self.hidden_size,), self.dtype))
        index = self.index
        hidden_size = self.hidden_size
        context_length = x.struct_info.shape[0]
        bb = relax.BlockBuilder.current()

        saved_a = _load_state(state[index * 5 + State.ATT_A], hidden_size, "float32")
        saved_b = _load_state(state[index * 5 + State.ATT_B], hidden_size, "float32")
        saved_p = _load_state(state[index * 5 + State.ATT_P], hidden_size, "float32")
        saved_x = _load_state(state[index * 5 + State.ATT_X], hidden_size, self.dtype)
        if not is_one(context_length):
            saved_x = nn.emit_te(_te_concat_saved_x, saved_x, x)

        xk = nn.emit(x * self.time_mix_key + saved_x * (ones - self.time_mix_key))
        xv = nn.emit(x * self.time_mix_value + saved_x * (ones - self.time_mix_value))
        xr = nn.emit(
            x * self.time_mix_receptance + saved_x * (ones - self.time_mix_receptance)
        )

        r = nn.emit(op.sigmoid(self.receptance(xr)))
        k = nn.emit(op.astype(self.key(xk), "float32"))
        v = nn.emit(op.astype(self.value(xv), "float32"))

        gv = bb.add_func(create_wkv_func(hidden_size, "float32", self.dtype), "wkv")
        ret = nn.emit(
            relax.call_tir(
                gv,
                [k, v, self.time_decay, self.time_first, saved_a, saved_b, saved_p],
                [
                    R.Tensor((context_length, hidden_size), self.dtype),
                    R.Tensor((1, hidden_size), "float32"),
                    R.Tensor((1, hidden_size), "float32"),
                    R.Tensor((1, hidden_size), "float32"),
                ],
            )
        )
        if not is_one(context_length):
            x = nn.emit_te(_te_get_last_x, x)

        assert is_one(x.struct_info.shape[0])
        saved_x = _store_state(state[self.index * 5 + State.ATT_X], x)
        saved_a = _store_state(state[self.index * 5 + State.ATT_A], ret[1])
        saved_b = _store_state(state[self.index * 5 + State.ATT_B], ret[2])
        saved_p = _store_state(state[self.index * 5 + State.ATT_P], ret[3])

        return nn.emit(self.output(r * ret[0])), [
            saved_x,
            saved_a,
            saved_b,
            saved_p,
        ]


class RWKVLayer(nn.Module):
    def __init__(self, config: RWKVConfig, index: int) -> None:
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
    def __init__(self, config: RWKVConfig) -> None:
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


class RWKVForCausalLM(nn.Module):
    def __init__(self, config: RWKVConfig):
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
    config: RWKVConfig,
    quant_scheme: QuantizationScheme,
    func_name=Literal["prefill", "decode"],
):
    if func_name not in ["prefill", "decode"]:
        raise ValueError(f"func_name must be 'prefill' or 'decode', got {func_name}")
    seq_len = 1 if func_name == "decode" else tir.SizeVar("n", "int64")

    with bb.function(func_name):
        model = RWKVForCausalLM(config)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        input_ids = nn.Placeholder((1, seq_len), dtype="int32", name="input_ids")
        # Placeholder for compatibility to LLAMA
        all_seq_len_shape = relax.Var("place_holder", R.Object())
        state = relax.Var("state", R.Tuple([R.Object()] * config.num_hidden_layers * 5))
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


def create_kv_cache_func(bb: relax.BlockBuilder, config: RWKVConfig) -> None:
    """NOTE: It's not typical kv-cache, but try to reuse the logic for the quick hack."""
    init_shape = relax.ShapeExpr((1, config.hidden_size))
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            input_dtype_zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            fp32_zeros = bb.emit(relax.op.zeros(init_shape, "float32"))
            fp32_neg_inf = bb.emit(fp32_zeros - relax.const(1e30, "float32"))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            conf = [
                ("att_x", input_dtype_zeros),
                ("att_a", fp32_zeros),
                ("att_b", fp32_zeros),
                ("att_p", fp32_neg_inf),
                ("ffn_x", input_dtype_zeros),
            ]
            for i in range(config.num_hidden_layers):
                for name, init_value in conf:
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


def create_kv_cache_reset_func(bb: relax.BlockBuilder, config: RWKVConfig) -> None:
    state = relax.Var("state", R.Tuple([R.Object()] * config.num_hidden_layers * 5))
    init_shape = relax.ShapeExpr((1, config.hidden_size))
    with bb.function("reset_kv_cache", [state]):
        with bb.dataflow():
            input_dtype_zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            fp32_zeros = bb.emit(relax.op.zeros(init_shape, "float32"))
            fp32_neg_inf = bb.emit(fp32_zeros - relax.const(1e30, "float32"))
            caches = []
            for i in range(config.num_hidden_layers):
                caches.append(
                    _store_state(state[i * 5 + State.ATT_X], input_dtype_zeros)
                )
                caches.append(_store_state(state[i * 5 + State.ATT_B], fp32_zeros))
                caches.append(_store_state(state[i * 5 + State.ATT_A], fp32_zeros))
                caches.append(_store_state(state[i * 5 + State.ATT_P], fp32_neg_inf))
                caches.append(
                    _store_state(state[i * 5 + State.FFN_X], input_dtype_zeros)
                )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_softmax_func(bb: relax.BlockBuilder, config: RWKVConfig) -> None:
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

    config = RWKVConfig(**hf_config, dtype=dtype)
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
        if "time_" in torch_pname:
            torch_param = torch_param.squeeze()

        # convert dtype
        if "time_decay" in torch_pname:  # need fp32 for this
            return [(torch_pname, -np.exp(torch_param.astype("float32")))]
        elif "time_first" in torch_pname:
            return [(torch_pname, torch_param.astype("float32"))]
        else:
            return [(torch_pname, torch_param.astype(config.dtype))]

    param_manager.set_param_loading_func(
        args.model_path, args.use_safetensors, f_convert_pname_fwd, f_convert_param_bkwd
    )
    return mod, param_manager, [None] * len(param_manager.param_names), config
