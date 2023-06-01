# pylint: disable=missing-docstring,invalid-name
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tvm import relax
from tvm.relax import Expr, op
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..utils import load_torch_pname2binname_map
from .commons import create_metadata_func
from .modules import ModuleList, named_parameters

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
    context_length: int = 1024
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
        self.context_length = context_length
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
        self.key_weight = nn.Parameter(
            (self.hidden_size, config.intermediate_size),
            dtype=config.dtype,
            name=f"ffn_{index}_key_weight",
        )
        self.receptance_weight = nn.Parameter(
            (self.hidden_size, self.hidden_size),
            dtype=config.dtype,
            name=f"ffn_{index}_receptance_weight",
        )
        self.value_weight = nn.Parameter(
            (config.intermediate_size, self.hidden_size),
            dtype=config.dtype,
            name=f"ffn_{index}_value_weight",
        )

    def forward(self, x: Expr, state: Expr) -> Expr:
        offset = self.index * 5 + State.FFN_X

        saved_x = _load_state(state[offset], self.hidden_size, self.dtype)
        ones = nn.emit(relax.op.ones((self.hidden_size,), self.dtype))
        xk = nn.emit(x * self.time_mix_key + saved_x * (ones - self.time_mix_key))
        xr = nn.emit(
            x * self.time_mix_receptance + saved_x * (ones - self.time_mix_receptance)
        )
        saved_x = _store_state(state[offset], x)

        r = nn.emit(op.sigmoid(op.matmul(xr, self.receptance_weight)))
        xv = nn.emit(op.square(op.nn.relu(op.matmul(xk, self.key_weight))))

        return nn.emit(r * op.matmul(xv, self.value_weight)), [saved_x]


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
        self.key_weight = nn.Parameter(
            (self.hidden_size, self.hidden_size),
            dtype=config.dtype,
            name=f"att_{index}_key_weight",
        )
        self.value_weight = nn.Parameter(
            (self.hidden_size, self.hidden_size),
            dtype=config.dtype,
            name=f"att_{index}_value_weight",
        )
        self.receptance_weight = nn.Parameter(
            (self.hidden_size, self.hidden_size),
            dtype=config.dtype,
            name=f"att_{index}_receptance_weight",
        )
        self.output_weight = nn.Parameter(
            (self.hidden_size, self.hidden_size),
            dtype=config.dtype,
            name=f"att_{index}_output_weight",
        )

    def forward(self, x: Expr, state: Expr) -> Expr:
        # Load current state
        saved_x = _load_state(
            state[self.index * 5 + State.ATT_X], self.hidden_size, self.dtype
        )
        saved_a = _load_state(
            state[self.index * 5 + State.ATT_A], self.hidden_size, "float32"
        )
        saved_b = _load_state(
            state[self.index * 5 + State.ATT_B], self.hidden_size, "float32"
        )
        saved_p = _load_state(
            state[self.index * 5 + State.ATT_P], self.hidden_size, "float32"
        )
        ones = nn.emit(relax.op.ones((self.hidden_size,), self.dtype))
        xk = nn.emit(x * self.time_mix_key + saved_x * (ones - self.time_mix_key))
        xv = nn.emit(x * self.time_mix_value + saved_x * (ones - self.time_mix_value))
        xr = nn.emit(
            x * self.time_mix_receptance + saved_x * (ones - self.time_mix_receptance)
        )

        r = nn.emit(op.sigmoid(op.matmul(xr, self.receptance_weight)))
        k = nn.emit(op.astype(op.matmul(xk, self.key_weight), "float32"))
        v = nn.emit(op.astype(op.matmul(xv, self.value_weight), "float32"))

        w = nn.emit(k + self.time_first)
        p = nn.emit(op.maximum(saved_p, w))
        e1 = nn.emit(op.exp(saved_p - p))
        e2 = nn.emit(op.exp(w - p))
        wkv = nn.emit(
            op.astype((e1 * saved_a + e2 * v) / (e1 * saved_b + e2), self.dtype)
        )
        w = nn.emit(saved_p + self.time_decay)
        p = nn.emit(op.maximum(w, k))
        e1 = nn.emit(op.exp(w - p))
        e2 = nn.emit(op.exp(k - p))

        aa = nn.emit(e1 * saved_a + e2 * v)
        bb = nn.emit(e1 * saved_b + e2)

        saved_x = _store_state(state[self.index * 5 + State.ATT_X], x)
        saved_a = _store_state(state[self.index * 5 + State.ATT_A], aa)
        saved_b = _store_state(state[self.index * 5 + State.ATT_B], bb)
        saved_p = _store_state(state[self.index * 5 + State.ATT_P], p)

        return nn.emit(op.matmul(r * wkv, self.output_weight)), [
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

    def forward(self, input_ids: Expr, state: Expr) -> Tuple[Expr, List[Expr]]:
        hidden_states = self.embeddings(input_ids)
        states = []
        for _, layer in enumerate(self.blocks):
            hidden_states, layer_states = layer(hidden_states, state)
            states += layer_states
        hidden_states = self.ln_out(hidden_states)
        return hidden_states, states


class RWKVForCausalLM(nn.Module):
    def __init__(self, config: RWKVConfig):
        self.rwkv = RWKVModel(config)
        self.head_weight = nn.Parameter(
            (config.hidden_size, config.vocab_size),
            dtype=config.dtype,
            name="head_weight",
        )
        self.vocab_size = config.vocab_size
        ############ End ############

    def forward(
        self,
        input_ids: relax.Expr,
        state: relax.Expr,
    ):
        hidden_states, key_value_cache = self.rwkv(input_ids, state)
        logits = nn.emit(op.matmul(hidden_states, self.head_weight))
        logits = nn.emit(op.reshape(logits, (1, 1, self.vocab_size)))
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, key_value_cache


def create_decoding_func(bb: relax.BlockBuilder, config: RWKVConfig) -> Dict[int, str]:
    pidx2pname: Dict[int, str] = {}
    with bb.function("decode"):
        model = RWKVForCausalLM(config)
        input_ids = nn.Placeholder((1, 1), dtype="int32", name="input_ids")
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

            named_params = named_parameters(model)
            for i, (name, param) in enumerate(named_params.items()):
                pidx2pname[i] = name
                assert param.same_as(params[i + 3])

            gv = bb.emit_output((logits, relax.Tuple(states)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("decode")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))

    return pidx2pname


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
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    model_name = args.model
    max_seq_len = args.max_seq_len
    dtype = args.quantization.model_dtype

    if not model_name.startswith("rwkv-"):
        raise ValueError(f"Unsupported model name: {model_name}")

    config = RWKVConfig(**hf_config, dtype=dtype)
    if max_seq_len != -1:
        config.context_length = max_seq_len

    bb = relax.BlockBuilder()
    pidx2pname = create_decoding_func(bb, config)
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

    def f_convert_pname_fwd(pname: str) -> str:
        if (
            "key_weight" in pname
            or "value_weight" in pname
            or "receptance_weight" in pname
            or "output_weight" in pname
            or "head_weight" in pname
        ):
            return pname.replace("_weight", ".weight")
        else:
            return pname

    pname2binname = load_torch_pname2binname_map(
        args.model_path, set(pidx2pname.values()), f_convert_pname_fwd
    )

    def f_convert_param_bkwd(torch_pname: str, raw_param):
        # raw_param: numpy.ndarray
        import numpy as np  # pylint: disable=import-outside-toplevel

        # rescale_every
        if config.rescale_every > 0 and "blocks." in torch_pname:
            # based-on the assumption that the layer id is the second element in torch_pname
            layer_id = int(torch_pname.split(".")[2])
            if (
                "attention.output.weight" in torch_pname
                or "feed_forward.value.weight" in torch_pname
            ):
                raw_param = raw_param / (2 ** (layer_id // config.rescale_every))

        # reshape
        if "time_" in torch_pname:
            raw_param = raw_param.squeeze()
        if (
            "key.weight" in torch_pname
            or "value.weight" in torch_pname
            or "receptance.weight" in torch_pname
            or "output.weight" in torch_pname
            or "head.weight" in torch_pname
        ):
            pname = torch_pname.replace(".weight", "_weight")
            raw_param = raw_param.T
        else:
            pname = torch_pname

        # convert dtype
        if "time_decay" in torch_pname:  # need fp32 for this
            return [(pname, -np.exp(raw_param.astype("float32")))]
        elif "time_first" in torch_pname:
            return [(pname, raw_param.astype("float32"))]
        else:
            return [(pname, raw_param.astype(config.dtype))]

    args.pidx2pname = pidx2pname
    args.pname2binname = pname2binname
    args.f_convert_pname_fwd = f_convert_pname_fwd
    args.f_convert_param_bkwd = f_convert_param_bkwd

    return mod, [None] * len(pidx2pname)
