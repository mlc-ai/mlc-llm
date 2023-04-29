# pylint: disable=missing-docstring,too-few-public-methods,too-many-instance-attributes,invalid-name,too-many-locals,too-many-arguments
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import tvm
from tvm import relax, te
from tvm.relax.op import (
    astype,
    broadcast_to,
    full,
    matmul,
    maximum,
    minimum,
    permute_dims,
    reshape,
    squeeze,
    triu,
)
from tvm.relax.op.nn import gelu, softmax
from tvm.relax.testing import nn
from tvm.runtime import NDArray
from tvm.script import relax as R

from .modules import (
    Embedding,
    LayerNorm,
    Linear,
    ModuleList,
    RotaryEmbedding,
    named_parameters,
)


@dataclass
class GPTNeoXConfig:  # pylint: disable=too-many-instance-attributes
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    dtype: str = "float32"
    layer_norm_eps: float = 1e-05
    max_sequence_length: int = 2048
    rotary_emb_base: int = 10000
    rotary_pct: float = 0.25


MODEL_CONFIG = {
    "dolly-v2-3b": {
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 50280,
    },
    "dolly-v2-7b": {
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 50280,
    },
    "dolly-v2-12b": {
        "hidden_size": 5120,
        "intermediate_size": 20480,
        "num_attention_heads": 40,
        "num_hidden_layers": 36,
        "vocab_size": 50280,
    },
    "stablelm-tuned-alpha-3b": {
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "vocab_size": 50688,
    },
    "stablelm-tuned-alpha-7b": {
        "hidden_size": 6144,
        "intermediate_size": 24576,
        "num_attention_heads": 48,
        "num_hidden_layers": 16,
        "vocab_size": 50432,
    },
}


def _min_value(dtype) -> relax.Expr:
    v = tvm.tir.min_value(dtype).value
    if dtype == "float16":
        v = -55504.0
    return relax.const(v, dtype)


def _max_value(dtype) -> relax.Expr:
    v = tvm.tir.max_value(dtype).value
    if dtype == "float16":
        v = 55504.0
    return relax.const(v, dtype)


class GPTNeoXAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_embedding: RotaryEmbedding,
        dtype: str,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_embedding = rotary_embedding
        self.q_proj = Linear(hidden_size, hidden_size, dtype, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, dtype, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, dtype, bias=True)
        self.dense = Linear(hidden_size, hidden_size, dtype, bias=True)
        self.dtype = dtype

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr, relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Union[Tuple[None, None], Tuple[relax.Expr, relax.Expr]]]:
        # hidden_states: [batch_size, seq_len, hidden_size]
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        batch_size, seq_len, _ = hidden_states.struct_info.shape
        kv_seq_len = all_seq_len_shape.struct_info.values[0]

        def _project(proj):
            return nn.emit(
                reshape(
                    proj(hidden_states),
                    (batch_size, seq_len, self.num_heads, self.head_dim),
                )
            )

        # q/k/v states: [batch_size, seq_len, num_attention_heads, head_size]
        q, k, v = (
            _project(self.q_proj),
            _project(self.k_proj),
            _project(self.v_proj),
        )
        q, k = self.rotary_embedding(q, k, kv_seq_len - seq_len)

        if past_key_value is not None:
            f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
            f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
            k_cache, v_cache = past_key_value
            k_cache = nn.emit(
                relax.Call(
                    f_kv_cache_append,
                    args=[k_cache, squeeze(k, axis=0)],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            v_cache = nn.emit(
                relax.Call(
                    f_kv_cache_append,
                    args=[v_cache, squeeze(v, axis=0)],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            batch_size, _, num_heads, head_size = k.struct_info.shape
            kv_cache_shape = R.shape([kv_seq_len, num_heads, head_size])
            kv_states_shape = R.shape([batch_size, kv_seq_len, num_heads, head_size])
            k = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[k_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, k.struct_info.dtype)],
                )
            )
            v = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[v_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, v.struct_info.dtype)],
                )
            )
            k = nn.emit(reshape(k, kv_states_shape))
            v = nn.emit(reshape(v, kv_states_shape))
            past_key_value = (k_cache, v_cache)
        else:
            past_key_value = (None, None)

        q = nn.emit(permute_dims(q, [0, 2, 1, 3]))
        k = nn.emit(permute_dims(k, [0, 2, 1, 3]))
        v = nn.emit(permute_dims(v, [0, 2, 1, 3]))

        # Calculate QK
        attn_weights = nn.emit(
            matmul(q, permute_dims(k, [0, 1, 3, 2]))
            / relax.const(
                math.sqrt(self.head_dim),
                q.struct_info.dtype,
            )
        )
        # Apply attention mask
        attn_weights = nn.emit(maximum(attn_weights, relax.const(tvm.tir.min_value(attn_weights.struct_info.dtype).value, attn_weights.struct_info.dtype)))
        attn_weights = nn.emit(minimum(attn_weights, attention_mask))
        # Calculate Softmax(QK)
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != q.struct_info.dtype:
            attn_weights = astype(attn_weights, q.struct_info.dtype)
        # Calculate Softmax(QK)V
        attn_output = nn.emit(matmul(attn_weights, v))
        # Apply output projection
        attn_output = self.dense(
            reshape(
                permute_dims(attn_output, [0, 2, 1, 3]),
                (batch_size, seq_len, self.hidden_size),
            )
        )
        return attn_output, past_key_value


class GPTNeoXMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: str,
    ):
        super().__init__()
        self.dense_h_to_4h = Linear(hidden_size, intermediate_size, dtype)
        self.dense_4h_to_h = Linear(intermediate_size, hidden_size, dtype)
        self.dtype = dtype

    def forward(self, hidden_states):
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return nn.emit(hidden_states)


class GPTNeoXLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_heads: int,
        rotary_embedding: RotaryEmbedding,
        dtype: str,
    ):
        self.input_layernorm = LayerNorm(
            hidden_size,
            eps=layer_norm_eps,
            dtype=dtype,
        )
        self.post_attention_layernorm = LayerNorm(
            hidden_size,
            eps=layer_norm_eps,
            dtype=dtype,
        )
        self.attention = GPTNeoXAttention(
            hidden_size,
            num_heads=num_heads,
            rotary_embedding=rotary_embedding,
            dtype=dtype,
        )
        self.mlp = GPTNeoXMLP(
            hidden_size, intermediate_size=intermediate_size, dtype=dtype
        )
        self.dtype = dtype

    def forward(
        self,
        hidden_states,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ):
        attn_input = self.input_layernorm(hidden_states)
        mlp_input = self.post_attention_layernorm(hidden_states)
        attn_output, present_key_value = self.attention(
            attn_input,
            all_seq_len_shape,
            past_key_value,
            attention_mask,
        )
        mlp_output = self.mlp(mlp_input)
        hidden_states = nn.emit(mlp_output + attn_output + hidden_states)
        return hidden_states, present_key_value


def _prepare_decoder_attention_mask(input_shape, src_len, dtype):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    if isinstance(input_shape[-1], tvm.tir.Var) or input_shape[-1] > 1:
        bsz, tgt_len = input_shape
        def min_max_triu_te():
            return te.compute(
                (tgt_len, tgt_len),
                lambda i, j: tvm.tir.Select(
                    j > i, tvm.tir.min_value(dtype), tvm.tir.max_value(dtype)
                ),
                name="make_diag_mask_te",
            )
    
        mask = nn.emit_te(min_max_triu_te)
        diag_mask = nn.emit(broadcast_to(mask, (bsz, 1, tgt_len, tgt_len)))
        if src_len == tgt_len:
            return diag_mask

        def extend_te(x, tgt_len, src_len):
            return te.compute(
                (bsz, 1, tgt_len, src_len),
                lambda b, _, i, j: te.if_then_else(
                    j < src_len - tgt_len, tvm.tir.max_value(dtype), x[b, _, i, j - (src_len - tgt_len)]
                ),
                name="concat_te",
            )

        return nn.emit_te(extend_te, diag_mask, tgt_len, src_len)
    else:
        # Get src_len from input parameters
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        bsz, tgt_len = input_shape
        mask = relax.op.full((bsz, 1, tgt_len, src_len), relax.const(tvm.tir.max_value(dtype).value, dtype), dtype)
    return nn.emit(mask)


class GPTNeoXModel(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
    ):
        rotary_embedding = RotaryEmbedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            position_embedding_base=config.rotary_emb_base,
            max_sequence_length=config.max_sequence_length,
            rotary_pct=config.rotary_pct,
            dtype=config.dtype,
        )
        self.embed_in = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        self.layers = ModuleList(
            [
                GPTNeoXLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    layer_norm_eps=config.layer_norm_eps,
                    num_heads=config.num_attention_heads,
                    rotary_embedding=rotary_embedding,
                    dtype=config.dtype,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = LayerNorm(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[Tuple[relax.Expr, relax.Expr]],
    ):
        batch_size, seq_length = input_ids.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]
        # embed positions
        hidden_states = self.embed_in(input_ids)
        attention_mask = _prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            dtype=hidden_states.struct_info.dtype,
        )
        present_kv_cache = []
        for i, layer in enumerate(self.layers):
            past_key_value = (
                (past_key_values[i * 2], past_key_values[i * 2 + 1])
                if past_key_values is not None
                else None
            )
            hidden_states, (present_k_cache, present_v_cache) = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
            )
            present_kv_cache.append(present_k_cache)
            present_kv_cache.append(present_v_cache)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, present_kv_cache


class GPTNeoXForCausalLM(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
    ):
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype="float32",
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[List[relax.Expr]],
    ):
        hidden_states, key_value_cache = self.gpt_neox(
            input_ids=input_ids,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def _slice(x: te.Tensor):
            _, seq_len, hidden_dim = x.shape
            return te.compute(
                shape=(1, 1, hidden_dim),
                fcompute=lambda i, _, k: x[i, seq_len - 1, k],
                name="slice",
            )

        hidden_states = nn.emit_te(
            _slice,
            hidden_states,
            primfunc_name_hint="slice",
        )
        hidden_states = astype(hidden_states, "float32")
        logits = self.embed_out(hidden_states)
        return logits, key_value_cache


def create_encoding_func(
    bb: relax.BlockBuilder,
    config: GPTNeoXConfig,
    ordered_params: List[str],
) -> None:
    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.Var("n", "int64")
    all_seq_len = tvm.tir.Var("m", "int64")
    with bb.function("encoding"):
        model = GPTNeoXForCausalLM(config)
        input_ids = nn.Placeholder(
            (batch_size, seq_len), dtype="int32", name="input_ids"
        )
        all_seq_len_shape = relax.Var(
            "all_seq_len", relax.ShapeStructInfo((all_seq_len,))
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids=input_ids,
                all_seq_len_shape=all_seq_len_shape,
                past_key_values=past_key_values,
            )
            params = [
                input_ids,
                all_seq_len_shape,
                past_key_values,
            ]
            named_params = named_parameters(model)
            for name in ordered_params:
                params.append(named_params[name])
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var("encoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(
    bb: relax.BlockBuilder,
    config: GPTNeoXConfig,
    ordered_params: List[str],
) -> None:
    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.IntImm("int64", 1)
    all_seq_len = tvm.tir.Var("n", "int64")
    with bb.function("decoding"):
        model = GPTNeoXForCausalLM(config)
        input_ids = nn.Placeholder(
            (batch_size, seq_len), dtype="int32", name="input_ids"
        )
        all_seq_len_shape = relax.Var(
            "all_seq_len",
            relax.ShapeStructInfo((all_seq_len,)),
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids=input_ids,
                all_seq_len_shape=all_seq_len_shape,
                past_key_values=past_key_values,
            )
            params = [
                input_ids,
                all_seq_len_shape,
                past_key_values,
            ]
            named_params = named_parameters(model)
            for name in ordered_params:
                params.append(named_params[name])
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var("decoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_kv_cache_func(
    bb: relax.BlockBuilder,
    config: GPTNeoXConfig,
) -> None:
    init_shape = relax.ShapeExpr(
        (
            1,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )
    )
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            for _ in range(config.num_hidden_layers * 2):
                caches.append(
                    bb.emit(
                        relax.Call(
                            f_kv_cache_create,
                            args=[zeros, init_shape, relax.PrimValue(0)],
                            sinfo_args=[relax.ObjectStructInfo()],
                        )
                    )
                )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def get_model(
    model_name: str,
    model_path: str,
    dtype: str,
):
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    if model_name.startswith("dolly-") or model_name.startswith("stablelm-"):
        config = GPTNeoXConfig(**MODEL_CONFIG[model_name], dtype=dtype)
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        param_list: List[Tuple[str, NDArray]] = []
        hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        for name, param in hf_model.named_parameters():
            param = param.detach().cpu().numpy()
            if param.dtype == "float32":
                if "layernorm" in name or "layer_norm" in name or "embed_out" in name:
                    param = param.astype("float32")
                else:
                    param = param.astype(dtype)
            if name.endswith("query_key_value.weight"):
                name = name.replace("query_key_value.weight", "{}_proj.weight")
                assert param.ndim == 2
                param = param.reshape(num_heads, 3, head_dim, hidden_size)
                q = param[:, 0, :, :].reshape(hidden_size, hidden_size)
                k = param[:, 1, :, :].reshape(hidden_size, hidden_size)
                v = param[:, 2, :, :].reshape(hidden_size, hidden_size)
                param_list.append((name.format("q"), q))
                param_list.append((name.format("k"), k))
                param_list.append((name.format("v"), v))
            elif name.endswith("query_key_value.bias"):
                name = name.replace("query_key_value.bias", "{}_proj.bias")
                assert param.ndim == 1
                param = param.reshape(num_heads, 3, head_dim)
                q = param[:, 0, :].reshape(hidden_size)
                k = param[:, 1, :].reshape(hidden_size)
                v = param[:, 2, :].reshape(hidden_size)
                param_list.append((name.format("q"), q))
                param_list.append((name.format("k"), k))
                param_list.append((name.format("v"), v))
            else:
                param_list.append((name, param))
        del hf_model
        param_list = [
            (
                name,
                tvm.nd.array(param, tvm.cpu()),
            )
            for name, param in param_list
        ]
        bb = relax.BlockBuilder()
        create_encoding_func(bb, config, [k for k, _ in param_list])
        create_decoding_func(bb, config, [k for k, _ in param_list])
        create_kv_cache_func(bb, config)
        mod = bb.get()
        for gv in mod.functions:
            func = mod[gv]
            if isinstance(func, relax.Function):
                mod[gv] = func.with_attr(
                    "tir_var_upper_bound",
                    {
                        "n": config.max_sequence_length,
                        "m": config.max_sequence_length,
                    },
                )
        return mod, [v for _, v in param_list]
    raise ValueError(f"Unsupported model: {model_name}")
