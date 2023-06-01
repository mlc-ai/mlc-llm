# pylint: disable=missing-docstring,too-few-public-methods,too-many-instance-attributes,invalid-name,too-many-locals,too-many-arguments
import argparse
import math
from typing import Dict, List, Optional, Tuple, Union

import tvm
from tvm import relax, te
from tvm.relax.op import (
    astype,
    broadcast_to,
    matmul,
    maximum,
    minimum,
    permute_dims,
    reshape,
    squeeze,
)
from tvm.relax.op.nn import gelu, softmax
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..utils import load_torch_pname2binname_map
from .commons import create_metadata_func
from .modules import (
    Embedding,
    LayerNorm,
    Linear,
    ModuleList,
    RotaryEmbedding,
    named_parameters,
)


class GPTNeoXConfig:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        use_parallel_residual,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
        vocab_size,
        rotary_pct,
        rotary_emb_base,
        layer_norm_eps,
        max_sequence_length,
        dtype,
        ffn_out_dtype,
        **kwargs,
    ):
        self.use_parallel_residual = use_parallel_residual
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.layer_norm_eps = layer_norm_eps
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.ffn_out_dtype = ffn_out_dtype
        self.kwargs = kwargs


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
        attn_weights = nn.emit(
            maximum(
                attn_weights,
                relax.const(
                    tvm.tir.min_value(attn_weights.struct_info.dtype).value,
                    attn_weights.struct_info.dtype,
                ),
            )
        )
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
        out_dtype: Optional[str],
    ):
        super().__init__()
        if out_dtype is None:
            out_dtype = dtype
        self.dense_h_to_4h = Linear(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.dense_4h_to_h = Linear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.dtype = dtype

    def forward(self, hidden_states):
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = nn.emit(gelu(hidden_states))
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        hidden_states = self.dense_4h_to_h(hidden_states)
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        return hidden_states


class GPTNeoXLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_heads: int,
        use_parallel_residual: bool,
        rotary_embedding: RotaryEmbedding,
        dtype: str,
        ffn_out_dtype: Optional[str],
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
            hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            out_dtype=ffn_out_dtype,
        )
        self.use_parallel_residual = use_parallel_residual
        self.dtype = dtype

    def forward(
        self,
        hidden_states,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ):
        attn_input = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.attention(
            attn_input,
            all_seq_len_shape,
            past_key_value,
            attention_mask,
        )
        if self.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = nn.emit(mlp_output + attn_output + hidden_states)
        else:
            attn_output = nn.emit(attn_output + hidden_states)
            mlp_input = self.post_attention_layernorm(attn_output)
            mlp_output = self.mlp(mlp_input)
            hidden_states = nn.emit(astype(mlp_output, self.dtype) + attn_output)
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
                    j < src_len - tgt_len,
                    tvm.tir.max_value(dtype),
                    x[b, _, i, j - (src_len - tgt_len)],
                ),
                name="concat_te",
            )

        return nn.emit_te(extend_te, diag_mask, tgt_len, src_len)
    else:
        # Get src_len from input parameters
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        bsz, tgt_len = input_shape
        mask = relax.op.full(
            (bsz, 1, tgt_len, src_len),
            relax.const(tvm.tir.max_value(dtype).value, dtype),
            dtype,
        )
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
                    use_parallel_residual=config.use_parallel_residual,
                    dtype=config.dtype,
                    ffn_out_dtype=config.ffn_out_dtype,
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
) -> Dict[int, str]:
    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.Var("n", "int64")
    all_seq_len = tvm.tir.Var("m", "int64")
    pidx2pname: Dict[int, str] = {}
    with bb.function("prefill"):
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
            ] + model.parameters()
            named_params = named_parameters(model)
            for i, (name, param) in enumerate(named_params.items()):
                pidx2pname[i] = name
                assert param.same_as(params[i + 3])

            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var("prefill")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))

    return pidx2pname


def create_decoding_func(
    bb: relax.BlockBuilder,
    config: GPTNeoXConfig,
) -> None:
    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.IntImm("int64", 1)
    all_seq_len = tvm.tir.Var("n", "int64")
    with bb.function("decode"):
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
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var("decode")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_kv_cache_func(
    bb: relax.BlockBuilder,
    config: GPTNeoXConfig,
) -> None:
    init_shape = relax.ShapeExpr(
        (
            config.max_sequence_length,
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


def create_softmax_func(bb: relax.BlockBuilder, config: GPTNeoXConfig) -> None:
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


def get_model(
    args: argparse.Namespace,
    hf_config,
):
    model = args.model
    dtype = args.quantization.model_dtype
    ffn_out_dtype = "float32"

    if model.startswith("dolly-"):
        stop_tokens = [2]
        ffn_out_dtype = "float16"
    elif model.startswith("stablelm-"):
        stop_tokens = [50278, 50279, 50277, 1, 0]
        ffn_out_dtype = "float16"
    elif model.lower().startswith("redpajama-"):
        stop_tokens = [0]
    else:
        raise ValueError(f"Unsupported model {model}")

    config = GPTNeoXConfig(
        **hf_config,
        max_sequence_length=args.max_seq_len if args.max_seq_len != -1 else 2048,
        dtype=dtype,
        ffn_out_dtype=ffn_out_dtype,
    )

    bb = relax.BlockBuilder()
    pidx2pname = create_encoding_func(bb, config)
    create_decoding_func(bb, config)
    create_kv_cache_func(bb, config)
    create_softmax_func(bb, config)
    create_metadata_func(
        bb,
        model_name=model,
        max_window_size=config.max_sequence_length,
        stop_tokens=stop_tokens,
        add_prefix_space=False,
    )
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

    def f_convert_pname_fwd(pname: str) -> str:
        import re  # pylint: disable=import-outside-toplevel

        str_pattern = re.compile(r"(q|k|v)_proj")
        if re.search(str_pattern, pname) is not None:
            return str_pattern.sub("query_key_value", pname)
        else:
            return pname

    pname2binname = load_torch_pname2binname_map(
        args.model_path, set(pidx2pname.values()), f_convert_pname_fwd
    )

    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    def f_convert_param_bkwd(torch_pname: str, raw_param):
        # raw_param: numpy.ndarray
        if torch_pname.endswith("query_key_value.weight"):
            assert raw_param.ndim == 2
            raw_param = raw_param.astype(dtype).reshape(
                num_heads, 3, head_dim, hidden_size
            )
            q_weight = raw_param[:, 0, :, :].reshape(hidden_size, hidden_size)
            k_weight = raw_param[:, 1, :, :].reshape(hidden_size, hidden_size)
            v_weight = raw_param[:, 2, :, :].reshape(hidden_size, hidden_size)
            return [
                (torch_pname.replace("query_key_value", "q_proj"), q_weight),
                (torch_pname.replace("query_key_value", "k_proj"), k_weight),
                (torch_pname.replace("query_key_value", "v_proj"), v_weight),
            ]
        elif torch_pname.endswith("query_key_value.bias"):
            assert raw_param.ndim == 1
            raw_param = raw_param.astype(dtype).reshape(num_heads, 3, head_dim)
            q_bias = raw_param[:, 0, :].reshape(hidden_size)
            k_bias = raw_param[:, 1, :].reshape(hidden_size)
            v_bias = raw_param[:, 2, :].reshape(hidden_size)
            return [
                (torch_pname.replace("query_key_value", "q_proj"), q_bias),
                (torch_pname.replace("query_key_value", "k_proj"), k_bias),
                (torch_pname.replace("query_key_value", "v_proj"), v_bias),
            ]
        elif (
            "layernorm" in torch_pname
            or "layer_norm" in torch_pname
            or "embed_out" in torch_pname
        ):
            return [(torch_pname, raw_param.astype("float32"))]
        elif (
            ".dense_h_to_4h.bias" in torch_pname or ".dense_4h_to_h.bias" in torch_pname
        ):
            return [(torch_pname, raw_param.astype(ffn_out_dtype))]
        else:
            return [(torch_pname, raw_param.astype(dtype))]

    args.pidx2pname = pidx2pname
    args.pname2binname = pname2binname
    args.f_convert_pname_fwd = f_convert_pname_fwd
    args.f_convert_param_bkwd = f_convert_param_bkwd

    return mod, [None] * len(pidx2pname)
