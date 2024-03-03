# pylint: disable=missing-docstring,too-few-public-methods,too-many-instance-attributes,invalid-name,too-many-locals,too-many-arguments
import argparse
import math
from typing import List, Optional, Tuple, Union

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

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import Embedding, LayerNorm, Linear, ModuleList, RotaryEmbedding
from .param_manager import ParamManager


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
        self.query_key_value = Linear(hidden_size, hidden_size * 3, dtype, bias=True)
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

        # qkv_states: [batch_size, seq_len, hidden_size * 3]
        qkv_states = nn.emit(
            relax.op.split(
                reshape(
                    self.query_key_value(hidden_states),
                    (batch_size, seq_len, self.num_heads, 3 * self.head_dim),
                ),
                indices_or_sections=3,
                axis=-1,
            )
        )

        # q/k/v states: [batch_size, seq_len, num_attention_heads, head_size]
        q, k, v = [relax.TupleGetItem(qkv_states, idx) for idx in range(3)]
        q, k = self.rotary_embedding(q, k, kv_seq_len - seq_len)

        if past_key_value is not None:
            f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
            f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
            k_cache, v_cache = past_key_value
            k_cache = nn.emit(
                relax.op.call_inplace_packed(
                    f_kv_cache_append,
                    k_cache,
                    squeeze(k, axis=0),
                    inplace_indices=[0],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            v_cache = nn.emit(
                relax.op.call_inplace_packed(
                    f_kv_cache_append,
                    v_cache,
                    squeeze(v, axis=0),
                    inplace_indices=[0],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            batch_size, _, num_heads, head_size = k.struct_info.shape
            kv_cache_shape = R.shape([kv_seq_len, num_heads, head_size])
            kv_states_shape = R.shape([batch_size, kv_seq_len, num_heads, head_size])
            k = nn.emit(
                relax.call_pure_packed(
                    f_kv_cache_view,
                    k_cache,
                    kv_cache_shape,
                    sinfo_args=[R.Tensor(kv_cache_shape, k.struct_info.dtype)],
                )
            )
            v = nn.emit(
                relax.call_pure_packed(
                    f_kv_cache_view,
                    v_cache,
                    kv_cache_shape,
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
    if isinstance(input_shape[-1], tvm.tir.SizeVar) or input_shape[-1] > 1:
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


class GPTNeoXEmbedTokens(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        self.embed_in = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(self, input_ids: relax.Expr):
        return self.embed_in(input_ids)


class GPTNeoXEmbedTokensWrapper(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        # build a wrapper to ensure that the naming of the embed_in parameter is consistent
        self.gpt_neox = GPTNeoXEmbedTokens(config)

    def forward(self, input_ids: relax.Expr):
        return self.gpt_neox(input_ids)


class GPTNeoXModel(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        sep_embed: bool = False,
    ):
        rotary_embedding = RotaryEmbedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            position_embedding_base=config.rotary_emb_base,
            max_sequence_length=config.max_sequence_length,
            rotary_pct=config.rotary_pct,
            dtype=config.dtype,
        )

        self.embed_in = None
        if not sep_embed:
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
        inputs: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[Tuple[relax.Expr, relax.Expr]],
    ):
        # embed positions
        hidden_states = self.embed_in(inputs) if self.embed_in else inputs

        batch_size, seq_length, _ = hidden_states.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]
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
        sep_embed: bool = False,
    ):
        self.gpt_neox = GPTNeoXModel(config, sep_embed)
        self.embed_out = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype="float32",
        )

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[List[relax.Expr]],
    ):
        hidden_states, key_value_cache = self.gpt_neox(
            inputs=inputs,
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


def get_param_quant_kind(name: str, param_info: relax.TensorStructInfo) -> ParamQuantKind:
    if "embed_in.weight" in name:
        return ParamQuantKind.embedding_table
    elif "embed_out.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_embed_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPTNeoXConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    bsz = 1
    seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = GPTNeoXEmbedTokensWrapper(config)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        with bb.dataflow():
            inputs_embeds = model(input_ids)
            params = [input_ids] + model.parameters()
            gv = bb.emit_output(inputs_embeds)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("embed")
    bb.update_func(gv, mod[gv].with_attr("num_input", 1))


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPTNeoXConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.SizeVar("n", "int64")
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = GPTNeoXForCausalLM(config, sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = (
            nn.Placeholder(
                (batch_size, seq_len, hidden_size),
                dtype=config.dtype,
                name="input_embeds",
            )
            if sep_embed
            else nn.Placeholder((batch_size, seq_len), dtype="int32", name="input_ids")
        )
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                inputs=inputs,
                all_seq_len_shape=all_seq_len_shape,
                past_key_values=past_key_values,
            )
            params = [
                inputs,
                all_seq_len_shape,
                past_key_values,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPTNeoXConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.IntImm("int64", 1)
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = GPTNeoXForCausalLM(config)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((batch_size, seq_len), dtype="int32", name="input_ids")
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
                inputs=input_ids,
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
    gv = mod.get_global_var(func_name)
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
                        relax.call_pure_packed(
                            f_kv_cache_create,
                            zeros,
                            init_shape,
                            relax.PrimValue(0),
                            sinfo_args=[relax.ObjectStructInfo()],
                        )
                    )
                )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_softmax_func(bb: relax.BlockBuilder, config: GPTNeoXConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder((1, 1, config.vocab_size), dtype="float32", name="logits")
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
    sep_embed = args.sep_embed

    if model.startswith("dolly-"):
        stop_tokens = [2]
        ffn_out_dtype = "float16"
    elif model.startswith("stablelm-"):
        stop_tokens = [50278, 50279, 50277, 1, 0]
        ffn_out_dtype = "float16"
    elif model.lower().startswith("stablecode-"):
        stop_tokens = [0]
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

    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    if sep_embed:
        create_embed_func(bb, param_manager, config, args.quantization)
    create_encoding_func(bb, param_manager, config, args.quantization, sep_embed)
    create_decoding_func(bb, param_manager, config, args.quantization)
    create_kv_cache_func(bb, config)
    create_softmax_func(bb, config)
    create_metadata_func(
        bb,
        model_name=model,
        max_window_size=config.max_sequence_length,
        stop_tokens=stop_tokens,
        add_prefix_space=False,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    mod = bb.get()

    tir_bound_map = dict()
    tir_bound_map["n"] = (
        args.prefill_chunk_size if args.prefill_chunk_size > 0 else config.max_sequence_length
    )
    tir_bound_map["m"] = config.max_sequence_length
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, relax.Function):
            mod[gv] = func.with_attr("tir_var_upper_bound", tir_bound_map)

    if args.build_model_only:
        return mod, param_manager, None, config

    def f_convert_pname_fwd(pname: str) -> List[str]:
        return [pname]

    def f_convert_param_bkwd(torch_pname: str, torch_param):
        # torch_param: numpy.ndarray
        if "layernorm" in torch_pname or "layer_norm" in torch_pname or "embed_out" in torch_pname:
            return [(torch_pname, torch_param.astype("float32"))]
        elif ".dense_h_to_4h.bias" in torch_pname or ".dense_4h_to_h.bias" in torch_pname:
            return [(torch_pname, torch_param.astype(ffn_out_dtype))]
        else:
            return [(torch_pname, torch_param.astype(dtype))]

    param_manager.set_param_loading_func(
        args.model_path, args.use_safetensors, f_convert_pname_fwd, f_convert_param_bkwd
    )
    return mod, param_manager, [None] * len(param_manager.param_names), config
