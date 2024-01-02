import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

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
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .gpt_neox import create_kv_cache_func
from .modules import Embedding, LayerNorm, Linear, ModuleList, RotaryEmbedding
from .param_manager import ParamManager


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


@dataclass
class GPTJConfig:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        vocab_size,
        n_embd,
        n_inner,
        n_head,
        n_layer,
        bos_token_id,
        eos_token_id,
        rotary_dim,
        tie_word_embeddings,
        dtype="float32",
        layer_norm_eps=1e-5,
        max_sequence_length=2048,
        rotary_emb_base=10000,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = n_embd
        self.intermediate_size = n_inner if n_inner is not None else 4 * n_embd
        self.num_attention_heads = n_head
        self.num_hidden_layers = n_layer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rotary_dim = rotary_dim
        self.tie_word_embeddings = tie_word_embeddings
        self.dtype = dtype
        self.layer_norm_eps = layer_norm_eps
        self.max_sequence_length = max_sequence_length
        self.rotary_emb_base = rotary_emb_base
        self.kwargs = kwargs


class GPTJMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: str):
        super().__init__()
        self.fc_in = Linear(hidden_size, intermediate_size, dtype, bias=True)
        self.fc_out = Linear(intermediate_size, hidden_size, dtype, bias=True)
        self.dtype = dtype

    def forward(self, hidden_states):
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        hidden_states = self.fc_in(hidden_states)
        hidden_states = nn.emit(gelu(hidden_states))
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        hidden_states = self.fc_out(hidden_states)
        return nn.emit(hidden_states)


class GPTJAttention(nn.Module):
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
        self.q_proj = Linear(hidden_size, hidden_size, dtype, bias=False)
        self.k_proj = Linear(hidden_size, hidden_size, dtype, bias=False)
        self.v_proj = Linear(hidden_size, hidden_size, dtype, bias=False)
        self.out_proj = Linear(hidden_size, hidden_size, dtype, bias=False)
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
        attn_weights = nn.emit(attn_weights + attention_mask)
        attn_weights = nn.emit(
            minimum(
                maximum(
                    attn_weights,
                    _min_value(attn_weights.struct_info.dtype),
                ),
                _max_value(attn_weights.struct_info.dtype),
            )
        )
        # Calculate Softmax(QK)
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != q.struct_info.dtype:
            attn_weights = astype(attn_weights, q.struct_info.dtype)
        # Calculate Softmax(QK)V
        attn_output = nn.emit(matmul(attn_weights, v))
        # Apply output projection
        attn_output = self.out_proj(
            reshape(
                permute_dims(attn_output, [0, 2, 1, 3]),
                (batch_size, seq_len, self.hidden_size),
            )
        )
        return attn_output, past_key_value


class GPTJLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_heads: int,
        rotary_embedding: RotaryEmbedding,
        dtype: str,
    ):
        self.ln_1 = LayerNorm(
            hidden_size,
            eps=layer_norm_eps,
            dtype=dtype,
        )
        self.attn = GPTJAttention(
            hidden_size,
            num_heads=num_heads,
            rotary_embedding=rotary_embedding,
            dtype=dtype,
        )
        self.mlp = GPTJMLP(
            hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )
        self.dtype = dtype

    def forward(
        self,
        hidden_states,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ):
        normalized_input = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attn(
            normalized_input,
            all_seq_len_shape,
            past_key_value,
            attention_mask,
        )
        mlp_output = self.mlp(normalized_input)
        hidden_states = nn.emit(mlp_output + attn_output + hidden_states)
        return hidden_states, present_key_value


def _prepare_decoder_attention_mask(input_shape, src_len, dtype):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    if isinstance(input_shape[-1], tvm.tir.SizeVar) or input_shape[-1] > 1:
        bsz, tgt_len = input_shape
        mask = full((tgt_len, tgt_len), _min_value(dtype))
        mask = triu(mask, k=1)
        diag_mask = nn.emit(broadcast_to(mask, (bsz, 1, tgt_len, tgt_len)))
        if src_len == tgt_len:
            return diag_mask

        def extend_te(x, tgt_len, src_len):
            return te.compute(
                (bsz, 1, tgt_len, src_len),
                lambda b, _, i, j: te.if_then_else(
                    j < src_len - tgt_len, 0, x[b, _, i, j - (src_len - tgt_len)]
                ),
                name="concat_te",
            )

        return nn.emit_te(extend_te, diag_mask, tgt_len, src_len)
    else:
        # Get src_len from input parameters
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        bsz, tgt_len = input_shape
        mask = relax.op.zeros((bsz, 1, tgt_len, src_len), dtype)
    return nn.emit(mask)


class GPTJEmbedTokens(nn.Module):
    def __init__(self, config: GPTJConfig):
        self.wte = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(self, input_ids: relax.Expr):
        return self.wte(input_ids)


class GPTJEmbedTokensWrapper(nn.Module):
    def __init__(self, config: GPTJConfig):
        # build a wrapper to ensure that the naming of the embed_in parameter is consistent
        self.gptj = GPTJEmbedTokens(config)

    def forward(self, input_ids: relax.Expr):
        return self.gptj(input_ids)


class GPTJModel(nn.Module):
    def __init__(
        self,
        config: GPTJConfig,
        sep_embed: bool = False,
    ):
        rotary_embedding = RotaryEmbedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            position_embedding_base=config.rotary_emb_base,
            max_sequence_length=config.max_sequence_length,
            rotary_dim=config.rotary_dim,
            swizzle_style="gptj",
            dtype=config.dtype,
        )
        self.wte = None
        if not sep_embed:
            self.wte = Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                dtype=config.dtype,
            )
        self.h = ModuleList(
            [
                GPTJLayer(
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
        self.ln_f = LayerNorm(
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
        batch_size, seq_length = inputs.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]
        # embed positions
        hidden_states = self.wte(inputs) if self.wte is not None else inputs
        attention_mask = _prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            dtype=hidden_states.struct_info.dtype,
        )
        present_kv_cache = []
        for i, layer in enumerate(self.h):
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
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, present_kv_cache


class GPTJForCausalLM(nn.Module):
    def __init__(
        self,
        config: GPTJConfig,
        sep_embed: bool = False,
    ):
        self.transformer = GPTJModel(config, sep_embed)
        self.lm_head = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=True,
            dtype=config.dtype,
        )
        self.dtype = config.dtype

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[List[relax.Expr]],
    ):
        hidden_states, key_value_cache = self.transformer(
            inputs=inputs,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

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
        logits = self.lm_head(hidden_states)
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(astype(logits, "float32"))

        return logits, key_value_cache


def check_parameters(param_dict, param_list):
    relax_shape_to_list = lambda _: [s.value for s in _.values]
    shape_dict_0 = {k: relax_shape_to_list(v.struct_info.shape) for k, v in param_dict.items()}
    shape_dict_1 = {k: list(v.shape) for (k, v) in param_list}
    assert len(shape_dict_0) == len(shape_dict_1)
    for k, v in shape_dict_0.items():
        assert k in shape_dict_1, "{}".format(k)
        assert v == shape_dict_1[k], "key={}, shape_0={}, shape_1={}".format(k, v, shape_dict_1[k])


def get_param_quant_kind(name: str, param_info: relax.TensorStructInfo) -> ParamQuantKind:
    if "wte.weight" in name:
        return ParamQuantKind.embedding_table
    elif "lm_head.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_embed_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPTJConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    bsz = 1
    seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = GPTJEmbedTokensWrapper(config)
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
    config: GPTJConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.SizeVar("n", "int64")
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = GPTJForCausalLM(config, sep_embed)
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
    config: GPTJConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.IntImm("int64", 1)
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = GPTJForCausalLM(config)
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


def create_softmax_func(bb: relax.BlockBuilder, config: GPTJConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder((1, 1, config.vocab_size), dtype="float32", name="logits")
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    sep_embed = args.sep_embed

    if model_name.startswith("gpt-j-"):
        stop_tokens = [50256]
    elif model_name.startswith("moss-"):
        stop_tokens = [106068]

    config = GPTJConfig(**hf_config, dtype=dtype)
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

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
        model_name=model_name,
        max_window_size=config.max_sequence_length,
        stop_tokens=stop_tokens,
        add_prefix_space=True,
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
        import re

        str_pattern = re.compile(r"(q|k|v)_proj")
        if re.search(str_pattern, pname) is not None:
            return [str_pattern.sub("qkv_proj", pname)]
        else:
            return [pname]

    hidden_size = config.hidden_size

    def f_convert_param_bkwd(torch_pname: str, torch_param) -> Optional[List[Tuple[str, Any]]]:
        # torch_param: numpy.ndarray
        if torch_pname.endswith("qkv_proj.weight"):
            assert torch_param.ndim == 2
            mp_num = 4
            torch_param = torch_param.astype(dtype).reshape(mp_num, 3, -1, hidden_size)
            q_weight = torch_param[:, 0, :, :].reshape(hidden_size, hidden_size)
            k_weight = torch_param[:, 2, :, :].reshape(hidden_size, hidden_size)
            v_weight = torch_param[:, 1, :, :].reshape(hidden_size, hidden_size)
            return [
                (torch_pname.replace("qkv_proj", "q_proj"), q_weight),
                (torch_pname.replace("qkv_proj", "k_proj"), k_weight),
                (torch_pname.replace("qkv_proj", "v_proj"), v_weight),
            ]
        if "ln_1" in torch_pname or "ln_f" in torch_pname:
            return [(torch_pname, torch_param.astype("float32"))]
        else:
            return [(torch_pname, torch_param.astype(dtype))]

    param_manager.set_param_loading_func(
        args.model_path, args.use_safetensors, f_convert_pname_fwd, f_convert_param_bkwd
    )
    return mod, param_manager, [None] * len(param_manager.param_names), config
