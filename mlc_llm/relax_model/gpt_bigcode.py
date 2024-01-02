import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tvm
from tvm import relax, te
from tvm.relax.op import (
    astype,
    broadcast_to,
    expand_dims,
    matmul,
    maximum,
    minimum,
    permute_dims,
    reshape,
    squeeze,
)
from tvm.relax.op.nn import gelu, layer_norm, softmax
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import Embedding, Linear, ModuleList
from .param_manager import ParamManager


@dataclass
class GPTBigCodeConfig:
    def __init__(
        self,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-05,
        max_sequence_length: int = 2048,
        n_embd: int = 6144,
        n_head: int = 48,
        n_inner: int = 24576,
        n_layer: int = 40,
        n_positions: int = 8192,
        scale_attn_weights: bool = True,
        vocab_size: int = 49152,
        dtype: str = "float32",
        **kwargs,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.max_sequence_length = max_sequence_length
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_inner = n_inner
        self.n_layer = n_layer
        self.n_positions = n_positions
        self.scale_attn_weights = scale_attn_weights
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.kwargs = kwargs


def _prepare_decoder_attention_mask(input_shape, src_len, dtype):
    # create causal mask
    # [bsz, seq_len] -> [bsz, tgt_seq_len, 1, src_seq_len]
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
        mask = nn.emit(expand_dims(mask, 1))
        diag_mask = nn.emit(broadcast_to(mask, (bsz, tgt_len, 1, tgt_len)))
        if src_len == tgt_len:
            return diag_mask

        def extend_te(x, tgt_len, src_len):
            return te.compute(
                (bsz, tgt_len, 1, src_len),
                lambda b, i, _, j: te.if_then_else(
                    j < src_len - tgt_len,
                    tvm.tir.max_value(dtype),
                    x[b, i, _, j - (src_len - tgt_len)],
                ),
                name="concat_te",
            )

        return nn.emit_te(extend_te, diag_mask, tgt_len, src_len)
    else:
        # Get src_len from input parameters
        # [bsz, seq_len] -> [bsz, tgt_seq_len, 1, src_seq_len]
        bsz, tgt_len = input_shape
        mask = relax.op.full(
            (bsz, tgt_len, 1, src_len),
            relax.const(tvm.tir.max_value(dtype).value, dtype),
            dtype,
        )
    return nn.emit(mask)


def apply_position_embedding(t_embd, weight, offset: int = 0):
    def f_position_embedding(tensor, weight, offset):
        def position_compute(*idx):
            b, s, e = idx
            return weight[s + offset, e] + tensor[b, s, e]

        return tvm.te.compute(tensor.shape, position_compute, name="position")

    hidden_states = nn.emit_te(
        f_position_embedding,
        t_embd,
        weight,
        offset,
        primfunc_name_hint="position_embedding",
    )
    return hidden_states


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        dtype,
        eps=1e-5,
    ):
        super().__init__()
        self.dtype = dtype

        self.eps = eps
        self.weight = nn.Parameter((hidden_size,), dtype=dtype, name="weight")
        self.bias = nn.Parameter((hidden_size,), dtype=dtype, name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
        if x.struct_info.dtype != self.dtype:
            x = nn.emit(relax.op.astype(x, self.dtype))
        x = nn.emit(
            layer_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )
        return x


class GPTBigCodeAttention(nn.Module):
    """Multi-query attention from 'Fast Transformer Decoding: One Write-Head is All You Need'"""

    def __init__(self, config: GPTBigCodeConfig):
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"hidden_size must be divisible by n_head (got `hidden_size`: {config.n_embd}"
                f" and `n_head`: {config.n_head})."
            )
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = Linear(self.n_embd, self.n_embd + 2 * self.head_dim, config.dtype, bias=True)
        self.c_proj = Linear(self.n_embd, self.n_embd, config.dtype, bias=True)

        self.dtype = config.dtype

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr, relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Union[Tuple[None, None], Tuple[relax.Expr, relax.Expr]]]:
        # hidden_states: [batch_size, seq_len, n_embd]
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

        batch_size, seq_len, _ = hidden_states.struct_info.shape
        kv_seq_len = all_seq_len_shape.struct_info.values[0]

        def te_slice(x: te.Tensor, start: int, end: int):
            batch_size, seq_len, _ = x.shape
            return te.compute(
                shape=(batch_size, seq_len, end - start),
                fcompute=lambda i, j, k: x[i, j, start + k],
                name="slice",
            )

        query_key_value = self.c_attn(hidden_states)
        # queries: [batch_size, seq_len, n_embd]
        q = nn.emit_te(te_slice, query_key_value, 0, self.n_embd, primfunc_name_hint="slice")
        # keys: [batch_size, seq_len, head_dim]
        k = nn.emit_te(
            te_slice,
            query_key_value,
            self.n_embd,
            self.n_embd + self.head_dim,
            primfunc_name_hint="slice",
        )
        # values: [batch_size, seq_len, head_dim]
        v = nn.emit_te(
            te_slice,
            query_key_value,
            self.n_embd + self.head_dim,
            self.n_embd + 2 * self.head_dim,
            primfunc_name_hint="slice",
        )

        squeezed_k = nn.emit(squeeze(k, axis=0))
        squeezed_v = nn.emit(squeeze(v, axis=0))

        assert k.struct_info.shape[0] == 1 and v.struct_info.shape[0] == 1

        k_cache, v_cache = past_key_value
        f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[k_cache, squeezed_k],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[v_cache, squeezed_v],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        past_key_value = (k_cache, v_cache)

        batch_size, _, head_size = k.struct_info.shape
        kv_cache_shape = R.shape([kv_seq_len, head_size])
        kv_states_shape = R.shape([batch_size, kv_seq_len, head_size])
        f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
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

        q_state_shape = R.shape([batch_size, seq_len * self.n_head, self.head_dim])
        q = nn.emit(reshape(q, q_state_shape))

        # Calculate Q.K
        attn_weights = nn.emit(
            matmul(q, permute_dims(k, [0, 2, 1]))
            / relax.const(math.sqrt(self.head_dim), q.struct_info.dtype)
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
        attn_shape = R.shape([batch_size, seq_len, self.n_head, kv_seq_len])
        attn_view = R.shape([batch_size, seq_len * self.n_head, kv_seq_len])
        attn_weights = nn.emit(reshape(attn_weights, attn_shape))
        attn_weights = nn.emit(minimum(attn_weights, attention_mask))
        attn_weights = nn.emit(reshape(attn_weights, attn_view))

        # Calculate Softmax(Q.K)
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != q.struct_info.dtype:
            attn_weights = astype(attn_weights, q.struct_info.dtype)

        # Calculate Softmax(Q.K).V
        attn_output = nn.emit(matmul(attn_weights, v))

        # Apply output projection
        attn_output = self.c_proj(
            reshape(
                attn_output,
                (batch_size, seq_len, self.n_embd),
            )
        )

        return attn_output, past_key_value


class GPTBigCodeMLP(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        super().__init__()
        self.dtype = config.dtype

        self.c_fc = Linear(config.n_embd, config.n_inner, config.dtype, bias=True)
        self.c_proj = Linear(config.n_inner, config.n_embd, config.dtype, bias=True)

    def forward(self, hidden_states):
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

        hidden_states = self.c_fc(hidden_states)
        hidden_states = nn.emit(gelu(hidden_states))
        hidden_states = self.c_proj(hidden_states)

        return hidden_states


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        self.dtype = config.dtype

        self.ln_1 = LayerNorm(
            hidden_size=config.n_embd, dtype=config.dtype, eps=config.layer_norm_epsilon
        )
        self.ln_2 = LayerNorm(
            hidden_size=config.n_embd, dtype=config.dtype, eps=config.layer_norm_epsilon
        )

        self.attn = GPTBigCodeAttention(config)
        self.mlp = GPTBigCodeMLP(config)

    def forward(
        self,
        hidden_states,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: Optional[relax.Expr] = None,
    ):
        attn_input = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attn(
            attn_input, all_seq_len_shape, past_key_value, attention_mask
        )

        # residual connection
        attn_output = nn.emit(attn_output + hidden_states)

        mlp_input = self.ln_2(attn_output)
        mlp_output = self.mlp(mlp_input)

        # residual connection
        hidden_states = nn.emit(astype(mlp_output, self.dtype) + attn_output)

        return hidden_states, present_key_value


class GPTBigCodeModel(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        self.wte = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
            dtype=config.dtype,
        )
        self.wpe = Embedding(
            num_embeddings=config.n_positions,
            embedding_dim=config.n_embd,
            dtype=config.dtype,
        )

        self.h = ModuleList([GPTBigCodeBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(
            hidden_size=config.n_embd, dtype=config.dtype, eps=config.layer_norm_epsilon
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        batch_size, seq_length = input_ids.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]

        # Token Embeddings
        t_embd = self.wte(input_ids)

        # Position Embeddings
        offset = seq_length_with_past - seq_length
        hidden_states = apply_position_embedding(t_embd, self.wpe.weight, offset=offset)

        attention_mask = _prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            dtype=hidden_states.struct_info.dtype,
        )

        present_kv_cache = []
        for i, block in enumerate(self.h):
            past_key_value = (
                (past_key_values[i * 2], past_key_values[i * 2 + 1])
                if past_key_values is not None
                else None
            )
            hidden_states, (present_k_cache, present_v_cache) = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
            )
            present_kv_cache.append(present_k_cache)
            present_kv_cache.append(present_v_cache)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, present_kv_cache


class GPTBigCodeForCausalLM(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        self.dtype = config.dtype

        self.transformer = GPTBigCodeModel(config)
        self.lm_head = Linear(
            in_features=config.n_embd,
            out_features=config.vocab_size,
            bias=False,
            dtype=config.dtype,
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        hidden_states, key_value_cache = self.transformer(
            input_ids=input_ids,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def te_slice_last(x: te.Tensor):
            _, seq_len, n_embd = x.shape
            return te.compute(
                shape=(1, 1, n_embd),
                fcompute=lambda i, _, k: x[i, seq_len - 1, k],
                name="slice_last",
            )

        hidden_states = nn.emit_te(
            te_slice_last,
            hidden_states,
            primfunc_name_hint="slice_last",
        )
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

        logits = self.lm_head(hidden_states)

        if logits.struct_info.dtype != "float32":
            logits = nn.emit(astype(logits, "float32"))

        return logits, key_value_cache


def get_param_quant_kind(name: str, param_info: relax.TensorStructInfo) -> ParamQuantKind:
    if "wte.weight" in name:
        return ParamQuantKind.embedding_table
    elif "lm_head.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif "wpe" not in name and param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPTBigCodeConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "prefill"

    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.SizeVar("n", "int64")
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = GPTBigCodeForCausalLM(config)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((batch_size, seq_len), dtype="int32", name="input_ids")
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo([relax.ObjectStructInfo() for _ in range(config.n_layer * 2)]),
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
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPTBigCodeConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    bsz = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.IntImm("int64", 1)
    all_seq_len = tvm.tir.SizeVar("m", "int64")

    with bb.function(func_name):
        model = GPTBigCodeForCausalLM(config)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo([relax.ObjectStructInfo() for _ in range(config.n_layer * 2)]),
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
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_kv_cache_func(bb: relax.BlockBuilder, config: GPTBigCodeConfig) -> None:
    init_shape = relax.ShapeExpr(
        (
            config.max_sequence_length,
            config.n_embd // config.n_head,
        )
    )
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            for _ in range(config.n_layer * 2):
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


def create_softmax_func(bb: relax.BlockBuilder, config: GPTBigCodeConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder((1, 1, config.vocab_size), dtype="float32", name="logits")
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def get_model(args: argparse.Namespace, hf_config):
    model = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len

    if (
        model.startswith("starcoder")
        or model.startswith("WizardCoder-")
        or model.startswith("gpt_bigcode")
    ):
        config = GPTBigCodeConfig(
            **hf_config,
            dtype=dtype,
        )
        if max_seq_len != -1:
            config.max_sequence_length = max_seq_len
        elif config.max_sequence_length is None:
            config.max_sequence_length = 2048

        param_manager = ParamManager()
        bb = relax.BlockBuilder()
        create_encoding_func(bb, param_manager, config, args.quantization)
        create_decoding_func(bb, param_manager, config, args.quantization)
        create_kv_cache_func(bb, config)
        create_softmax_func(bb, config)
        create_metadata_func(
            bb,
            model_name=model,
            max_window_size=config.max_sequence_length,
            stop_tokens=[0],
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

        param_manager.set_param_loading_func(
            args.model_path,
            args.use_safetensors,
            f_convert_param_bkwd=lambda torch_pname, torch_param: [
                (torch_pname, torch_param.astype(dtype))
            ],
        )
        return mod, param_manager, [None] * len(param_manager.param_names), config

    raise ValueError(f"Unsupported model {model}")
