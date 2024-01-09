import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.op import ccl
from tvm.relax.op.nn import layer_norm
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .llama import Embedding, Linear
from .modules import ModuleList, RotaryEmbedding
from .param_manager import ParamManager


@dataclass
class StableLM3bConfig:
    def __init__(
        self,
        dtype="float32",
        max_sequence_length=4096,
        vocab_size=50304,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        norm_eps=1e-5,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        position_embedding_base=10000,
        combine_matmul=True,
        num_shards=1,
        build_model_only=False,
        convert_weights_only=False,
        **kwargs,
    ):
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.position_embedding_base = position_embedding_base
        self.combine_matmul = combine_matmul
        if build_model_only and num_shards > 1:
            self.num_shards = num_shards
        else:
            self.num_shards = 1
        self.kwargs = kwargs

    def get_num_key_value_heads(self):
        if self.num_key_value_heads is None:
            return self.num_attention_heads
        return self.num_key_value_heads


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        dtype,
        eps=1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter((hidden_size,), dtype="float16", name="weight")
        self.bias = nn.Parameter((hidden_size,), dtype="float16", name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
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


class StableLM3bMLP(nn.Module):
    def __init__(self, config: StableLM3bConfig):
        self.combine_matmul = config.combine_matmul
        self.num_shards = config.num_shards
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size // self.num_shards
        dtype = config.dtype
        if self.combine_matmul:
            self.gate_up_proj = Linear(hidden_size, 2 * intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
            self.gate_up_proj.weight.shard_dim = 0
            self.down_proj.weight.shard_dim = 1
        else:
            self.gate_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
            self.up_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)
            self.gate_proj.weight.shard_dim = 0
            self.up_proj.weight.shard_dim = 0
            self.down_proj.weight.shard_dim = 1

    def forward(self, x):
        if self.combine_matmul:
            gate_up_results = nn.emit(
                relax.op.split(
                    self.gate_up_proj(x),
                    indices_or_sections=2,
                    axis=-1,
                )
            )
            gate_result = relax.TupleGetItem(gate_up_results, 0)
            up_result = relax.TupleGetItem(gate_up_results, 1)
        else:
            gate_result = self.gate_proj(x)
            up_result = self.up_proj(x)

        result = self.down_proj(relax.op.nn.silu(gate_result) * up_result)
        return result


class StableLM3bAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: StableLM3bConfig, rotary_embedding: RotaryEmbedding):
        dtype = config.dtype
        self.num_shards = config.num_shards
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = (
            config.num_key_value_heads is None
            and config.num_attention_heads
            or config.num_key_value_heads
        ) // config.num_shards
        self.num_query_heads = config.num_attention_heads // self.num_shards
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.position_embedding_base = config.position_embedding_base
        self.rotary_embedding = rotary_embedding

        self.combine_matmul = config.combine_matmul
        if self.combine_matmul:
            self.query_key_value_proj = Linear(
                self.hidden_size,
                (self.num_query_heads + 2 * self.num_key_value_heads) * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.query_key_value_proj.weight.shard_dim = 0
        else:
            self.q_proj = Linear(
                self.hidden_size,
                self.num_query_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.k_proj = Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.v_proj = Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.q_proj.weight.shard_dim = 0
            self.k_proj.weight.shard_dim = 0
            self.v_proj.weight.shard_dim = 0

        self.o_proj = Linear(
            self.head_dim * self.num_query_heads, self.hidden_size, dtype=dtype, bias=False
        )
        self.o_proj.weight.shard_dim = 1

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[relax.Expr], Optional[Tuple[relax.Expr]]]:
        from tvm.relax.op import (
            astype,
            matmul,
            maximum,
            permute_dims,
            reshape,
            split,
            squeeze,
        )
        from tvm.relax.op.nn import softmax

        bsz, q_len, _ = hidden_states.struct_info.shape
        assert bsz == 1, "Only support batch size 1 at this moment."

        if self.combine_matmul:
            qkv_states = nn.emit(
                split(
                    self.query_key_value_proj(hidden_states),
                    indices_or_sections=[
                        self.num_query_heads * self.head_dim,
                        (self.num_query_heads + self.num_key_value_heads) * self.head_dim,
                    ],
                    axis=-1,
                )
            )
            query_states = relax.TupleGetItem(qkv_states, 0)
            key_states = relax.TupleGetItem(qkv_states, 1)
            value_states = relax.TupleGetItem(qkv_states, 2)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = nn.emit(
            reshape(
                query_states,
                (bsz, q_len, self.num_query_heads, self.head_dim),
            ),
        )
        key_states = nn.emit(
            reshape(
                key_states,
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
            ),
        )
        value_states = nn.emit(
            reshape(
                value_states,
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
            ),
        )

        kv_seq_len = all_seq_len_shape.struct_info.values[0]
        offset = kv_seq_len - q_len
        query_states, key_states = self.rotary_embedding(query_states, key_states, offset)
        # [bsz, t, nh, hd]

        kv_states_shape = key_states.struct_info.shape
        kv_states_dtype = key_states.struct_info.dtype
        assert kv_states_shape[0] == 1  # bsz
        kv_states_shape = R.shape(
            [kv_states_shape[0], kv_seq_len, kv_states_shape[2], kv_states_shape[3]]
        )
        kv_cache_shape = R.shape([kv_seq_len, kv_states_shape[2], kv_states_shape[3]])

        squeezed_key = nn.emit(squeeze(key_states, axis=0))
        squeezed_value = nn.emit(squeeze(value_states, axis=0))
        k_cache, v_cache = past_key_value
        f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[k_cache, squeezed_key],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[v_cache, squeezed_value],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        past_key_value = (k_cache, v_cache)
        f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[k_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_states_dtype)],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[v_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_states_dtype)],
            )
        )
        key_states = nn.emit(reshape(k_cache, kv_states_shape))
        value_states = nn.emit(reshape(v_cache, kv_states_shape))
        if self.num_key_value_heads != self.num_query_heads:
            n_rep = self.num_query_heads // self.num_key_value_heads
            key_states = nn.emit(relax.op.repeat(key_states, n_rep, axis=2))
            value_states = nn.emit(relax.op.repeat(value_states, n_rep, axis=2))

        query_states = nn.emit(permute_dims(query_states, [0, 2, 1, 3]))
        key_states = nn.emit(permute_dims(key_states, [0, 2, 1, 3]))
        value_states = nn.emit(permute_dims(value_states, [0, 2, 1, 3]))

        attn_weights = nn.emit(
            matmul(query_states, permute_dims(key_states, [0, 1, 3, 2]))
            / relax.const(math.sqrt(self.head_dim), query_states.struct_info.dtype)
        )

        tvm.ir.assert_structural_equal(
            attention_mask.struct_info.shape.values,
            (bsz, tvm.tir.IntImm("int64", 1), q_len, kv_seq_len),
        )

        attn_weights = nn.emit(
            maximum(
                attn_weights,
                relax.const(
                    tvm.tir.min_value(attn_weights.struct_info.dtype).value,
                    attn_weights.struct_info.dtype,
                ),
            )
        )
        attn_weights = nn.emit(relax.op.minimum(attn_weights, attention_mask))

        # upcast attention to fp32
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != query_states.struct_info.dtype:
            attn_weights = astype(attn_weights, query_states.struct_info.dtype)
        attn_output = nn.emit(matmul(attn_weights, value_states))

        attn_output = nn.emit(permute_dims(attn_output, [0, 2, 1, 3]))
        attn_output = nn.emit(
            reshape(attn_output, (bsz, q_len, self.head_dim * self.num_query_heads))
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, ((None, None) if past_key_value is None else past_key_value)


class StableLM3bDecoderLayer(nn.Module):
    def __init__(self, config: StableLM3bConfig, rotary_embedding: RotaryEmbedding):
        self.hidden_size = config.hidden_size
        self.self_attn = StableLM3bAttention(config, rotary_embedding)
        self.mlp = StableLM3bMLP(config)
        self.input_layernorm = LayerNorm(
            config.hidden_size, dtype=config.dtype, eps=config.norm_eps
        )
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size, dtype=config.dtype, eps=config.norm_eps
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            all_seq_len_shape=all_seq_len_shape,
        )
        if self.self_attn.num_shards > 1:
            residual = nn.emit(
                residual / R.const(self.self_attn.num_shards, dtype=residual.struct_info.dtype)
            )
        hidden_states = nn.emit(residual + hidden_states)
        if self.self_attn.num_shards > 1:
            hidden_states = nn.emit(ccl.allreduce(hidden_states, "sum"))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp.num_shards > 1:
            residual = nn.emit(
                residual / R.const(self.mlp.num_shards, dtype=residual.struct_info.dtype)
            )
        hidden_states = nn.emit(residual + hidden_states)
        if self.mlp.num_shards > 1:
            hidden_states = nn.emit(ccl.allreduce(hidden_states, "sum"))
        return hidden_states, present_key_value


def _make_causal_mask(input_ids_shape, dtype, src_len):
    from tvm.relax.op import broadcast_to

    bsz, tgt_len = input_ids_shape

    def min_max_triu_te():
        return te.compute(
            (tgt_len, tgt_len),
            lambda i, j: tvm.tir.Select(j > i, tvm.tir.min_value(dtype), tvm.tir.max_value(dtype)),
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


class StableLM3bEmbedTokens(nn.Module):
    def __init__(self, config: StableLM3bConfig, vocab_size_var: tvm.tir.SizeVar):
        self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class StableLM3bEmbedTokensWrapper(nn.Module):
    def __init__(self, config: StableLM3bConfig, vocab_size_var: tvm.tir.SizeVar):
        # build a wrapper to ensure that the naming of the embed_tokens parameter is consistent
        self.model = StableLM3bEmbedTokens(config, vocab_size_var)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.model(input_ids)
        return inputs_embeds


class StableLM3bModell(nn.Module):
    def __init__(
        self, config: StableLM3bConfig, vocab_size_var: tvm.tir.SizeVar, sep_embed: bool = False
    ):
        rotary_embedding = RotaryEmbedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            position_embedding_base=config.position_embedding_base,
            max_sequence_length=config.max_sequence_length,
            rotary_pct=0.25,
            dtype=config.dtype,
        )
        self.num_shards = config.num_shards
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [
                StableLM3bDecoderLayer(config, rotary_embedding)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LayerNorm(config.hidden_size, dtype=config.dtype, eps=config.norm_eps)

    def _prepare_decoder_attention_mask(self, input_shape, src_len, dtype):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if isinstance(input_shape[-1], tvm.tir.SizeVar) or input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, dtype, src_len)
        else:
            # Get src_len from input parameters
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            bsz, tgt_len = input_shape
            combined_attention_mask = nn.emit(
                relax.op.full(
                    (bsz, 1, tgt_len, src_len),
                    relax.const(tvm.tir.max_value(dtype).value, dtype),
                    dtype,
                )
            )
        return combined_attention_mask

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        if self.num_shards > 1:
            inputs = nn.emit(ccl.broadcast_from_worker0(inputs))
        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(inputs)
        else:
            inputs_embeds = inputs
        # retrieve input_ids
        batch_size, seq_length, _ = inputs_embeds.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]
        # embed positions
        attention_mask = self._prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            inputs_embeds.struct_info.dtype,
        )

        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = ()

        for idx, decoder_layer in enumerate(self.layers):
            assert past_key_values is not None
            past_key_value = (past_key_values[idx * 2], past_key_values[idx * 2 + 1])

            hidden_states, key_value_cache = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
            )
            next_decoder_cache += key_value_cache

        hidden_states = self.norm(hidden_states)

        assert len(next_decoder_cache) == len(self.layers) * 2
        return hidden_states, next_decoder_cache


class StableLM3bForCausalLM(nn.Module):
    def __init__(
        self, config: StableLM3bConfig, vocab_size_var: tvm.tir.SizeVar, sep_embed: bool = False
    ):
        self.model = StableLM3bModell(config, vocab_size_var, sep_embed)
        self.lm_head = Linear(config.hidden_size, vocab_size_var, dtype=config.dtype, bias=False)

        assert config.hidden_size % config.num_attention_heads == 0

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        hidden_states, key_value_cache = self.model(
            inputs=inputs,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def te_slicing(x: te.Tensor):
            return te.compute(
                shape=(1, 1, x.shape[-1]),
                fcompute=lambda i, j, k: x[i, x.shape[1] - 1, k],
                name="slice",
            )

        logits = self.lm_head(nn.emit_te(te_slicing, hidden_states, primfunc_name_hint="slice"))
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, key_value_cache


def get_param_quant_kind(name: str, param_info: relax.TensorStructInfo) -> ParamQuantKind:
    if "embed_tokens" in name:
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
    config: StableLM3bConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    bsz = 1
    seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = StableLM3bEmbedTokensWrapper(config, tvm.tir.SizeVar("vocab_size", "int64"))
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        with bb.dataflow():
            inputs_embeds = model(input_ids)
            params = [input_ids] + model.parameters()
            gv = bb.emit_output(inputs_embeds)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 1))


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: StableLM3bConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    bsz = 1
    seq_len = tvm.tir.SizeVar("n", "int64")
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = StableLM3bForCausalLM(config, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = (
            nn.Placeholder((bsz, seq_len, hidden_size), dtype=config.dtype, name="inputs_embeds")
            if sep_embed
            else nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
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
                inputs, all_seq_len_shape, past_key_values=past_key_values
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
    config: StableLM3bConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    bsz = 1
    all_seq_len = tvm.tir.SizeVar("m", "int64")

    with bb.function(func_name):
        model = StableLM3bForCausalLM(config, tvm.tir.SizeVar("vocab_size", "int64"))
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids, all_seq_len_shape, past_key_values=past_key_values
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


def create_kv_cache_func(bb: relax.BlockBuilder, config: StableLM3bConfig) -> None:
    num_key_value_heads = (
        config.num_attention_heads
        if config.num_key_value_heads is None
        else config.num_key_value_heads
    ) // config.num_shards
    init_shape = relax.ShapeExpr(
        (
            config.max_sequence_length,
            num_key_value_heads,
            config.hidden_size // config.num_attention_heads,  # head_dim
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


def create_softmax_func(bb: relax.BlockBuilder, config: StableLM3bConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder(
            (1, 1, tvm.tir.SizeVar("vocab_size", "int64")), dtype="float32", name="logits"
        )
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def emit_shard3d(bb: relax.BlockBuilder) -> None:
    from tvm.script import tir as T

    def _emit(dtype: str, global_symbol: str):
        @T.prim_func
        def shard_3d(a: T.handle, num_shards: T.int64, b: T.handle):
            T.func_attr(
                {
                    "tir.noalias": T.bool(True),
                    "global_symbol": global_symbol,
                }
            )
            s_0, s_1, s_2 = T.int64(), T.int64(), T.int64()
            # pylint: disable=invalid-name
            A = T.match_buffer(a, (s_0, s_1, s_2), dtype)
            B = T.match_buffer(b, (num_shards, s_0, s_1 // num_shards, s_2), dtype)
            # pylint: enable=invalid-name
            for j_o, i, j_i, k in T.grid(num_shards, s_0, s_1 // num_shards, s_2):
                with T.block("B"):
                    v_j_o = T.axis.spatial(num_shards, j_o)
                    v_i = T.axis.spatial(s_0, i)
                    v_j_i = T.axis.spatial(s_1 // num_shards, j_i)
                    v_k = T.axis.spatial(s_2, k)
                    B[v_j_o, v_i, v_j_i, v_k] = A[v_i, v_j_o * (s_1 // num_shards) + v_j_i, v_k]

        bb.add_func(shard_3d, global_symbol)

    _emit("float32", "shard3d_fp32")
    _emit("float16", "shard3d_fp16")
    _emit("uint32", "shard3d_uint32")


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    sep_embed = args.sep_embed

    position_embedding_base = 10000
    if "rope_theta" in hf_config:
        position_embedding_base = hf_config["rope_theta"]

    config = StableLM3bConfig(
        **hf_config,
        dtype=dtype,
        position_embedding_base=position_embedding_base,
        combine_matmul=True,
        num_shards=args.num_shards,
        build_model_only=args.build_model_only,
        convert_weights_only=args.convert_weights_only,
    )
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    emit_shard3d(bb)

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
        stop_tokens=[2],
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
        if not config.combine_matmul:
            return [pname]

        qkv_str = "query_key_value_proj"
        gate_up_str = "gate_up_proj"
        if qkv_str in pname:
            return [
                pname.replace(qkv_str, "q_proj"),
                pname.replace(qkv_str, "k_proj"),
                pname.replace(qkv_str, "v_proj"),
            ]
        elif gate_up_str in pname:
            return [
                pname.replace(gate_up_str, "gate_proj"),
                pname.replace(gate_up_str, "up_proj"),
            ]
        else:
            return [pname]

    def f_convert_param_bkwd(torch_pname: str, torch_param):
        if not config.combine_matmul:
            return [(torch_pname, torch_param.astype(dtype))]

        combined_layers = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
        if any([name in torch_pname for name in combined_layers]):
            return None
        return [(torch_pname, torch_param.astype(dtype))]

    def f_compute_relax_param(relax_pname: str, torch_params: List[Any]):
        # Expected to enter this function only for the combined linear matmul weights.
        # Other weights are supposed to be loaded in `f_convert_param_bkwd` since
        # each other relax param has a unique corresponding torch param.
        if not config.combine_matmul:
            # When matmul combination is not turned on, each relax param has a unique
            # corresponding torch param, and this function is not expected to be entered.
            raise NotImplementedError(
                "Matmul combination is not turned on, and the function "
                "is not expected to be entered"
            )
        num_shards = args.num_shards
        hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads

        if "query_key_value_proj" in relax_pname:
            q_heads = config.num_attention_heads
            kv_heads = config.num_key_value_heads
            if kv_heads is None:
                kv_heads = q_heads
            q, k, v = torch_params
            assert q.shape == (q_heads * head_dim, hidden_size)
            assert k.shape == (kv_heads * head_dim, hidden_size)
            assert v.shape == (kv_heads * head_dim, hidden_size)
            q = q.reshape((num_shards, q_heads // num_shards, head_dim, hidden_size))
            k = k.reshape((num_shards, kv_heads // num_shards, head_dim, hidden_size))
            v = v.reshape((num_shards, kv_heads // num_shards, head_dim, hidden_size))
            qkv = np.concatenate([q, k, v], axis=1)
            qkv = qkv.reshape((-1, hidden_size)).astype(dtype)
            return qkv
        if "gate_up_proj" in relax_pname:
            intermediate_size = config.intermediate_size
            gate, up = torch_params
            gate = gate.reshape((num_shards, intermediate_size // num_shards, hidden_size))
            up = up.reshape((num_shards, intermediate_size // num_shards, hidden_size))
            gate_up = np.concatenate([gate, up], axis=1)
            gate_up = gate_up.reshape((-1, hidden_size)).astype(dtype)
            return gate_up
        raise ValueError("Unexpected param loading")

    param_manager.set_param_loading_func(
        args.model_path,
        args.use_safetensors,
        f_convert_pname_fwd,
        f_convert_param_bkwd,
        f_compute_relax_param,
    )

    param_list = [None] * param_manager.nparam_to_load

    return mod, param_manager, param_list, config
