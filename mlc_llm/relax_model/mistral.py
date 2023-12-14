# pylint: disable=too-many-lines, missing-class-docstring, missing-function-docstring
"""Implements the mistal model with sliding window attention."""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.op import ccl
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList
from .param_manager import ParamManager


@dataclass
class MistralConfig:
    """Configuration for mistral model."""

    def __init__(
        self,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=-1,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=14336,
        max_position_embeddings=32768,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_sink_size=0,
        tie_word_embeddings=False,
        vocab_size=32000,
        dtype="float32",
        max_sequence_length=16384,
        combine_matmul=True,
        build_model_only=False,
        num_shards=1,
        **kwargs,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_sink_size = attention_sink_size
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.max_sequence_length = sliding_window * 4
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


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dtype: str, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((out_features, in_features), dtype=dtype, name="linear_weight")
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype, name="linear_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(input, self.weight, self.bias))


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype: str):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype=dtype, name="embedding_weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        from tvm.relax.op import (  # pylint: disable=import-outside-toplevel
            reshape,
            take,
        )

        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(take(self.weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = self.weight.struct_info.shape.values[-1]
            x = nn.emit(reshape(x, shape=[-1]))
            embedding = nn.emit(take(self.weight, x, axis=0))
            return nn.emit(reshape(embedding, [*x_shape, emb_size]))


class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, eps=1e-6):
        self.weight = nn.Parameter((hidden_size,), dtype=dtype, name="rms_norm_weight")
        self.variance_epsilon = tvm.tir.const(eps, dtype)

    def forward(self, hidden_states):
        from tvm import te, tir

        def f_rms_norm(x, weight):
            is_float32 = x.dtype == "float32"

            def f_square(x):
                return tir.Cast("float32", x) * tir.Cast("float32", x) if not is_float32 else x * x

            k = te.reduce_axis((0, x.shape[2]), name="k")
            square_sum = te.compute(
                (x.shape[0], x.shape[1]),
                lambda bsz, i: te.sum(f_square(x[bsz, i, k]), axis=k),
                name=x.op.name + "red_temp",
            )

            def f_div_cast(bsz, i, k):
                x_val = x[bsz, i, k]
                if not is_float32:
                    x_val = tir.Cast("float32", x_val)
                return x_val / tir.sqrt(square_sum[bsz, i] / x.shape[2] + self.variance_epsilon)

            def f_mul_cast(x, y):
                value = x * y
                if not is_float32:
                    value = tir.Cast(x.dtype, value)
                return value

            return te.compute(
                x.shape,
                lambda bsz, i, k: f_mul_cast(weight(k), f_div_cast(bsz, i, k)),
                name="rms_norm",
            )

        return nn.emit_te(f_rms_norm, hidden_states, self.weight, primfunc_name_hint="rms_norm")


class MistralMLP(nn.Module):
    def __init__(self, config: MistralConfig):
        self.combine_matmul = config.combine_matmul
        self.num_shards = config.num_shards
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size // self.num_shards
        dtype = config.dtype
        if self.combine_matmul:
            self.gate_up_proj = Linear(hidden_size, 2 * intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
            self.gate_up_proj.weight.shard_dim = 0
            self.gate_up_proj.weight.shard_strategy = "shard_gate_up"
            self.down_proj.weight.shard_dim = 1
            self.down_proj.weight.shard_strategy = "shard_mlp_k"
        else:
            self.gate_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
            self.up_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)

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


def apply_rotary_pos_emb(q, k, base, q_offset):
    def f_rotary_embedding(tensor, offset):
        dtype = tensor.dtype
        head_dim = tensor.shape[-1]
        n_feat_half = tensor.shape[-1] // 2

        def rotary_compute(*idx):
            i, j = idx[-3], idx[-1]
            pos = (offset + i).astype("float32")
            inv_freq = te.const(1, "float32") / (
                te.power(
                    te.const(base, "float32"),
                    ((2 * j) % head_dim).astype("float32") / head_dim.astype("float32"),
                )
            )
            freq = pos * inv_freq
            return te.cos(freq).astype(dtype) * tensor(*idx) + te.sin(freq).astype(
                dtype
            ) * tvm.tir.Select(
                j >= n_feat_half,
                tensor[idx[0], i, idx[2], j - n_feat_half],
                -tensor[idx[0], i, idx[2], j + n_feat_half],
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(f_rotary_embedding, q, q_offset, primfunc_name_hint="rotary_embedding")
    k_embed = nn.emit_te(f_rotary_embedding, k, 0, primfunc_name_hint="rotary_embedding")
    return q_embed, k_embed


class MistralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig):
        dtype = config.dtype
        self.num_shards = config.num_shards
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
        self.num_query_heads = config.num_attention_heads // self.num_shards
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.sliding_window = config.sliding_window
        self.attention_sink_size = config.attention_sink_size

        self.combine_matmul = config.combine_matmul
        if self.combine_matmul:
            self.query_key_value_proj = Linear(
                self.hidden_size,
                (self.num_query_heads + 2 * self.num_key_value_heads) * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.query_key_value_proj.weight.shard_dim = 0
            self.query_key_value_proj.weight.shard_strategy = "shard_qkv"
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
        self.o_proj.weight.shard_strategy = "shard_o_proj_k"

    def interleave_kv(
        self,
        key_cur: relax.Expr,
        value_cur: relax.Expr,
        kv_seq_len: int,
        rolling_cache_len: int,
        cache_offset: int,
        attention_sink_size: int,
        past_key_value: Tuple[relax.Expr],
    ):
        from tvm.relax.op import reshape

        def te_cache_unrotate(x_cached, cache_offset, rolling_cache_len):
            return te.compute(
                (kv_cur_shape[0], rolling_cache_len, kv_cur_shape[2], kv_cur_shape[3]),
                lambda b, s, h, d: te.if_then_else(
                    s < attention_sink_size,
                    x_cached[b, s, h, d],
                    te.if_then_else(
                        s < rolling_cache_len - cache_offset + attention_sink_size,
                        x_cached[b, s + cache_offset - attention_sink_size, h, d],
                        x_cached[b, s + cache_offset - rolling_cache_len, h, d],
                    ),
                ),
                name="te_cache_unrotate",
            )

        def te_cache_cur_concat(x, x_cached, kv_seq_len, rolling_cache_len):
            return te.compute(
                (kv_cur_shape[0], kv_seq_len, kv_cur_shape[2], kv_cur_shape[3]),
                lambda b, s, h, d: te.if_then_else(
                    s < rolling_cache_len,
                    x_cached[b, s, h, d],
                    x[b, s - rolling_cache_len, h, d],
                ),
                name="te_cache_cur_concat",
            )

        def te_squeeze(x):
            return te.compute(
                x.shape[1:],
                lambda s, h, d: x[0, s, h, d],
                name="squeeze_te",
            )

        # [bsz, t, nh, hd]
        kv_cur_shape = key_cur.struct_info.shape
        kv_cur_dtype = key_cur.struct_info.dtype
        assert kv_cur_shape[0] == 1  # bsz
        kv_batched_cache_shape = R.shape(
            [kv_cur_shape[0], rolling_cache_len, kv_cur_shape[2], kv_cur_shape[3]]
        )
        kv_cache_shape = R.shape([rolling_cache_len, kv_cur_shape[2], kv_cur_shape[3]])

        # fecth past keys and values from cache
        k_cache, v_cache = past_key_value

        f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
        key_cached = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[k_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_cur_dtype)],
            )
        )
        value_cached = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[v_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_cur_dtype)],
            )
        )
        key_cached = nn.emit(reshape(key_cached, kv_batched_cache_shape))
        value_cached = nn.emit(reshape(value_cached, kv_batched_cache_shape))

        key_cached = nn.emit_te(
            te_cache_unrotate,
            key_cached,
            cache_offset,
            rolling_cache_len,
            primfunc_name_hint="te_cache_unrotate_key",
        )
        key = nn.emit_te(
            te_cache_cur_concat,
            key_cur,
            key_cached,
            kv_seq_len,
            rolling_cache_len,
            primfunc_name_hint="te_cache_cur_concat_key",
        )

        value_cached = nn.emit_te(
            te_cache_unrotate,
            value_cached,
            cache_offset,
            rolling_cache_len,
            primfunc_name_hint="te_cache_unrotate_value",
        )
        value = nn.emit_te(
            te_cache_cur_concat,
            value_cur,
            value_cached,
            kv_seq_len,
            rolling_cache_len,
            primfunc_name_hint="te_cache_cur_concat_value",
        )

        # update cache
        squeezed_key = nn.emit_te(te_squeeze, key_cur)
        squeezed_value = nn.emit_te(te_squeeze, value_cur)

        assert attention_sink_size >= 0
        f_kv_cache_override = relax.extern(
            "vm.builtin.attention_kv_cache_window_override_with_sinks"
        )
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_override,
                args=[
                    k_cache,
                    squeezed_key,
                    relax.PrimValue(self.sliding_window),
                    relax.PrimValue(attention_sink_size),
                ],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_override,
                args=[
                    v_cache,
                    squeezed_value,
                    relax.PrimValue(self.sliding_window),
                    relax.PrimValue(attention_sink_size),
                ],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )

        return key, value, (k_cache, v_cache)

    def forward(
        self,
        hidden_states: relax.Expr,
        cache_len_shape: relax.Expr,
        kv_seq_len_shape: relax.Expr,
        cache_offset_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[relax.Expr], Optional[Tuple[relax.Expr]]]:
        # pylint: disable=import-outside-toplevel
        from tvm.relax.op import astype, matmul, maximum, permute_dims, reshape, split
        from tvm.relax.op.nn import softmax

        bsz, q_len, _ = hidden_states.struct_info.shape
        assert bsz == 1, "Only support batch size 1 at this moment."

        if self.combine_matmul:
            qkv_cur = nn.emit(
                split(
                    self.query_key_value_proj(hidden_states),
                    indices_or_sections=[
                        self.num_query_heads * self.head_dim,
                        (self.num_query_heads + self.num_key_value_heads) * self.head_dim,
                    ],
                    axis=-1,
                )
            )
            query = relax.TupleGetItem(qkv_cur, 0)
            key_cur = relax.TupleGetItem(qkv_cur, 1)
            value_cur = relax.TupleGetItem(qkv_cur, 2)
        else:
            query = self.q_proj(hidden_states)
            key_cur = self.k_proj(hidden_states)
            value_cur = self.v_proj(hidden_states)

        query = nn.emit(
            reshape(
                query,
                (bsz, q_len, self.num_query_heads, self.head_dim),
            ),
        )
        key_cur = nn.emit(
            reshape(
                key_cur,
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
            ),
        )
        value_cur = nn.emit(
            reshape(
                value_cur,
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
            ),
        )

        # concat current kv with cached kv (unrotating the cache)
        rolling_cache_len = cache_len_shape.struct_info.values[0]
        kv_seq_len = kv_seq_len_shape.struct_info.values[0]
        cache_offset = cache_offset_shape.struct_info.values[0]
        key, value, updated_key_value = self.interleave_kv(
            key_cur,
            value_cur,
            kv_seq_len,
            rolling_cache_len,
            cache_offset,
            self.attention_sink_size,
            past_key_value,
        )

        # cache relative position embeddings (after KV Cache)
        query, key = apply_rotary_pos_emb(
            query,
            key,
            self.rope_theta,
            q_offset=rolling_cache_len,
        )

        if self.num_key_value_heads != self.num_query_heads:
            n_rep = self.num_query_heads // self.num_key_value_heads
            key = nn.emit(relax.op.repeat(key, n_rep, axis=2))
            value = nn.emit(relax.op.repeat(value, n_rep, axis=2))

        query = nn.emit(permute_dims(query, [0, 2, 1, 3]))
        key = nn.emit(permute_dims(key, [0, 2, 1, 3]))
        value = nn.emit(permute_dims(value, [0, 2, 1, 3]))

        attn_weights = nn.emit(
            matmul(query, permute_dims(key, [0, 1, 3, 2]))
            / relax.const(math.sqrt(self.head_dim), query.struct_info.dtype)
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
        if attn_weights.struct_info.dtype != query.struct_info.dtype:
            attn_weights = astype(attn_weights, query.struct_info.dtype)
        attn_output = nn.emit(matmul(attn_weights, value))

        attn_output = nn.emit(permute_dims(attn_output, [0, 2, 1, 3]))
        attn_output = nn.emit(
            reshape(attn_output, (bsz, q_len, self.head_dim * self.num_query_heads))
        )

        attn_output = self.o_proj(attn_output)

        return attn_output, ((None, None) if updated_key_value is None else updated_key_value)


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MistralRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        cache_len_shape: relax.Expr,
        kv_seq_len_shape: relax.Expr,
        cache_offset_shape: relax.Expr,
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
            cache_len_shape=cache_len_shape,
            kv_seq_len_shape=kv_seq_len_shape,
            cache_offset_shape=cache_offset_shape,
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


def _make_sliding_window_mask(input_shape, kv_seq_len, sliding_window, dtype):
    # See `tests/python/test_sliding_window_mask.py` for more on its behavior.
    # [bsz, tgt_len] -> [bsz, 1, tgt_len, kv_seq_len]

    bsz, tgt_len = input_shape  # TODO: only support batch size of 1 for now
    cache_len = kv_seq_len - tgt_len  # number of elements in cache

    if isinstance(tgt_len, tvm.tir.SizeVar) or tgt_len > 1:
        # Either 1. First prefill, or 2. Subsequent prefill
        from tvm.relax.op import broadcast_to  # pylint: disable=import-outside-toplevel

        def sliding_window_min_max_te(sliding_window):
            return te.compute(
                (tgt_len, kv_seq_len),
                lambda i, j: tvm.tir.Select(
                    tvm.tir.all(i + cache_len >= j, i + cache_len - j < sliding_window),
                    tvm.tir.max_value(dtype),
                    tvm.tir.min_value(dtype),
                ),
                name="make_diag_mask_sliding_window_te",
            )

        mask = nn.emit_te(sliding_window_min_max_te, sliding_window)
        return nn.emit(broadcast_to(mask, (bsz, 1, tgt_len, kv_seq_len)))

    else:
        # 3. Decode (equivalent to prefilling a chunk of size 1)
        # Mask nothing here since WS == cache_size
        bsz, tgt_len = input_shape
        return nn.emit(
            relax.op.full(
                (bsz, 1, tgt_len, kv_seq_len),
                relax.const(tvm.tir.max_value(dtype).value, dtype),
                dtype,
            )
        )


class MistralEmbedTokens(nn.Module):
    def __init__(self, config: MistralConfig, vocab_size_var: tvm.tir.SizeVar):
        self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class MistralEmbedTokensWrapper(nn.Module):
    def __init__(self, config: MistralConfig, vocab_size_var: tvm.tir.SizeVar):
        # build a wrapper to ensure that the naming of the embed_tokens parameter is consistent
        self.model = MistralEmbedTokens(config, vocab_size_var)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.model(input_ids)
        return inputs_embeds


class MistralModel(nn.Module):
    def __init__(self, config: MistralConfig, vocab_size_var: tvm.tir.SizeVar, sep_embed: bool = False):
        self.num_shards = config.num_shards
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = MistralRMSNorm(config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

    def forward(
        self,
        inputs: relax.Expr,
        cache_len_shape: relax.Expr,
        kv_seq_len_shape: relax.Expr,
        cache_offset_shape: relax.Expr,
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
        kv_seq_len = kv_seq_len_shape.struct_info.values[0]

        # embed positions
        attention_mask = _make_sliding_window_mask(
            (batch_size, seq_length),
            kv_seq_len,
            self.sliding_window,
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
                cache_len_shape=cache_len_shape,
                kv_seq_len_shape=kv_seq_len_shape,
                cache_offset_shape=cache_offset_shape,
            )
            next_decoder_cache += key_value_cache

        hidden_states = self.norm(hidden_states)

        assert len(next_decoder_cache) == len(self.layers) * 2
        return hidden_states, next_decoder_cache


class MistralForCausalLM(nn.Module):
    def __init__(self, config: MistralConfig, vocab_size_var: tvm.tir.SizeVar, sep_embed: bool = False):
        self.model = MistralModel(config, vocab_size_var, sep_embed)
        self.lm_head = Linear(config.hidden_size, vocab_size_var, dtype=config.dtype, bias=False)

        ############ Rotary embedding constants ############
        assert config.hidden_size % config.num_attention_heads == 0
        head_dim = config.hidden_size // config.num_attention_heads

        # Set the cached sin/cos to the maximum of 2048 and max seq len.
        # This will be eliminated further with online rotary embedding calculation.
        rope_cache_len = te.var("rope_cache_len", "int64")
        self.cos_cached = nn.Parameter(
            (rope_cache_len, head_dim), dtype=config.dtype, name="cos_cached"
        )
        self.sin_cached = nn.Parameter(
            (rope_cache_len, head_dim), dtype=config.dtype, name="sin_cached"
        )
        ############ End ############

    def forward(
        self,
        inputs: relax.Expr,
        cache_len_shape: relax.Expr,
        kv_seq_len_shape: relax.Expr,
        cache_offset_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        hidden_states, key_value_cache = self.model(
            inputs=inputs,
            cache_len_shape=cache_len_shape,
            kv_seq_len_shape=kv_seq_len_shape,
            cache_offset_shape=cache_offset_shape,
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
    config: MistralConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    bsz = 1
    seq_len = tvm.tir.SizeVar("n", "int64")
    with bb.function(func_name):
        model = MistralEmbedTokensWrapper(config, tvm.tir.SizeVar("vocab_size", "int64"))
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
    config: MistralConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    bsz = 1
    seq_len = tvm.tir.SizeVar("n", "int64")  # number of tokens for the input
    rolling_cache_len = tvm.tir.SizeVar("c", "int64")  # rolling_cache_len captures number of elements in the cache
    kv_seq_len = tvm.tir.SizeVar(
        "k", "int64"
    )  # kv_seq_len captures number of elements in cache + seq_len
    cache_offset = tvm.tir.SizeVar(
        "o", "int64"
    )  # slidinf window kv cache offset

    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = MistralForCausalLM(config, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = (
            nn.Placeholder((bsz, seq_len, hidden_size), dtype=config.dtype, name="inputs_embeds")
            if sep_embed
            else nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        )
        cache_len_shape = relax.Var(
            "rolling_cache_len", relax.ShapeStructInfo((rolling_cache_len,))
        )
        kv_seq_len_shape = relax.Var("kv_seq_len", relax.ShapeStructInfo((kv_seq_len,)))
        cache_offset_shape = relax.Var("cache_offset", relax.ShapeStructInfo((cache_offset,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                inputs,
                cache_len_shape,
                kv_seq_len_shape,
                cache_offset_shape,
                past_key_values=past_key_values,
            )
            params = [
                inputs,
                cache_len_shape,
                kv_seq_len_shape,
                cache_offset_shape,
                past_key_values,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 5))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: MistralConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    bsz = 1
    rolling_cache_len = tvm.tir.SizeVar("c", "int64")  # rolling_cache_len captures number of elements in the cache
    kv_seq_len = tvm.tir.SizeVar(
        "k", "int64"
    )  # kv_seq_len captures number of elements in cache + seq_len
    cache_offset = tvm.tir.SizeVar(
        "o", "int64"
    )  # sliding window kv cache offset

    with bb.function(func_name):
        model = MistralForCausalLM(config, tvm.tir.SizeVar("vocab_size", "int64"))
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
        cache_len_shape = relax.Var(
            "rolling_cache_len", relax.ShapeStructInfo((rolling_cache_len,))
        )
        kv_seq_len_shape = relax.Var("kv_seq_len", relax.ShapeStructInfo((kv_seq_len,)))
        cache_offset_shape = relax.Var("cache_offset", relax.ShapeStructInfo((cache_offset,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids,
                cache_len_shape,
                kv_seq_len_shape,
                cache_offset_shape,
                past_key_values=past_key_values,
            )
            params = [
                input_ids,
                cache_len_shape,
                kv_seq_len_shape,
                cache_offset_shape,
                past_key_values,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 5))


def create_kv_cache_func(bb: relax.BlockBuilder, config: MistralConfig) -> None:
    num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
    init_shape = relax.ShapeExpr(
        (
            config.sliding_window,
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


def create_softmax_func(bb: relax.BlockBuilder, config: MistralConfig) -> None:
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


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    sep_embed = args.sep_embed
    assert not sep_embed, "Mistral does not support separate embedding."

    if args.sliding_window != -1:
        hf_config["sliding_window"] = args.sliding_window
        if args.attention_sink_size > 0:
            hf_config["attention_sink_size"] = args.attention_sink_size
    if args.max_seq_len != -1:
        hf_config["max_sequence_length"] = args.max_seq_len

    config = MistralConfig(
        **hf_config,
        dtype=dtype,
        combine_matmul=True,
        num_shards=args.num_shards,
        build_model_only=args.build_model_only,
    )

    # prefill chunk size same as sliding window by default
    if args.prefill_chunk_size < 1:
        args.prefill_chunk_size = config.sliding_window - config.attention_sink_size

    assert config.sliding_window != -1
    assert args.prefill_chunk_size <= config.sliding_window - config.attention_sink_size

    param_manager = ParamManager()
    bb = relax.BlockBuilder()

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
        sliding_window=config.sliding_window,
        prefill_chunk_size=args.prefill_chunk_size,
    )

    mod = bb.get()
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, relax.Function):
            mod[gv] = func.with_attr(
                "tir_var_upper_bound",
                {
                    "n": args.prefill_chunk_size,
                    "c": config.sliding_window,
                    "k": config.sliding_window + args.prefill_chunk_size,
                },
            )

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
        hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads

        if "query_key_value_proj" in relax_pname:
            q_heads = config.num_attention_heads
            kv_heads = config.get_num_key_value_heads()
            q, k, v = torch_params
            assert q.shape == (q_heads * head_dim, hidden_size)
            assert k.shape == (kv_heads * head_dim, hidden_size)
            assert v.shape == (kv_heads * head_dim, hidden_size)
            qkv = np.concatenate([q, k, v], axis=0).astype(dtype)
            return qkv
        if "gate_up_proj" in relax_pname:
            gate, up = torch_params
            gate_up = np.concatenate([gate, up], axis=0).astype(dtype)
            return gate_up
        raise ValueError("Unexpected param loading")

    param_manager.set_param_loading_func(
        args.model_path,
        args.use_safetensors,
        f_convert_pname_fwd,
        f_convert_param_bkwd,
        f_compute_relax_param,
    )

    device = tvm.cpu()
    param_list = [None] * param_manager.nparam_to_load

    head_dim = config.hidden_size / config.num_attention_heads
    inv_freq = 1.0 / (config.rope_theta ** (np.arange(0, head_dim, 2).astype("float32") / head_dim))

    # The following cos/sin values can be removed but **are kept for compatibility issues**.
    t = np.arange(2048, dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    param_list[-2] = tvm.nd.array(np.cos(emb).astype(config.dtype), device)
    param_list[-1] = tvm.nd.array(np.sin(emb).astype(config.dtype), device)

    return mod, param_manager, param_list, config
