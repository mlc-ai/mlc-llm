import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import relax, te, tir
from tvm.relax.op import ccl
from tvm.relax.testing import nn
from tvm.script import relax as R, tir as T

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList
from .param_manager import ParamManager


@dataclass
class LlamaConfig:
    def __init__(
        self,
        dtype="float32",
        max_sequence_length=2048,
        vocab_size=32000,  # some models like WizardMath can have 32001
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        position_embedding_base=10000,
        combine_matmul=True,
        build_model_only=False,
        num_shards=1,
        sliding_window=None,
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
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.position_embedding_base = position_embedding_base
        self.combine_matmul = combine_matmul
        self.sliding_window = sliding_window

        if build_model_only and num_shards > 1:
            self.num_shards = num_shards
        else:
            self.num_shards = 1
        self.kwargs = kwargs

    def get_num_key_value_heads(self):
        if self.num_key_value_heads is None:
            return self.num_attention_heads

        return self.num_key_value_heads


class MixtralConfig(LlamaConfig):
    num_experts_per_tok: int
    num_local_experts: int
    sliding_window: int
    # router_aux_loss_coef: float  # not sure if needed
    quantization_scheme: QuantizationScheme

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts_per_tok = kwargs["num_experts_per_tok"]
        self.num_local_experts = kwargs["num_local_experts"]
        self.sliding_window = kwargs["sliding_window"]
        # self.router_aux_loss_coef = kwargs["router_aux_loss_coef"]

        # FIXME: remove this
        self.quantization_scheme = kwargs["quantization_scheme"]


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
        from tvm.relax.op import reshape, take

        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(take(self.weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = self.weight.struct_info.shape.values[-1]
            x = nn.emit(reshape(x, shape=[-1]))
            embedding = nn.emit(take(self.weight, x, axis=0))
            return nn.emit(reshape(embedding, [*x_shape, emb_size]))


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, eps=1e-6):
        self.weight = nn.Parameter((hidden_size,), dtype=dtype, name="rms_norm_weight")
        self.variance_epsilon = tvm.tir.const(eps, dtype)

    def forward(self, hidden_states):
        from tvm import te, tir

        def f_rms_norm(x, weight):
            is_float32 = x.dtype == "float32"

            def f_square(x):
                return tir.Cast("float32", x) * tir.Cast("float32", x) if not is_float32 else x * x

            def f_mul_cast(x, y):
                value = x * y
                if not is_float32:
                    value = tir.Cast(x.dtype, value)
                return value

            def f_div_cast_2d(i, k):
                x_val = x[i, k]
                if not is_float32:
                    x_val = tir.Cast("float32", x_val)
                return x_val / tir.sqrt(square_sum[i] / x.shape[1] + self.variance_epsilon)

            def f_div_cast_3d(bsz, i, k):
                x_val = x[bsz, i, k]
                if not is_float32:
                    x_val = tir.Cast("float32", x_val)
                return x_val / tir.sqrt(square_sum[bsz, i] / x.shape[2] + self.variance_epsilon)

            k = te.reduce_axis((0, x.shape[-1]), name="k")

            if len(x.shape) == 2:
                square_sum = te.compute(
                    (x.shape[0],),
                    lambda i: te.sum(f_square(x[i, k]), axis=k),
                    name=x.op.name + "red_temp",
                )

                return te.compute(
                    x.shape,
                    lambda i, k: f_mul_cast(weight(k), f_div_cast_2d(i, k)),
                    name="rms_norm",
                )
            else:
                square_sum = te.compute(
                    (x.shape[0], x.shape[1]),
                    lambda bsz, i: te.sum(f_square(x[bsz, i, k]), axis=k),
                    name=x.op.name + "red_temp",
                )

                return te.compute(
                    x.shape,
                    lambda bsz, i, k: f_mul_cast(weight(k), f_div_cast_3d(bsz, i, k)),
                    name="rms_norm",
                )

        return nn.emit_te(f_rms_norm, hidden_states, self.weight, primfunc_name_hint="rms_norm")


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
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


def rotary_modulate_by_freq(tensor, idx, pos, position_embedding_base):
    head_dim = tensor.shape[-1]
    dtype = tensor.dtype
    n_feat_half = head_dim // 2
    feat_idx = idx[-1]
    inv_freq = te.const(1, "float32") / (
        te.power(
            te.const(position_embedding_base, "float32"),
            ((2 * feat_idx) % head_dim).astype("float32") / head_dim.astype("float32"),
        )
    )
    freq = pos * inv_freq
    left_indices = idx[:-1] + (feat_idx - n_feat_half,)
    right_indices = idx[:-1] + (feat_idx + n_feat_half,)
    return te.cos(freq).astype(dtype) * tensor(*idx) + te.sin(freq).astype(dtype) * tvm.tir.Select(
        feat_idx >= n_feat_half,
        tensor[(*left_indices,)],
        -tensor[(*right_indices,)],
    )


def apply_rotary_pos_emb(q, k, position_embedding_base, offset: int = 0):
    def f_rotary_embedding(tensor, offset):
        def rotary_compute(*idx):
            pos = (offset + idx[-3]).astype("float32")
            return rotary_modulate_by_freq(
                tensor,
                idx,
                pos,
                position_embedding_base,
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(f_rotary_embedding, q, offset, primfunc_name_hint="rotary_embedding")
    k_embed = nn.emit_te(f_rotary_embedding, k, offset, primfunc_name_hint="rotary_embedding")
    return q_embed, k_embed


class LlamaAttentionBase(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        dtype = config.dtype
        self.num_shards = config.num_shards
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
        self.num_query_heads = config.num_attention_heads // self.num_shards
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.position_embedding_base = config.position_embedding_base

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

    def project_qkv(self, hidden_states, query_output_shape, kv_output_shape):
        from tvm.relax.op import reshape, split

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
            reshape(query_states, query_output_shape),
        )
        key_states = nn.emit(
            reshape(key_states, kv_output_shape),
        )
        value_states = nn.emit(
            reshape(value_states, kv_output_shape),
        )

        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: Optional[relax.Expr],
        past_key_values: Union[relax.Expr, Tuple[relax.Expr]],
        layer_id: int,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Union[relax.Expr, Tuple[relax.Expr]]]:
        bsz, q_len, _ = hidden_states.struct_info.shape

        query_states, key_states, value_states = self.project_qkv(
            hidden_states,
            (bsz, q_len, self.num_query_heads, self.head_dim),
            (bsz, q_len, self.num_key_value_heads, self.head_dim),
        )

        from tvm.relax.op import reshape

        attn_output, past_key_values = self.attention_fwd(
            query_states,
            key_states,
            value_states,
            past_key_values,
            bsz,
            q_len,
            layer_id=layer_id,
            all_seq_len_shape=all_seq_len_shape,
            attention_mask=attention_mask,
        )

        attn_output = nn.emit(
            reshape(attn_output, (bsz, q_len, self.head_dim * self.num_query_heads))
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_values

    def attention_fwd(
        self,
        query_states: relax.Expr,
        key_states: relax.Expr,
        value_states: relax.Expr,
        past_key_values: relax.Expr,
        batch_size: tir.PrimExpr,
        q_len: tir.PrimExpr,
        **kwargs,
    ):
        raise NotImplementedError()


class LlamaPagedAttention(LlamaAttentionBase):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        ctx_mod = relax.BlockBuilder.current().get()
        self.kv_cache_transpose_append = ctx_mod.get_global_var("kv_cache_transpose_append")
        self.attention_compute_prefill = ctx_mod.get_global_var("attention_prefill")
        self.attention_compute_decode = ctx_mod.get_global_var("attention_decode")

    def attention_fwd(
        self,
        query_states: relax.Expr,
        key_states: relax.Expr,
        value_states: relax.Expr,
        past_key_values: relax.Expr,
        batch_size: tir.PrimExpr,
        q_len: tir.PrimExpr,
        **kwargs,
    ) -> Tuple[relax.Expr, relax.Expr]:
        assert "layer_id" in kwargs and isinstance(kwargs["layer_id"], int)
        layer_id = kwargs["layer_id"]

        f_kv_cache_append = relax.extern("vm.builtin.paged_attention_kv_cache_append")
        past_key_values = nn.emit(
            relax.call_pure_packed(
                f_kv_cache_append,
                past_key_values,
                self.kv_cache_transpose_append,
                key_states,
                value_states,
                relax.PrimValue(layer_id),
                sinfo_args=relax.ObjectStructInfo(),
            )
        )

        f_kv_cache_attention = relax.extern("vm.builtin.paged_attention_kv_cache_attention")
        is_decode = query_states.struct_info.shape[1] == 1
        attn_output = nn.emit(
            relax.call_dps_packed(
                f_kv_cache_attention,
                [
                    past_key_values,
                    self.attention_compute_decode if is_decode else self.attention_compute_prefill,
                    query_states,
                    relax.PrimValue(layer_id),
                    True,
                    1.0,
                    self.position_embedding_base,
                ],
                out_sinfo=relax.TensorStructInfo(
                    ((batch_size, q_len, self.num_query_heads, self.head_dim)),
                    query_states.struct_info.dtype,
                ),
            )
        )
        return attn_output, past_key_values


class LlamaAttention(LlamaAttentionBase):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def attention_fwd(
        self,
        query_states: relax.Expr,
        key_states: relax.Expr,
        value_states: relax.Expr,
        past_key_values: relax.Expr,
        batch_size: tir.PrimExpr,
        q_len: tir.PrimExpr,
        **kwargs,
    ) -> Tuple[relax.Expr, Tuple[relax.Expr]]:
        assert "attention_mask" in kwargs
        assert "all_seq_len_shape" in kwargs
        attention_mask = kwargs["attention_mask"]
        kv_seq_len = kwargs["all_seq_len_shape"].struct_info.values[0]

        from tvm.relax.op import astype, matmul, maximum, permute_dims, reshape, squeeze
        from tvm.relax.op.nn import softmax

        offset = kv_seq_len - q_len
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            self.position_embedding_base,
            offset=offset,
        )
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
        k_cache, v_cache = past_key_values
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
        past_key_values = (k_cache, v_cache)
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
            (batch_size, tvm.tir.IntImm("int64", 1), q_len, kv_seq_len),
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
        return attn_output, past_key_values


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, enable_batching: bool):
        attn_class = LlamaPagedAttention if enable_batching else LlamaAttention
        self.hidden_size = config.hidden_size
        self.self_attn = attn_class(config)
        if isinstance(config, MixtralConfig):
            from .mixtral import MoE

            self.use_moe = True
            self.block_sparse_moe = MoE(config)
        else:
            self.use_moe = False
            self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )

    def post_self_attn(self, hidden_states, residual):
        if self.self_attn.num_shards > 1:
            residual = nn.emit(
                residual / R.const(self.self_attn.num_shards, dtype=residual.struct_info.dtype)
            )
        hidden_states = nn.emit(residual + hidden_states)
        if self.self_attn.num_shards > 1:
            hidden_states = nn.emit(ccl.allreduce(hidden_states, "sum"))

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        model = self.block_sparse_moe if self.use_moe else self.mlp

        hidden_states = model(hidden_states)
        if model.num_shards > 1:
            residual = nn.emit(
                residual / R.const(model.num_shards, dtype=residual.struct_info.dtype)
            )
        hidden_states = nn.emit(residual + hidden_states)
        if model.num_shards > 1:
            hidden_states = nn.emit(ccl.allreduce(hidden_states, "sum"))

        return hidden_states

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: Optional[relax.Expr],
        past_key_values: Union[relax.Expr, Tuple[relax.Expr]],
        layer_id: int,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            all_seq_len_shape=all_seq_len_shape,
            layer_id=layer_id,
        )
        hidden_states = self.post_self_attn(hidden_states, residual)
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


class LlamaEmbedTokens(nn.Module):
    def __init__(self, config: LlamaConfig, vocab_size_var: tvm.tir.SizeVar):
        self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class LlamaEmbedTokensWrapper(nn.Module):
    def __init__(self, config: LlamaConfig, vocab_size_var: tvm.tir.SizeVar):
        # build a wrapper to ensure that the naming of the embed_tokens parameter is consistent
        self.model = LlamaEmbedTokens(config, vocab_size_var)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.model(input_ids)
        return inputs_embeds


class LlamaModelBase(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tir.SizeVar,
        sep_embed: bool = False,
        enable_batching: bool = False,
    ):
        self.num_shards = config.num_shards
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [LlamaDecoderLayer(config, enable_batching) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: Optional[relax.Expr],
        past_key_values: relax.Expr,
    ):
        raise NotImplementedError()


class LlamaModelForSingleSequence(LlamaModelBase):
    def __init__(self, config: LlamaConfig, vocab_size_var: tvm.tir.SizeVar, sep_embed: bool = False):
        super().__init__(config, vocab_size_var, sep_embed, enable_batching=False)

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
        all_seq_len_shape: Optional[relax.Expr],
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
                past_key_values=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
                layer_id=idx,
            )
            next_decoder_cache += key_value_cache

        hidden_states = self.norm(hidden_states)

        assert len(next_decoder_cache) == len(self.layers) * 2
        return hidden_states, next_decoder_cache


class LlamaModelForBatching(LlamaModelBase):
    def __init__(self, config: LlamaConfig, vocab_size_var: tvm.tir.SizeVar, sep_embed: bool):
        assert sep_embed
        super().__init__(config, vocab_size_var, sep_embed=True, enable_batching=True)

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: Optional[relax.Expr],
        past_key_values: relax.Expr,
    ):
        assert all_seq_len_shape is None
        if self.num_shards > 1:
            inputs = nn.emit(ccl.broadcast_from_worker0(inputs))
        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(inputs)
        else:
            inputs_embeds = inputs

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            assert past_key_values is not None
            hidden_states, past_key_values = decoder_layer(
                hidden_states,
                attention_mask=None,
                past_key_values=past_key_values,
                all_seq_len_shape=all_seq_len_shape,
                layer_id=idx,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tvm.tir.SizeVar,
        sep_embed: bool = False,
        enable_batching: bool = False,
    ):
        model_class = LlamaModelForBatching if enable_batching else LlamaModelForSingleSequence
        self.model = model_class(config, vocab_size_var, sep_embed)
        self.lm_head = Linear(config.hidden_size, vocab_size_var, dtype=config.dtype, bias=False)

        ############ Rotary embedding constants ############
        assert config.hidden_size % config.num_attention_heads == 0
        head_dim = config.hidden_size // config.num_attention_heads

        # Set the cached sin/cos to the maximum of 2048 and max seq len.
        # This will be eliminated further with online rotary embedding calculation.
        cache_len = te.var("cache_len", "int64")
        self.cos_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="cos_cached")
        self.sin_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="sin_cached")
        ############ End ############

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: Optional[relax.Expr],
        past_key_values: relax.Expr,
        logit_positions: Optional[relax.Expr] = None,
    ):
        hidden_states, key_value_cache = self.model(
            inputs=inputs,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def te_slicing(x: te.Tensor):
            assert x.ndim == 3
            return te.compute(
                shape=(x.shape[0], 1, x.shape[2]),
                fcompute=lambda i, j, k: x[i, x.shape[1] - 1, k],
                name="slice",
            )

        if hidden_states.struct_info.shape[1] != 1:
            if logit_positions is None:
                hidden_states = nn.emit_te(te_slicing, hidden_states, primfunc_name_hint="slice")
            else:
                hidden_states = relax.op.take(hidden_states, logit_positions, axis=1)
        logits = self.lm_head(hidden_states)

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
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = LlamaEmbedTokensWrapper(config, tvm.tir.SizeVar("vocab_size", "int64"))
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((1, seq_len), dtype="int32", name="input_ids")
        with bb.dataflow():
            inputs_embeds = model(input_ids)
            params = [input_ids] + model.parameters()
            gv = bb.emit_output(inputs_embeds)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 1))


def create_prefill_func_for_single_seq(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    bsz = 1
    seq_len = tvm.tir.SizeVar("n", "int64")
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = LlamaForCausalLM(
            config, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed, enable_batching=False
        )
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


def create_prefill_func_for_batching(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "prefill_with_embed"

    bsz = tir.SizeVar("nseq", "int64")
    total_seq_len = tvm.tir.SizeVar("m", "int64")
    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = LlamaForCausalLM(
            config, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed=True, enable_batching=True
        )
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = nn.Placeholder(
            (1, total_seq_len, hidden_size), dtype=config.dtype, name="inputs_embeds"
        )
        logit_pos = nn.Placeholder((bsz,), dtype="int32", name="logit_positions")
        past_key_values = relax.Var("kv_cache", relax.ObjectStructInfo())
        with bb.dataflow():
            logits, key_value_cache = model(
                inputs,
                all_seq_len_shape=None,
                past_key_values=past_key_values,
                logit_positions=logit_pos,
            )
            params = [inputs, logit_pos, past_key_values] + model.parameters()
            gv = bb.emit_output((logits, key_value_cache))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func_for_single_seq(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    bsz = 1
    all_seq_len = tvm.tir.SizeVar("m", "int64")

    with bb.function(func_name):
        model = LlamaForCausalLM(config, tvm.tir.SizeVar("vocab_size", "int64"))
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


def create_decoding_func_for_batching(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode_with_embed"

    bsz = tir.SizeVar("nseq", "int64")
    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = LlamaForCausalLM(
            config, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed=True, enable_batching=True
        )
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = nn.Placeholder((bsz, 1, hidden_size), dtype=config.dtype, name="inputs_embeds")
        past_key_values = relax.Var("kv_cache", relax.ObjectStructInfo())
        with bb.dataflow():
            logits, key_value_cache = model(
                inputs, all_seq_len_shape=None, past_key_values=past_key_values
            )
            params = [inputs, past_key_values] + model.parameters()
            gv = bb.emit_output((logits, key_value_cache))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 2))


def create_kv_cache_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
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


def create_paged_kv_cache_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    head_dim = config.hidden_size // config.num_attention_heads
    num_key_value_heads = config.get_num_key_value_heads() // config.num_shards

    page_size = tir.SizeVar("page_size", "int64")
    total_seq_len = tir.SizeVar("total_seq_len", "int64")
    reserved_nseq = tir.SizeVar("reserved_nseq", "int64")
    cache_config = relax.Var(
        "cache_config",
        relax.ShapeStructInfo([reserved_nseq, total_seq_len, page_size]),
    )

    with bb.function("create_kv_cache", [cache_config]):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros((), config.dtype))
            f_kv_cache_create = relax.extern("vm.builtin.paged_attention_kv_cache_create")
            cache = bb.emit_output(
                relax.Call(
                    f_kv_cache_create,
                    args=[
                        cache_config,
                        relax.PrimValue(config.num_hidden_layers),
                        relax.PrimValue(num_key_value_heads),
                        relax.PrimValue(head_dim),
                        zeros,
                        relax.PrimValue(0),
                    ],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
        bb.emit_func_output(cache)


def create_softmax_func_for_single_seq(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
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


def create_softmax_func_for_batching(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    with bb.function("softmax_with_temperature"):
        bsz = tvm.tir.SizeVar("nseq", "int64")
        logits = nn.Placeholder(
            (bsz, 1, tvm.tir.SizeVar("vocab_size", "int64")),
            dtype="float32",
            name="logits",
        )
        temperature = nn.Placeholder((bsz,), dtype="float32", name="temperature")
        with bb.dataflow():
            t_reshaped = bb.emit(relax.op.reshape(temperature, (bsz, 1, 1)))
            div = bb.emit(relax.op.divide(logits, t_reshaped))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def emit_paged_kv_cache_op(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    from tvm.script import tir as T

    num_layers = config.num_hidden_layers
    num_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # fmt: off
    @T.prim_func
    def kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_page_table_indptr: T.handle,
        var_page_table_values: T.handle,
        var_last_page_offset: T.handle,
        var_append_length_indptr: T.handle,
        var_pos2seqidx: T.handle,
        layer_id: T.int64,
    ):
        nseq = T.int64()
        ntoken = T.SizeVar("ntoken", "int64")
        npage = T.int64()
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()

        pages = T.match_buffer(var_pages, (num_pages, num_layers, 2, num_heads, page_size, head_dim), config.dtype)
        k_data = T.match_buffer(var_k_data, (ntoken, num_heads, head_dim), config.dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, num_heads, head_dim), config.dtype)
        last_page_offset = T.match_buffer(var_last_page_offset, (nseq,), "int32")
        page_table_indptr = T.match_buffer(var_page_table_indptr, (nseq + 1,), "int32")
        page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
        append_length_indptr = T.match_buffer(var_append_length_indptr, (nseq + 1,), "int32")
        pos2seqidx = T.match_buffer(var_pos2seqidx, (ntoken,), "int32")

        for global_pos, h, f in T.grid(ntoken, num_heads, head_dim):
            with T.block("k_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                seq_idx: T.int64 = T.Cast("int64", pos2seqidx[vgpos])
                seqlen: T.int64 = T.Cast("int64", (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx])
                pages[
                    page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)],
                    layer_id,
                    0,
                    vh,
                    T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                    vf,
                ] = k_data[vgpos, vh, vf]
            with T.block("v_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                seq_idx: T.int64 = T.Cast("int64", pos2seqidx[vgpos])
                seqlen: T.int64 = T.Cast("int64", (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx])
                pages[
                    page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)],
                    layer_id,
                    1,
                    vh,
                    T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                    vf,
                ] = v_data[vgpos, vh, vf]
    # fmt: on

    bb.add_func(kv_cache_transpose_append, "kv_cache_transpose_append")
    bb.add_func(relax.extern("paged_kv_cache.attention_kernel_prefill"), "attention_prefill")
    bb.add_func(relax.extern("paged_kv_cache.attention_kernel_decode"), "attention_decode")


def setup_params(mod, param_manager, dtype, config, args):
    mappings = [
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
    ]

    def f_convert_pname_fwd(pname: str) -> List[str]:
        qkv_str = "query_key_value_proj"
        gate_up_str = "gate_up_proj"

        if isinstance(config, MixtralConfig):
            for k, v in mappings:
                pname = pname.replace(k, v)
            # pname = pname.replace("model.", "")
            if config.quantization_scheme.name == "q4f16_ft":
                if pname.endswith("scales"):
                    # TODO: remove after quantization integarted
                    pname = pname.replace("scales", "weight")

            if config.combine_matmul:
                if qkv_str in pname:
                    return [
                        pname.replace(qkv_str, "q_proj"),
                        pname.replace(qkv_str, "k_proj"),
                        pname.replace(qkv_str, "v_proj"),
                    ]
                if "experts.gate_up_combined_proj" in pname:
                    return [
                        pname.replace("experts.gate_up_combined_proj", f"experts.{i}.w1")
                        for i in range(config.num_local_experts)
                    ] + [
                        pname.replace("experts.gate_up_combined_proj", f"experts.{i}.w3")
                        for i in range(config.num_local_experts)
                    ]

            if "experts" in pname:
                # not needed if using combine_matmul
                return [
                    pname.replace("experts", f"experts.{i}")
                    for i in range(config.num_local_experts)
                ]

        if not config.combine_matmul:
            return [pname]

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
        if isinstance(config, MixtralConfig):
            if "experts" in torch_pname:
                return None
            for v, k in mappings:
                torch_pname = torch_pname.replace(k, v)
        if not config.combine_matmul:
            return [(torch_pname, torch_param.astype(dtype))]

        combined_layers = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
        # print('bkwd pname ', torch_pname)
        if any([name in torch_pname for name in combined_layers]):
            return None
        return [(torch_pname, torch_param.astype(dtype))]

    def quantize(experts, relax_pname):
        print("quantizing experts", relax_pname)
        func = tvm.get_global_func("cutlass.symmetric_quantize")
        nd_experts = tvm.nd.array(experts)
        qweight, qscale = func(nd_experts, True)
        if relax_pname.endswith("weight"):
            return qweight
        else:
            assert relax_pname.endswith("scales")
            return qscale

    def f_compute_relax_param(relax_pname: str, torch_params: List[Any]):
        # Expected to enter this function only for the combined linear matmul weights.
        # Other weights are supposed to be loaded in `f_convert_param_bkwd` since
        # each other relax param has a unique corresponding torch param.
        if isinstance(config, MixtralConfig):
            if "gate_up_combined_proj" in relax_pname:
                # combine along out_features dimension and then experts dimension
                experts = []
                assert len(torch_params) == 2 * config.num_local_experts

                use_pytorch = True
                if use_pytorch and dtype == "float16":
                    import torch

                    torch_params = [torch.from_numpy(param).cuda() for param in torch_params]
                    for i in range(config.num_local_experts):
                        gate, up = (
                            torch_params[i],
                            torch_params[i + config.num_local_experts],
                        )  # torch weight in col major
                        gate_up = torch.concatenate([gate, up], axis=0).type(torch.float16)
                        experts.append(gate_up.transpose(1, 0))
                    result = torch.stack(experts)
                    result = result.cpu().numpy()
                else:
                    for i in range(config.num_local_experts):
                        gate, up = (
                            torch_params[i],
                            torch_params[i + config.num_local_experts],
                        )  # torch weight in col major
                        gate_up = np.concatenate([gate, up], axis=0).astype(dtype)
                        experts.append(gate_up.transpose())
                    result = np.stack(experts)
                # print(config.quantization_scheme.name)
                if config.quantization_scheme.name == "q4f16_ft" and "experts" in relax_pname:
                    result = quantize(result, relax_pname)
                return result
            if "experts" in relax_pname:
                use_pytorch = True
                if use_pytorch and dtype == "float16":
                    import torch

                    torch_params = [torch.from_numpy(param).cuda() for param in torch_params]
                    experts = torch.stack(
                        [expert.type(torch.float16).transpose(1, 0) for expert in torch_params]
                    )
                    result = experts.cpu().numpy()
                else:
                    experts = [expert.astype(dtype).transpose() for expert in torch_params]
                    result = np.stack(experts)
                # torch_params = [torch.from_numpy(param).cuda() for param in torch_params]
                # experts = [expert.type(dtype).transpose(1, 0) for expert in torch_params]
                # result = torch.stack(experts).detach().numpy()
                if config.quantization_scheme.name == "q4f16_ft" and "experts" in relax_pname:
                    result = quantize(result, relax_pname)
                return result

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
    inv_freq = 1.0 / (
        config.position_embedding_base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
    )

    # The following cos/sin values can be removed but **are kept for compatibility issues**.
    t = np.arange(2048, dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    param_list[-2] = tvm.nd.array(np.cos(emb).astype(config.dtype), device)
    param_list[-1] = tvm.nd.array(np.sin(emb).astype(config.dtype), device)

    return mod, param_manager, param_list, config


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    enable_batching = args.enable_batching
    sep_embed = args.sep_embed

    if enable_batching and not sep_embed:
        raise ValueError("`sep_embed` is required when batching is enabled.")

    position_embedding_base = 10000

    if "rope_theta" in hf_config:
        position_embedding_base = hf_config["rope_theta"]

    # Llama-2 variants use `max_position_embeddings` to encode maximum sequence length in their hf model cards,
    # while Llama-1 variants use `max_sequence_length`.
    # Thus, use `max_sequence_length` if defined. Otherwise, use `max_position_embeddings`.
    # If none of them is defined, throw an error.
    if "mixtral" in args.model.lower():
        # FIXME
        config = MixtralConfig(
            **hf_config,
            dtype=dtype,
            max_sequence_length=hf_config["max_position_embeddings"],
            position_embedding_base=position_embedding_base,
            combine_matmul=True,
            num_shards=args.num_shards,
            build_model_only=args.build_model_only,
            quantization_scheme=args.quantization,
        )
    elif "max_sequence_length" in hf_config:
        config = LlamaConfig(
            **hf_config,
            dtype=dtype,
            position_embedding_base=position_embedding_base,
            combine_matmul=True,
            num_shards=args.num_shards,
            build_model_only=args.build_model_only,
        )
    elif "max_position_embeddings" in hf_config:
        config = LlamaConfig(
            **hf_config,
            dtype=dtype,
            max_sequence_length=hf_config["max_position_embeddings"],
            position_embedding_base=position_embedding_base,
            combine_matmul=True,
            num_shards=args.num_shards,
            build_model_only=args.build_model_only,
        )
    else:
        raise Exception(
            "The model config should contain information about maximum sequence length."
        )

    # If there is a user-provided maximum sequence length, override hf config.
    if args.max_seq_len != -1:
        config.max_sequence_length = args.max_seq_len

    keep_params_after_load = (
        isinstance(config, MixtralConfig) and args.quantization.name == "q4f16_ft"
    )
    param_manager = ParamManager(keep_params_after_load)
    bb = relax.BlockBuilder()

    if sep_embed:
        create_embed_func(bb, param_manager, config, args.quantization)

    if enable_batching:
        emit_paged_kv_cache_op(bb, config)
        create_prefill_func_for_batching(bb, param_manager, config, args.quantization)
        create_decoding_func_for_batching(bb, param_manager, config, args.quantization)
        create_paged_kv_cache_func(bb, config)
        create_softmax_func_for_batching(bb, config)
    else:
        create_prefill_func_for_single_seq(bb, param_manager, config, args.quantization, sep_embed)
        create_decoding_func_for_single_seq(bb, param_manager, config, args.quantization)
        create_kv_cache_func(bb, config)
        create_softmax_func_for_single_seq(bb, config)

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
    tir_bound_map["vocab_size"] = args.max_vocab_size
    if enable_batching:
        tir_bound_map["nseq"] = args.max_batch_size
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, relax.Function):
            mod[gv] = func.with_attr("tir_var_upper_bound", tir_bound_map)

    if args.build_model_only:
        return mod, param_manager, None, config

    return setup_params(mod, param_manager, dtype, config, args)
