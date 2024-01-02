import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import tvm
from tvm import relax, te, tir
from tvm.relax.op import (
    astype,
    broadcast_to,
    expand_dims,
    matmul,
    maximum,
    minimum,
    permute_dims,
    repeat,
    reshape,
    split,
    squeeze,
)
from tvm.relax.op.nn import silu, softmax
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import Embedding, Linear, ModuleList, RotaryEmbedding
from .param_manager import ParamManager


@dataclass
class ChatGLMConfig:
    def __init__(
        self,
        add_bias_linear: bool = False,
        add_qkv_bias: bool = True,
        ffn_hidden_size: int = 13696,
        hidden_size: int = 4096,
        kv_channels: int = 128,
        layernorm_epsilon: float = 1e-05,
        multi_query_group_num: int = 2,
        num_attention_heads: int = 32,
        num_layers: int = 28,
        max_sequence_length: int = 2048,
        padded_vocab_size: int = 65024,
        eos_token_id: int = 2,
        bos_token_id: int = 0,
        dtype: str = "float32",
        **kwargs,
    ):
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_size = hidden_size
        self.kv_channels = kv_channels
        self.layernorm_epsilon = layernorm_epsilon
        self.multi_query_group_num = multi_query_group_num
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.max_sequence_length = min(2048, max_sequence_length)
        self.padded_vocab_size = padded_vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dtype = dtype
        self.kwargs = kwargs


def _repeat_kv(k: relax.Expr, v: relax.Expr, n_rep: int, shape: relax.Expr):
    k = nn.emit(reshape(repeat(k, n_rep, 1), shape))
    v = nn.emit(reshape(repeat(v, n_rep, 1), shape))
    return k, v


def _reshape(x: relax.Expr, shape: Tuple[int]):
    x = nn.emit(reshape(x, R.shape(shape)))
    return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, eps=1e-5):
        self.weight = nn.Parameter((hidden_size,), dtype=dtype, name="rms_norm_weight")
        self.eps = tvm.tir.const(eps, dtype)

    def forward(self, hidden_states):
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
                return x_val / tir.sqrt(square_sum[bsz, i] / x.shape[2] + self.eps)

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

        return nn.emit_te(
            f_rms_norm,
            hidden_states,
            self.weight,
            primfunc_name_hint="rms_norm",
        )


class CoreAttention(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.dtype = config.dtype

    def forward(
        self,
        q: relax.Expr,
        k: relax.Expr,
        v: relax.Expr,
        attention_mask: relax.Expr,
    ) -> relax.Expr:
        bsz, sl, nh, hd = q.struct_info.shape
        kv_sl = k.struct_info.shape[1]

        # [bsz, nh, sl, hd]
        q = nn.emit(permute_dims(q, [0, 2, 1, 3]))

        # [bsz, nh, kv_sl, hd]
        k = nn.emit(permute_dims(k, [0, 2, 1, 3]))
        v = nn.emit(permute_dims(v, [0, 2, 1, 3]))

        # Calculate Q.K: [bsz, nh, sl, kv_sl]
        matmul_result = nn.emit(
            matmul(q, permute_dims(k, [0, 1, 3, 2]))
            / relax.const(self.norm_factor, q.struct_info.dtype)
        )
        attention_scores = _reshape(matmul_result, (bsz, nh, sl, kv_sl))

        # Apply attention mask: [bsz, nh, sl, kv_sl]
        attention_scores = nn.emit(
            maximum(
                attention_scores,
                relax.const(
                    tvm.tir.min_value(attention_scores.struct_info.dtype).value,
                    attention_scores.struct_info.dtype,
                ),
            )
        )
        attention_scores = nn.emit(minimum(attention_scores, attention_mask))

        # Calculate Softmax(Q.K)
        if attention_scores.struct_info.dtype != "float32":
            attention_scores = astype(attention_scores, "float32")
        attention_probs = nn.emit(softmax(attention_scores, axis=-1))
        if attention_probs.struct_info.dtype != q.struct_info.dtype:
            attention_probs = astype(attention_probs, q.struct_info.dtype)

        # Calculate Softmax(Q.K).V
        context = nn.emit(matmul(attention_probs, v))
        context = nn.emit(permute_dims(context, [0, 2, 1, 3]))
        context = _reshape(context, (bsz, sl, nh * hd))

        return context


class SelfAttention(nn.Module):
    def __init__(
        self,
        config: ChatGLMConfig,
        rotary_pos_emb: RotaryEmbedding,
    ):
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        # Multi-query attention config
        self.num_multi_query_groups_per_partition = config.multi_query_group_num
        self.qkv_hidden_size = (
            self.projection_size
            + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
        )

        self.query_key_value = Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            config.dtype,
            bias=config.add_bias_linear or config.add_qkv_bias,
        )

        self.rotary_pos_emb = rotary_pos_emb

        self.core_attention = CoreAttention(config)

        self.dense = Linear(
            self.projection_size,
            config.hidden_size,
            config.dtype,
            bias=config.add_bias_linear,
        )

        self.dtype = config.dtype

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr, relax.Expr],
        attention_mask: relax.Expr,
    ) -> Tuple[relax.Expr, Tuple[relax.Expr, relax.Expr]]:
        # hidden_states: [bsz, sl, hs]
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

        bsz, sl, _ = hidden_states.struct_info.shape
        kv_sl = all_seq_len_shape.struct_info.values[0]

        mixed_x_layer = nn.emit(
            split(
                self.query_key_value(hidden_states),
                indices_or_sections=[
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    (
                        self.num_attention_heads_per_partition
                        + self.num_multi_query_groups_per_partition
                    )
                    * self.hidden_size_per_attention_head,
                ],
                axis=-1,
            )
        )

        q_shape = (
            bsz,
            sl,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        kv_shape = (
            bsz,
            sl,
            self.num_multi_query_groups_per_partition,
            self.hidden_size_per_attention_head,
        )

        # queries: [bsz, sl, nh, hd]
        q = _reshape(relax.TupleGetItem(mixed_x_layer, 0), q_shape)

        # keys: [bsz, sl, ng, hd]
        k = _reshape(relax.TupleGetItem(mixed_x_layer, 1), kv_shape)

        # values: [bsz, sl, ng, hd]
        v = _reshape(relax.TupleGetItem(mixed_x_layer, 2), kv_shape)

        # apply rotary embeddings
        q, k = self.rotary_pos_emb(q, k, kv_sl - sl)

        assert k.struct_info.shape[0] == 1 and v.struct_info.shape[0] == 1
        squeezed_k, squeezed_v = nn.emit(squeeze(k, axis=0)), nn.emit(squeeze(v, axis=0))

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

        kv_sl = all_seq_len_shape.struct_info.values[0]
        bsz, _, n_groups, head_dim = k.struct_info.shape
        kv_cache_shape = R.shape([kv_sl, n_groups, head_dim])
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

        n_rep = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
        kv_attn_shape = R.shape(
            [
                bsz,
                kv_sl,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            ]
        )
        k, v = _repeat_kv(k, v, n_rep, kv_attn_shape)

        # core attention computation
        context_layer = self.core_attention(q, k, v, attention_mask)

        # apply output projection
        output = self.dense(context_layer)

        return output, past_key_value


class MLP(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.dtype = config.dtype

        self.dense_h_to_4h = Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            config.dtype,
            bias=config.add_bias_linear,
        )

        def swiglu(x: relax.Expr):
            x = nn.emit(split(x, 2, axis=-1))
            return nn.emit(silu(x[0]) * x[1])

        self.activation_func = swiglu

        self.dense_4h_to_h = Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            config.dtype,
            bias=config.add_bias_linear,
        )

    def forward(self, hidden_states):
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)

        return hidden_states


class GLMBlock(nn.Module):
    def __init__(self, config: ChatGLMConfig, rotary_pos_emb: RotaryEmbedding):
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            dtype=config.dtype,
            eps=config.layernorm_epsilon,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            dtype=config.dtype,
            eps=config.layernorm_epsilon,
        )

        self.self_attention = SelfAttention(config, rotary_pos_emb)
        self.mlp = MLP(config)

        self.dtype = config.dtype

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: relax.Expr,
    ):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, present_key_value = self.self_attention(
            layernorm_output, all_seq_len_shape, past_key_value, attention_mask
        )

        # residual connection
        layernorm_input = nn.emit(attention_output + hidden_states)

        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)

        # residual connection
        output = nn.emit(mlp_output + layernorm_input)

        return output, present_key_value


class GLMTransformer(nn.Module):
    def __init__(self, config: ChatGLMConfig, rotary_pos_emb: RotaryEmbedding):
        self.num_layers = config.num_layers

        self.layers = ModuleList([GLMBlock(config, rotary_pos_emb) for _ in range(self.num_layers)])
        self.final_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            dtype=config.dtype,
            eps=config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
        attention_mask: relax.Expr,
    ):
        present_kv_cache = []
        for i, block in enumerate(self.layers):
            past_key_value = past_key_values[i * 2], past_key_values[i * 2 + 1]
            hidden_states, (present_k_cache, present_v_cache) = block(
                hidden_states,
                all_seq_len_shape=all_seq_len_shape,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
            )
            present_kv_cache.append(present_k_cache)
            present_kv_cache.append(present_v_cache)
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, present_kv_cache


class ChatGLMModel(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        self.num_layers = config.num_layers

        self.embedding = Embedding(
            num_embeddings=config.padded_vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )

        self.seq_length = config.max_sequence_length
        rotary_dim = config.kv_channels // 2

        self.rotary_pos_emb = RotaryEmbedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            position_embedding_base=10000,
            max_sequence_length=config.max_sequence_length,
            rotary_dim=rotary_dim,
            swizzle_style="glm",
            dtype=config.dtype,
        )
        self.encoder = GLMTransformer(config, self.rotary_pos_emb)
        self.output_layer = Linear(
            in_features=config.hidden_size,
            out_features=config.padded_vocab_size,
            bias=False,
            dtype=config.dtype,
        )

        self.dtype = config.dtype

    def _prepare_decoder_attention_mask(self, input_shape, kv_sl, dtype):
        # create causal mask
        # [bsz, sl] -> [bsz, 1, sl, kv_sl]
        if isinstance(input_shape[-1], tvm.tir.SizeVar) or input_shape[-1] > 1:
            bsz, sl = input_shape

            def min_max_triu_te():
                return te.compute(
                    (sl, sl),
                    lambda i, j: tvm.tir.Select(
                        j > i, tvm.tir.min_value(dtype), tvm.tir.max_value(dtype)
                    ),
                    name="make_diag_mask_te",
                )

            mask = nn.emit_te(min_max_triu_te)
            mask = nn.emit(expand_dims(mask, 0))
            diag_mask = nn.emit(broadcast_to(mask, (bsz, 1, sl, sl)))
            if kv_sl == sl:
                return diag_mask

            def extend_te(x, sl, kv_sl):
                return te.compute(
                    (bsz, 1, sl, kv_sl),
                    lambda b, _, i, j: te.if_then_else(
                        j < kv_sl - sl,
                        tvm.tir.max_value(dtype),
                        x[b, _, i, j - (kv_sl - sl)],
                    ),
                    name="concat_te",
                )

            return nn.emit_te(extend_te, diag_mask, sl, kv_sl)
        else:
            # Get kv_sl from input parameters
            # [bsz, sl=1] -> [bsz, 1, sl=1, kv_sl]
            bsz, sl = input_shape
            mask = relax.op.full(
                (bsz, 1, sl, kv_sl),
                relax.const(tvm.tir.max_value(dtype).value, dtype),
                dtype,
            )
        return nn.emit(mask)

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        batch_size, seq_length = input_ids.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]

        # Token Embeddings
        inputs_embeds = self.embedding(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            dtype=self.dtype,
        )

        hidden_states, present_kv_cache = self.encoder(
            inputs_embeds,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )

        return hidden_states, present_kv_cache


class ChatGLMForCausalLM(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        self.transformer = ChatGLMModel(config)

        self.dtype = config.dtype

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
            _, sl, hs = x.shape
            return te.compute(
                shape=(1, 1, hs),
                fcompute=lambda i, _, k: x[i, sl - 1, k],
                name="slice_last",
            )

        hidden_states = nn.emit_te(
            te_slice_last,
            hidden_states,
            primfunc_name_hint="slice_last",
        )
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

        lm_logits = self.transformer.output_layer(hidden_states)

        if lm_logits.struct_info.dtype != "float32":
            lm_logits = nn.emit(astype(lm_logits, "float32"))

        return lm_logits, key_value_cache


def get_param_quant_kind(name: str, param_info: relax.TensorStructInfo) -> ParamQuantKind:
    if "embedding.weight" in name:
        return ParamQuantKind.embedding_table
    elif "transformer.output_layer.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: ChatGLMConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "prefill"

    bsz = tvm.tir.IntImm("int64", 1)
    sl = tvm.tir.SizeVar("n", "int64")
    all_seq_len = tvm.tir.SizeVar("m", "int64")
    with bb.function(func_name):
        model = ChatGLMForCausalLM(config)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, sl), dtype="int32", name="input_ids")
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo([relax.ObjectStructInfo() for _ in range(config.num_layers * 2)]),
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
    config: ChatGLMConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    bsz = 1
    all_seq_len = tvm.tir.SizeVar("m", "int64")

    with bb.function(func_name):
        model = ChatGLMForCausalLM(config)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo([relax.ObjectStructInfo() for _ in range(config.num_layers * 2)]),
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


def create_kv_cache_func(bb: relax.BlockBuilder, config: ChatGLMConfig) -> None:
    init_shape = relax.ShapeExpr(
        (
            config.max_sequence_length,
            config.multi_query_group_num,
            config.hidden_size // config.num_attention_heads,
        )
    )
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            for _ in range(config.num_layers * 2):
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


def create_softmax_func(bb: relax.BlockBuilder, config: ChatGLMConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder((1, 1, config.padded_vocab_size), dtype="float32", name="logits")
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def get_model(args: argparse.Namespace, hf_config):
    model = args.model
    dtype = args.quantization.model_dtype

    if model.startswith("chatglm2") or model.startswith("codegeex2") or model.startswith("chatglm3"):
        config = ChatGLMConfig(
            **hf_config,
            dtype=dtype,
        )

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

        def f_convert_pname_fwd(pname: str) -> List[str]:
            if "transformer.embedding" in pname:
                return [
                    pname.replace("transformer.embedding", "transformer.embedding.word_embeddings")
                ]
            else:
                return [pname]

        def f_convert_param_bkwd(torch_pname: str, torch_param):
            if "transformer.embedding.word_embeddings" in torch_pname:
                return [
                    (
                        torch_pname.replace(
                            "transformer.embedding.word_embeddings",
                            "transformer.embedding",
                        ),
                        torch_param.astype(dtype),
                    )
                ]
            else:
                return [(torch_pname, torch_param.astype(dtype))]

        param_manager.set_param_loading_func(
            args.model_path, args.use_safetensors, f_convert_pname_fwd, f_convert_param_bkwd
        )
        return mod, param_manager, [None] * len(param_manager.param_names), config

    raise ValueError(f"Unsupported model {model}")
