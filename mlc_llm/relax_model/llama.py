import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.testing import nn
from tvm.script import relax as R


@dataclass
class LlamaConfig:
    def __init__(
        self,
        dtype="float32",
        max_sequence_length=2048,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        position_embedding_base=10000,
        **kwargs,
    ):
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.position_embedding_base = position_embedding_base
        self.kwargs = kwargs


MODEL_CONFIG = {
    "vicuna-v1-7b": {},
    "llama-7b": {},
}


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dtype: str, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            (out_features, in_features), dtype=dtype, name="linear_weight"
        )
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
                return (
                    tir.Cast("float32", x) * tir.Cast("float32", x)
                    if not is_float32
                    else x * x
                )

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
                return x_val / tir.sqrt(
                    square_sum[bsz, i] / x.shape[2] + self.variance_epsilon
                )

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
            f_rms_norm, hidden_states, self.weight, primfunc_name_hint="rms_norm"
        )


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: str):
        self.gate_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)

    def forward(self, x):
        return self.down_proj(relax.op.nn.silu(self.gate_proj(x)) * self.up_proj(x))


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    def f_rotary_embedding(tensor, cos, sin, offset):
        n_feat_half = tensor.shape[-1] // 2

        def rotary_compute(*idx):
            i, j = idx[-3], idx[-1]
            return cos[offset + i, j] * tensor(*idx) + sin[
                offset + i, j
            ] * tvm.tir.Select(
                j >= n_feat_half,
                tensor[idx[0], i, idx[2], j - n_feat_half],
                -tensor[idx[0], i, idx[2], j + n_feat_half],
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(
        f_rotary_embedding, q, cos, sin, offset, primfunc_name_hint="rotary_embedding"
    )
    k_embed = nn.emit_te(
        f_rotary_embedding, k, cos, sin, offset, primfunc_name_hint="rotary_embedding"
    )
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size: int, num_heads: int, dtype: str):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = Linear(
            self.hidden_size, self.num_heads * self.head_dim, dtype=dtype, bias=False
        )
        self.k_proj = Linear(
            self.hidden_size, self.num_heads * self.head_dim, dtype=dtype, bias=False
        )
        self.v_proj = Linear(
            self.hidden_size, self.num_heads * self.head_dim, dtype=dtype, bias=False
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim, self.hidden_size, dtype=dtype, bias=False
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        cos_cached: relax.Expr,
        sin_cached: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[relax.Expr], Optional[Tuple[relax.Expr]]]:
        from tvm.relax.op import astype, matmul, maximum, permute_dims, reshape, squeeze
        from tvm.relax.op.nn import softmax

        bsz, q_len, _ = hidden_states.struct_info.shape
        assert bsz == 1, "Only support batch size 1 at this moment."

        query_states = nn.emit(
            reshape(
                self.q_proj(hidden_states),
                (bsz, q_len, self.num_heads, self.head_dim),
            ),
        )
        key_states = nn.emit(
            reshape(
                self.k_proj(hidden_states),
                (bsz, q_len, self.num_heads, self.head_dim),
            ),
        )
        value_states = nn.emit(
            reshape(
                self.v_proj(hidden_states),
                (bsz, q_len, self.num_heads, self.head_dim),
            ),
        )

        kv_seq_len = all_seq_len_shape.struct_info.values[0]
        offset = kv_seq_len - q_len
        assert query_states.struct_info.dtype == cos_cached.struct_info.dtype
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos_cached, sin_cached, offset=offset
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

        query_states = nn.emit(permute_dims(query_states, [0, 2, 1, 3]))
        key_states = nn.emit(permute_dims(key_states, [0, 2, 1, 3]))
        value_states = nn.emit(permute_dims(value_states, [0, 2, 1, 3]))

        attn_weights = nn.emit(
            matmul(query_states, permute_dims(key_states, [0, 1, 3, 2]))
            / relax.const(math.sqrt(self.head_dim), query_states.struct_info.dtype)
        )

        tvm.ir.assert_structural_equal(
            attn_weights.struct_info.shape.values,
            (bsz, tvm.tir.IntImm("int64", self.num_heads), q_len, kv_seq_len),
        )
        tvm.ir.assert_structural_equal(
            attention_mask.struct_info.shape.values,
            (bsz, tvm.tir.IntImm("int64", 1), q_len, kv_seq_len),
        )
        
        attn_weights = nn.emit(maximum(attn_weights, relax.const(tvm.tir.min_value(attn_weights.struct_info.dtype).value, attn_weights.struct_info.dtype)))
        attn_weights = nn.emit(relax.op.minimum(attn_weights, attention_mask))


        # upcast attention to fp32
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != query_states.struct_info.dtype:
            attn_weights = astype(attn_weights, query_states.struct_info.dtype)
        attn_output = nn.emit(matmul(attn_weights, value_states))

        tvm.ir.assert_structural_equal(
            attn_output.struct_info.shape.values,
            (
                bsz,
                tvm.tir.IntImm("int64", self.num_heads),
                q_len,
                tvm.tir.IntImm("int64", self.head_dim),
            ),
        )

        attn_output = permute_dims(attn_output, [0, 2, 1, 3])
        attn_output = reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)
        return attn_output, ((None, None) if past_key_value is None else past_key_value)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            dtype=config.dtype,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=config.dtype,
        )
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        cos_cached: relax.Expr,
        sin_cached: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            cos_cached=cos_cached,
            sin_cached=sin_cached,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            all_seq_len_shape=all_seq_len_shape,
        )
        hidden_states = nn.emit(residual + hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.emit(residual + hidden_states)

        return hidden_states, present_key_value


def _make_causal_mask(input_ids_shape, dtype, src_len):
    from tvm.relax.op import broadcast_to, full, triu

    bsz, tgt_len = input_ids_shape
    
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


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, dtype=config.dtype
        )
        self.layers = [
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )

    def _prepare_decoder_attention_mask(self, input_shape, src_len, dtype):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if isinstance(input_shape[-1], tvm.tir.Var) or input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, dtype, src_len)
        else:
            # Get src_len from input parameters
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            bsz, tgt_len = input_shape
            combined_attention_mask = nn.emit(relax.op.full((bsz, 1, tgt_len, src_len), relax.const(tvm.tir.max_value(dtype).value, dtype), dtype))
        return combined_attention_mask

    def forward(
        self,
        input_ids: relax.Expr,
        cos_cached: relax.Expr,
        sin_cached: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        # retrieve input_ids
        batch_size, seq_length = input_ids.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]
        inputs_embeds = self.embed_tokens(input_ids)
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
                cos_cached=cos_cached,
                sin_cached=sin_cached,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
            )
            next_decoder_cache += key_value_cache

        hidden_states = self.norm(hidden_states)

        assert len(next_decoder_cache) == len(self.layers) * 2
        return hidden_states, next_decoder_cache


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.model = LlamaModel(config)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, dtype=config.dtype, bias=False
        )

        ############ Rotary embedding constants ############
        assert config.hidden_size % config.num_attention_heads == 0
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos_cached = nn.Parameter(
            (config.max_sequence_length, head_dim),
            dtype=config.dtype,
            name="cos_cached",
        )
        self.sin_cached = nn.Parameter(
            (config.max_sequence_length, head_dim),
            dtype=config.dtype,
            name="sin_cached",
        )
        ############ End ############

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        hidden_states, key_value_cache = self.model(
            input_ids=input_ids,
            cos_cached=self.cos_cached,
            sin_cached=self.sin_cached,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def te_slicing(x: te.Tensor):
            return te.compute(
                shape=(1, 1, x.shape[-1]),
                fcompute=lambda i, j, k: x[i, x.shape[1] - 1, k],
                name="slice",
            )

        logits = self.lm_head(
            nn.emit_te(te_slicing, hidden_states, primfunc_name_hint="slice")
        )
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, key_value_cache


def create_encoding_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    bsz = 1
    seq_len = tvm.tir.Var("n", "int64")
    all_seq_len = tvm.tir.Var("m", "int64")
    with bb.function("encoding"):
        model = LlamaForCausalLM(config)
        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
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
    gv = mod.get_global_var("encoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    bsz = 1
    all_seq_len = tvm.tir.Var("n", "int64")

    with bb.function("decoding"):
        model = LlamaForCausalLM(config)
        input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
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
    gv = mod.get_global_var("decoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_kv_cache_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
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


def get_model(args):
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    model_name = args.model
    model_path = args.model_path
    dtype = args.dtype
    max_seq_len = args.max_seq_len

    if model_name.startswith("vicuna-") or model_name.startswith("llama-"):
        config = LlamaConfig(**MODEL_CONFIG[model_name], dtype=dtype)
        if max_seq_len != -1:
            config.max_sequence_length = max_seq_len

        bb = relax.BlockBuilder()
        create_encoding_func(bb, config)
        create_decoding_func(bb, config)
        create_kv_cache_func(bb, config)
        mod = bb.get()

        param_list = []
        device = tvm.cpu()
        hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        for _, param in hf_model.named_parameters():
            param_list.append(
                tvm.nd.array(param.detach().cpu().numpy().astype(config.dtype), device)
            )
        del hf_model
        head_dim = config.hidden_size / config.num_attention_heads
        inv_freq = 1.0 / (
            config.position_embedding_base
            ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
        )

        t = np.arange(config.max_sequence_length, dtype=inv_freq.dtype)
        freqs = np.einsum("i,j->ij", t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        param_list.append(tvm.nd.array(np.cos(emb).astype(config.dtype), device))
        param_list.append(tvm.nd.array(np.sin(emb).astype(config.dtype), device))

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
        return mod, param_list

    raise ValueError(f"Unsupported model: {model_name}")
