from typing import Optional, Tuple, List

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.op import ccl, reshape, split, expand_dims, concat, zeros, repeat
from tvm.relax.op.nn import attention_var_len
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import QuantizationScheme
from .modules import ModuleList
from .param_manager import ParamManager
from .llama import (
    LlamaConfig,
    Linear,
    Embedding,
    LlamaRMSNorm,
    LlamaMLP,
    get_param_quant_kind,
    setup_params,
    rotary_modulate_by_freq,
)


def apply_rotary_pos_emb(q, k, positions, position_embedding_base, offset: int = 0):
    def f_rotary_embedding(tensor, pos_tensor, offset):
        def rotary_compute(*idx):
            pos = (offset + pos_tensor[idx[0]]).astype("float32")
            return rotary_modulate_by_freq(
                tensor,
                idx,
                pos,
                position_embedding_base,
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(
        f_rotary_embedding, q, positions, offset, primfunc_name_hint="rotary_embedding"
    )
    k_embed = nn.emit_te(
        f_rotary_embedding, k, positions, offset, primfunc_name_hint="rotary_embedding"
    )
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, head_mapping, prefill):
        dtype = config.dtype
        self.num_shards = config.num_shards
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.get_num_key_value_heads() // self.num_shards
        self.num_query_heads = config.num_attention_heads // self.num_shards
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.position_embedding_base = config.position_embedding_base
        self.head_mapping = head_mapping
        self.prefill = prefill

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

    def forward(
        self,
        hidden_states: relax.Expr,
        positions: relax.Expr,
        seq_lens: relax.Expr,
        kv_cache: relax.Expr,
        slot_mapping: relax.Expr,
        max_seqlen: relax.Expr,
        seqstart: relax.Expr,  # only for prefill
        block_tables: relax.Expr,  # only for decode
    ):
        num_tokens, _ = hidden_states.struct_info.shape

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

        queries = nn.emit(
            reshape(
                query_states,
                (num_tokens, self.num_query_heads, self.head_dim),
            ),
        )
        keys = nn.emit(
            reshape(
                key_states,
                (num_tokens, self.num_key_value_heads, self.head_dim),
            ),
        )
        values = nn.emit(
            reshape(
                value_states,
                (num_tokens, self.num_key_value_heads, self.head_dim),
            ),
        )

        queries, keys = apply_rotary_pos_emb(
            queries, keys, positions, self.position_embedding_base, offset=0
        )

        # Paged KV cache update
        k_cache, v_cache = kv_cache

        # kv caches are updated inplace, but make it look like a pure operation
        kv = nn.emit(
            relax.op.call_pure_packed(
                "tvm.contrib.vllm.reshape_and_cache",
                keys,
                values,
                k_cache,
                v_cache,
                slot_mapping,
                sinfo_args=[k_cache.struct_info, v_cache.struct_info],
            )
        )

        k_cache, v_cache = kv[0], kv[1]

        if self.prefill:
            if self.num_key_value_heads != self.num_query_heads:
                # TODO(masahi): If repeats turn out to be expensive, remove them by
                # enabling Flash Attention MQA offload for attention_var_len.
                n_rep = self.num_query_heads // self.num_key_value_heads
                keys = nn.emit(repeat(keys, n_rep, axis=1))
                values = nn.emit(repeat(values, n_rep, axis=1))

            attn_output = nn.emit(
                attention_var_len(
                    nn.emit(expand_dims(queries, axis=0)),
                    nn.emit(expand_dims(keys, axis=0)),
                    nn.emit(expand_dims(values, axis=0)),
                    seqstart_q=seqstart,
                    max_seqlen_q=max_seqlen,
                    causal_mask="BottomRight",
                )
            )
        else:
            attn_output = nn.emit(
                relax.op.call_dps_packed(
                    "tvm.contrib.vllm.single_query_cached_kv_attention",
                    [
                        queries,
                        k_cache,
                        v_cache,
                        self.head_mapping,
                        block_tables,
                        seq_lens,
                        16,  # block_size
                        max_seqlen,
                    ],
                    out_sinfo=queries.struct_info,
                )
            )

        attn_output = nn.emit(reshape(attn_output, (num_tokens, self.num_query_heads * self.head_dim)))
        attn_output = self.o_proj(attn_output)

        return attn_output, (k_cache, v_cache)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, head_mapping, prefill: bool):
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, head_mapping, prefill)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        positions: relax.Expr,
        seq_lens: relax.Expr,
        kv_cache: relax.Expr,
        slot_mapping: relax.Expr,
        max_seqlen: relax.Expr,
        seqstart: relax.Expr,
        block_tables: relax.Expr,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_kv = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            seq_lens=seq_lens,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            max_seqlen=max_seqlen,
            seqstart=seqstart,
            block_tables=block_tables,
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

        return hidden_states, new_kv


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tvm.tir.Var,
        prefill: bool,
        sep_embed: bool = False,
    ):
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None
        self.prefill = prefill

        num_query_heads = config.num_attention_heads // config.num_shards
        num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
        num_queries_per_kv = num_query_heads // num_key_value_heads
        head_mapping = relax.const(
            tvm.nd.array(
                np.repeat(np.arange(num_key_value_heads, dtype="int32"), num_queries_per_kv)
            )
        )

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [
                LlamaDecoderLayer(config, head_mapping, prefill)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs: relax.Expr,
        positions: relax.Expr,
        seq_lens: relax.Expr,
        kv_caches: relax.Expr,
        slot_mapping: relax.Expr,
        seqstart: relax.Expr,
        block_tables: relax.Expr,
    ):
        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(inputs)
        else:
            inputs_embeds = inputs

        hidden_states = inputs_embeds

        max_seqlen = R.max(seq_lens)

        new_kvs = ()

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_kv = decoder_layer(
                hidden_states,
                positions,
                seq_lens,
                (kv_caches[2 * idx], kv_caches[2 * idx + 1]),
                slot_mapping,
                max_seqlen,
                seqstart,
                block_tables,
            )
            new_kvs += new_kv

        return self.norm(hidden_states), new_kvs


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tvm.tir.Var,
        prefill: bool,
        sep_embed: bool = False,
    ):
        self.prefill = prefill
        self.num_shards = config.num_shards
        self.model = LlamaModel(config, vocab_size_var, prefill, sep_embed)
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
        input_ids: relax.Expr,
        positions: relax.Expr,
        seq_lens: relax.Expr,
        kv_caches: relax.Expr,
        slot_mapping: relax.Expr,
        block_tables: relax.Expr,  # only for decode
    ):
        if self.num_shards > 1:
            input_ids = nn.emit(ccl.broadcast_from_worker0(input_ids))
            positions = nn.emit(ccl.broadcast_from_worker0(positions))
            seq_lens = nn.emit(ccl.broadcast_from_worker0(seq_lens))
            slot_mapping = nn.emit(ccl.broadcast_from_worker0(slot_mapping))

            if not self.prefill:
                block_tables = nn.emit(ccl.broadcast_from_worker0(block_tables))

        if self.prefill:
            cumsum = nn.emit(
                relax.op.call_dps_packed(
                    "tvm.contrib.thrust.sum_scan", seq_lens, out_sinfo=seq_lens.struct_info
                )
            )
            seqstart = nn.emit(concat([zeros((1,), "int32"), cumsum]))
        else:
            seqstart = None

        hidden_states, new_kvs = self.model(
            input_ids, positions, seq_lens, kv_caches, slot_mapping, seqstart, block_tables
        )

        if self.prefill:

            def get_logits_last_tokens(x, seq_len_tensor, seqstart):
                return te.compute(
                    shape=(seq_len_tensor.shape[0], x.shape[-1]),
                    fcompute=lambda i, j: x[seqstart[i] + seq_len_tensor[i] - 1, j],
                    name="get_logits_last_tokens",
                )

            logits = self.lm_head(
                nn.emit_te(
                    get_logits_last_tokens,
                    hidden_states,
                    seq_lens,
                    seqstart,
                    primfunc_name_hint="get_logits_last_tokens",
                )
            )
        else:
            logits = self.lm_head(hidden_states)

        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, new_kvs


def get_inputs(num_token, num_seq, config, max_num_blocks_per_seq=None, sep_embed=False):
    hidden_size = config.hidden_size

    inputs = (
        nn.Placeholder((num_token, hidden_size), dtype=config.dtype, name="inputs_embeds")
        if sep_embed
        else nn.Placeholder((num_token,), dtype="int32", name="input_ids")
    )

    seq_lens = nn.Placeholder((num_seq,), dtype="int32", name="seq_lens")
    positions = nn.Placeholder((num_token,), dtype="int32", name="positions")

    num_blocks = tvm.tir.Var("num_blocks", "int64")
    block_size = 16

    vec_size = 8  # 128 bit, fp16 x 8
    num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
    head_size = hidden_size // config.num_attention_heads

    k_cache_shape = (
        num_blocks,
        num_key_value_heads,
        head_size // vec_size,
        block_size,
        vec_size,
    )
    v_cache_shape = (num_blocks, num_key_value_heads, head_size, block_size)

    get_cache_sinfo = lambda i: relax.TensorStructInfo(
        k_cache_shape if i % 2 == 0 else v_cache_shape, dtype="float16"
    )

    past_key_values = relax.Var(
        "kv_cache",
        relax.TupleStructInfo([get_cache_sinfo(i) for i in range(config.num_hidden_layers * 2)]),
    )
    slot_mapping = nn.Placeholder((num_token,), dtype="int32", name="slot_mapping")

    if max_num_blocks_per_seq is None:
        block_tables = None
    else:
        block_tables = nn.Placeholder(
            (num_seq, max_num_blocks_per_seq), dtype="int32", name="block_tables"
        )

    return inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    num_token = tvm.tir.Var("num_token", "int64")
    num_seq = tvm.tir.Var("num_seq", "int64")

    with bb.function(func_name):
        model = LlamaForCausalLM(config, tvm.tir.Var("vocab_size", "int64"), True, sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs, positions, seq_lens, past_key_values, slot_mapping, _ = get_inputs(
            num_token, num_seq, config, sep_embed=sep_embed
        )

        with bb.dataflow():
            logits, new_kvs = model(
                inputs, positions, seq_lens, past_key_values, slot_mapping, None
            )
            params = [
                inputs,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(new_kvs)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 5))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    num_seq = tvm.tir.Var("num_seq", "int64")
    max_num_blocks_per_seq = tvm.tir.Var("max_num_blocks_per_seq", "int64")

    with bb.function(func_name):
        inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables = get_inputs(
            num_seq, num_seq, config, max_num_blocks_per_seq
        )

        with bb.dataflow():
            model = LlamaForCausalLM(config, tvm.tir.Var("vocab_size", "int64"), False)
            param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

            logits, new_kvs = model(
                inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables
            )
            params = [
                inputs,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
                block_tables,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(new_kvs)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 6))


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    sep_embed = False

    position_embedding_base = 10000
    max_position_embeddings = 2048
    if "rope_theta" in hf_config:
        position_embedding_base = hf_config["rope_theta"]
    if "max_position_embeddings" in hf_config:
        max_position_embeddings = hf_config["max_position_embeddings"]

    config = LlamaConfig(
        **hf_config,
        dtype=dtype,
        position_embedding_base=position_embedding_base,
        combine_matmul=True,
        num_shards=args.num_shards,
        build_model_only=args.build_model_only,
    )
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

    param_manager = ParamManager()
    bb = relax.BlockBuilder()

    create_encoding_func(bb, param_manager, config, args.quantization, sep_embed)
    create_decoding_func(bb, param_manager, config, args.quantization)

    mod = bb.get()

    if args.build_model_only:
        return mod, param_manager, None, config

    return setup_params(mod, param_manager, dtype, config, args)
