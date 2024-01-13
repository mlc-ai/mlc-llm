from typing import Optional, Tuple

from dataclasses import dataclass

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.op import ccl, reshape, expand_dims, concat, zeros, take, concat
from tvm.relax.op.nn import attention_var_len
from tvm.relax.testing import nn
from tvm.ir import VDevice
from tvm.script import relax as R
from tvm.script.ir_builder import tir as T

from ..quantization import QuantizationScheme
from .modules import ModuleList
from .param_manager import ParamManager
from .llama import (
    LlamaConfig,
    MixtralConfig,
    Linear,
    Embedding,
    LlamaRMSNorm,
    LlamaAttentionBase,
    LlamaDecoderLayer,
    get_param_quant_kind,
    setup_params,
    rotary_modulate_by_freq,
)


def apply_rotary_pos_emb(q, k, positions, position_embedding_base):
    def f_rotary_embedding(tensor, pos_tensor):
        def rotary_compute(*idx):
            pos = pos_tensor[idx[0]].astype("float32")
            return rotary_modulate_by_freq(
                tensor,
                idx,
                pos,
                position_embedding_base,
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(f_rotary_embedding, q, positions, primfunc_name_hint="rotary_embedding")
    k_embed = nn.emit_te(f_rotary_embedding, k, positions, primfunc_name_hint="rotary_embedding")
    return q_embed, k_embed


@dataclass
class EvaluateMultiQueryInput:
    query_start: relax.Expr  # (num_query_token + 1,)
    max_query_len: relax.Expr  # (), must be on CPU
    # The followings are only needed for our naive implementation of multi-query eval
    # with paged KV cache. They can be replaced with block_tables when a proper attention
    # kernel becomes available.
    past_slot_mapping: relax.Expr  # (num_past_token,)
    permute_indices_after_concat: relax.Expr  # (num_past_token + num_query_token,)


class LlamaAttentionBatched(LlamaAttentionBase):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.sliding_window = None

        if config.sliding_window:
            self.sliding_window = T.IntImm("int32", config.sliding_window)
        max_context_length = config.sliding_window or config.max_sequence_length
        partition_size = 512  # partition_size in vLLM attention
        self.max_num_partitions = (max_context_length + partition_size - 1) // partition_size

    def forward(
        self,
        hidden_states: relax.Expr,  # (num_query_token, hidden_size)
        positions: relax.Expr,  # (num_query_token,), for batched RoPE
        seq_lens: relax.Expr,  # (num_seq,)
        kv_cache: Optional[Tuple[relax.Expr, relax.Expr]],
        slot_mapping: Optional[relax.Expr],  # (num_query_token,)
        max_seqlen: Optional[relax.Expr],  # (), must be on CPU
        seq_start: Optional[relax.Expr],  # (num_seq + 1,), for prefill
        block_tables: Optional[relax.Expr],  # (num_seq, max_num_blocks_per_seq), for decode
        indices_within_window: Optional[
            relax.Expr
        ],  # (num_cached_total,), for prefill with sliding-window attention,
        eval_multi_input: Optional[EvaluateMultiQueryInput],
    ):
        num_query_tokens, _ = hidden_states.struct_info.shape

        queries, keys, values = self.project_qkv(
            hidden_states,
            (num_query_tokens, self.num_query_heads, self.head_dim),
            (num_query_tokens, self.num_key_value_heads, self.head_dim),
        )

        queries, keys = apply_rotary_pos_emb(queries, keys, positions, self.position_embedding_base)

        if kv_cache:
            # Paged KV cache update
            k_cache, v_cache = kv_cache

            if indices_within_window:
                # Cache only the most recent keys and values within the window.
                keys_to_cache = nn.emit(take(keys, indices_within_window, axis=0))
                values_to_cache = nn.emit(take(values, indices_within_window, axis=0))
                slot_mapping = nn.emit(take(slot_mapping, indices_within_window, axis=0))
            else:
                # For decode or prefill without sliding window, cache all keys / values.
                keys_to_cache = keys
                values_to_cache = values

            # kv caches are updated inplace, but make it look like a pure operation
            kv = nn.emit(
                relax.op.call_pure_packed(
                    "tvm.contrib.vllm.reshape_and_cache",
                    keys_to_cache,
                    values_to_cache,
                    k_cache,
                    v_cache,
                    slot_mapping,
                    sinfo_args=[k_cache.struct_info, v_cache.struct_info],
                )
            )

            k_cache, v_cache = kv[0], kv[1]
        else:
            k_cache = v_cache = None

        if eval_multi_input:
            assert k_cache and v_cache
            num_kv_head = v_cache.struct_info.shape[1]
            head_size = v_cache.struct_info.shape[2]
            num_past_token = eval_multi_input.past_slot_mapping.struct_info.shape[0]
            kv_shape = (num_past_token, num_kv_head, head_size)
            kv_sinfo = relax.TensorStructInfo(kv_shape, k_cache.struct_info.dtype)

            kv_tensors = nn.emit(
                relax.op.call_pure_packed(
                    "tvm.contrib.vllm.reconstruct_from_cache",
                    k_cache,
                    v_cache,
                    eval_multi_input.past_slot_mapping,
                    sinfo_args=[kv_sinfo, kv_sinfo],
                )
            )
            keys_past, values_past = kv_tensors[0], kv_tensors[1]
            # Say we have past tokens [P1, P2, P3] and the current ones [C1, C2, C3].
            # Each of P1, C1 etc is a sequence of tokens.
            # After concat, we have [P1, P2, P3, C1, C2, C3], but batched sequences need to
            # be in the format [P1, C1, P2, C2, P3, C3]. This permutation is done by the take
            # op and the provided permutation indices.
            keys = nn.emit(
                take(
                    concat([keys_past, keys]), eval_multi_input.permute_indices_after_concat, axis=0
                )
            )
            values = nn.emit(
                take(
                    concat([values_past, values]),
                    eval_multi_input.permute_indices_after_concat,
                    axis=0,
                )
            )
            seq_start_q = eval_multi_input.query_start
            max_seqlen_q = eval_multi_input.max_query_len
            seq_start_k = seq_start
            max_seqlen_k = max_seqlen
        elif seq_start:
            # prefill
            seq_start_q = seq_start_k = seq_start
            max_seqlen_q = max_seqlen_k = max_seqlen
        else:
            # decode
            seq_start_q = seq_start_k = None
            max_seqlen_q = max_seqlen_k = None

        if seq_start_q:
            # Prefill or multi-query evaluation, batched attention over variable sequence lengths
            attn_output = nn.emit(
                attention_var_len(
                    nn.emit(expand_dims(queries, axis=0)),
                    nn.emit(expand_dims(keys, axis=0)),
                    nn.emit(expand_dims(values, axis=0)),
                    seq_start_q,
                    max_seqlen_q,
                    seq_start_k,
                    max_seqlen_k,
                    causal_mask="BottomRight",
                    window_size=self.sliding_window,
                )
            )
        else:
            # Decode, using vLLM kernel
            exp_sums = nn.emit(
                relax.op.builtin.alloc_tensor(
                    relax.ShapeExpr((num_query_tokens, self.num_query_heads, self.max_num_partitions)),
                    dtype="float32",
                    runtime_device_index=0,
                )
            )
            max_logits = nn.emit(
                relax.op.builtin.alloc_tensor(
                    relax.ShapeExpr((num_query_tokens, self.num_query_heads, self.max_num_partitions)),
                    dtype="float32",
                    runtime_device_index=0,
                )
            )
            tmp_out = nn.emit(
                relax.op.builtin.alloc_tensor(
                    relax.ShapeExpr(
                        (num_query_tokens, self.num_query_heads, self.max_num_partitions, self.head_dim)
                    ),
                    dtype=queries.struct_info.dtype,
                    runtime_device_index=0,
                )
            )
            attn_output = nn.emit(
                relax.op.call_dps_packed(
                    "tvm.contrib.vllm.single_query_cached_kv_attention",
                    [
                        queries,
                        k_cache,
                        v_cache,
                        block_tables,
                        seq_lens,
                        16,  # block_size
                        max_seqlen,
                        exp_sums,
                        max_logits,
                        tmp_out,
                    ],
                    out_sinfo=queries.struct_info,
                )
            )

        attn_output = nn.emit(
            reshape(attn_output, (num_query_tokens, self.num_query_heads * self.head_dim))
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, (k_cache, v_cache)


class LlamaDecoderLayerBatched(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config, False)
        self.self_attn = LlamaAttentionBatched(config)

    def forward(
        self,
        hidden_states: relax.Expr,
        positions: relax.Expr,
        seq_lens: relax.Expr,
        kv_cache: Optional[Tuple[relax.Expr, relax.Expr]],
        slot_mapping: Optional[relax.Expr],
        max_seqlen: Optional[relax.Expr],
        seq_start: Optional[relax.Expr],
        block_tables: Optional[relax.Expr],
        indices_within_window: Optional[relax.Expr],
        eval_multi_input: Optional[EvaluateMultiQueryInput],
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
            seq_start=seq_start,
            block_tables=block_tables,
            indices_within_window=indices_within_window,
            eval_multi_input=eval_multi_input,
        )

        hidden_states = self.post_self_attn(hidden_states, residual)

        return hidden_states, new_kv


def create_seq_start(seq_lens):
    # https://github.com/apache/tvm/issues/15851 for why we need to use Thrust
    cumsum = nn.emit(
        relax.op.call_dps_packed(
            "tvm.contrib.thrust.sum_scan", seq_lens, out_sinfo=seq_lens.struct_info
        )
    )
    return nn.emit(concat([zeros((1,), "int32"), cumsum]))


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        cpu_device: VDevice,
        vocab_size_var: tvm.tir.Var,
        sep_embed: bool = False,
    ):
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

        num_query_heads = config.num_attention_heads // config.num_shards
        num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
        num_queries_per_kv = num_query_heads // num_key_value_heads

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [LlamaDecoderLayerBatched(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps)

        self.cpu_device = cpu_device

    def forward(
        self,
        inputs: relax.Expr,
        positions: relax.Expr,
        seq_lens: relax.Expr,
        kv_caches: Optional[relax.Expr],
        slot_mapping: Optional[relax.Expr],
        seq_start: Optional[relax.Expr],
        block_tables: Optional[relax.Expr],
        indices_within_window: Optional[relax.Expr],
        query_lens: Optional[relax.Expr],
        past_slot_mapping: Optional[relax.Expr],
        permute_indices_after_concat: Optional[relax.Expr],
    ):
        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(inputs)
        else:
            inputs_embeds = inputs

        hidden_states = inputs_embeds

        # max_seqlen needs to be on CPU, so that vLLM and Flash Attention can directly get the
        # integer length by max_seqlen->data[0]. Otherwise, we need to repeatedly do cudaMemcpy
        # of a single int32.
        max_seqlen = R.to_vdevice(R.max(seq_lens), self.cpu_device)

        new_kvs = ()

        if query_lens:
            max_query_len = R.to_vdevice(R.max(query_lens), self.cpu_device)
            query_start = create_seq_start(query_lens)
            eval_multi_input = EvaluateMultiQueryInput(
                query_start, max_query_len, past_slot_mapping, permute_indices_after_concat
            )
        else:
            eval_multi_input = None

        for idx, decoder_layer in enumerate(self.layers):
            if kv_caches:
                cache = (kv_caches[2 * idx], kv_caches[2 * idx + 1])
            else:
                cache = None

            hidden_states, new_kv = decoder_layer(
                hidden_states,
                positions,
                seq_lens,
                cache,
                slot_mapping,
                max_seqlen,
                seq_start,
                block_tables,
                indices_within_window,
                eval_multi_input,
            )
            new_kvs += new_kv

        return self.norm(hidden_states), new_kvs


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        cpu_device: VDevice,
        vocab_size_var: tvm.tir.SizeVar,
        sep_embed: bool = False,
    ):
        self.num_shards = config.num_shards
        self.model = LlamaModel(config, cpu_device, vocab_size_var, sep_embed)
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
        input_ids: relax.Expr,  # (num_query_token,)
        positions: relax.Expr,  # (num_query_token,), for batched RoPE
        seq_lens: relax.Expr,  # (num_seq,)
        kv_caches: Optional[relax.Expr],  # For prefill and decode, not needed for evaluate
        slot_mapping: Optional[relax.Expr],  # (num_query_token,), Not needed for evaluate
        block_tables: Optional[relax.Expr],  # (num_seq, max_num_blocks_per_seq), for decode
        indices_within_window: Optional[
            relax.Expr
        ],  # (num_cached_total,), for prefill with sliding-window attention
        query_lens: Optional[relax.Expr],
        past_slot_mapping: Optional[relax.Expr],
        permute_indices_after_concat: Optional[relax.Expr],
    ):
        """
        In vLLM, the paged KV cache is simply a pair of tensors, one for keys and the other
        for values. The tensor has shape (num_blocks, num_kv_heads, head_size, block_size).
        (In practice, the key cache has a slightly different shape for an efficiency reason,
        but that's not important.)

        The mapping between sequences / tokens to blocks is specified by two inputs.
        - block_tables: A list of block IDs allocated for the sequence.
        - slot_mapping: A linear index into the 2D grid (num_blocks, block_size), for each token.

        Support for sliding-window attention is realized by making a block table a circular buffer.
        So the length of a block table for each sequence is at most ceil(window_size / block_size).

        With sliding window, not all past K / V values need to be cached during prefill.
        The last input, indices_within_window, tells which tokens among (num_query_token,) need to have
        their K / V values cached.
        """
        if self.num_shards > 1:
            input_ids = nn.emit(ccl.broadcast_from_worker0(input_ids))
            positions = nn.emit(ccl.broadcast_from_worker0(positions))
            seq_lens = nn.emit(ccl.broadcast_from_worker0(seq_lens))

            if slot_mapping:
                slot_mapping = nn.emit(ccl.broadcast_from_worker0(slot_mapping))

            if block_tables:
                block_tables = nn.emit(ccl.broadcast_from_worker0(block_tables))

            if indices_within_window:
                indices_within_window = nn.emit(ccl.broadcast_from_worker0(indices_within_window))

            if query_lens:
                query_lens = nn.emit(ccl.broadcast_from_worker0(query_lens))
                past_slot_mapping = nn.emit(ccl.broadcast_from_worker0(past_slot_mapping))
                permute_indices_after_concat = nn.emit(
                    ccl.broadcast_from_worker0(permute_indices_after_concat)
                )

        # TODO: Update this condition for evaluate multi
        is_prompt = block_tables is None and query_lens is None
        is_eval_multi = query_lens is not None

        if is_prompt or is_eval_multi:  # prefill and evaluate
            seq_start = create_seq_start(seq_lens)
        else:
            seq_start = None

        hidden_states, new_kvs = self.model(
            input_ids,
            positions,
            seq_lens,
            kv_caches,
            slot_mapping,
            seq_start,
            block_tables,
            indices_within_window,
            query_lens,
            past_slot_mapping,
            permute_indices_after_concat,
        )

        if is_prompt:
            # Extract logits for the last token in each sequence

            def get_logits_last_tokens(x, seq_len_tensor, seq_start):
                return te.compute(
                    shape=(seq_len_tensor.shape[0], x.shape[-1]),
                    fcompute=lambda i, j: x[seq_start[i] + seq_len_tensor[i] - 1, j],
                    name="get_logits_last_tokens",
                )

            logits = self.lm_head(
                nn.emit_te(
                    get_logits_last_tokens,
                    hidden_states,
                    seq_lens,
                    seq_start,
                    primfunc_name_hint="get_logits_last_tokens",
                )
            )
        else:
            logits = self.lm_head(hidden_states)

        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, new_kvs


def get_inputs(
    num_query_token, num_seq, config, max_num_blocks_per_seq=None, sep_embed=False, need_cache=True
):
    hidden_size = config.hidden_size

    inputs = (
        nn.Placeholder((num_query_token, hidden_size), dtype=config.dtype, name="inputs_embeds")
        if sep_embed
        else nn.Placeholder((num_query_token,), dtype="int32", name="input_ids")
    )

    seq_lens = nn.Placeholder((num_seq,), dtype="int32", name="seq_lens")
    positions = nn.Placeholder((num_query_token,), dtype="int32", name="positions")

    if need_cache:
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
            relax.TupleStructInfo(
                [get_cache_sinfo(i) for i in range(config.num_hidden_layers * 2)]
            ),
        )
        slot_mapping = nn.Placeholder((num_query_token,), dtype="int32", name="slot_mapping")
    else:
        past_key_values = None
        slot_mapping = None
        block_tables = None

    if max_num_blocks_per_seq is None:
        block_tables = None
    else:
        block_tables = nn.Placeholder(
            (num_seq, max_num_blocks_per_seq), dtype="int32", name="block_tables"
        )

    return inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables


def create_evaluate_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    """Evaluate logits for the last token in each sequence. Same as prefill but without KV cache."""
    func_name = "evaluate"

    num_query_token = tvm.tir.SizeVar("num_query_token", "int64")
    num_seq = tvm.tir.SizeVar("num_seq", "int64")

    with bb.function(func_name):
        model = LlamaForCausalLM(config, cpu_dev, tvm.tir.Var("vocab_size", "int64"), sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs, positions, seq_lens, _, _, _ = get_inputs(
            num_query_token, num_seq, config, sep_embed=sep_embed
        )

        with bb.dataflow():
            logits, _ = model(
                inputs,
                positions,
                seq_lens,
                kv_caches=None,
                slot_mapping=None,
                block_tables=None,
                indices_within_window=None,
                query_lens=None,
                past_slot_mapping=None,
                permute_indices_after_concat=None,
            )
            params = [
                inputs,
                positions,
                seq_lens,
            ] + model.parameters()
            gv = bb.emit_output(logits)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    """Batched prefill with vLLM paged KV cache.

    The batched attention op is intended to be offloaded to CUTLASS or Flash Attention
    via BYOC.
    """
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    num_query_token = tvm.tir.SizeVar("num_query_token", "int64")
    num_seq = tvm.tir.SizeVar("num_seq", "int64")

    num_inputs = 5

    with bb.function(func_name):
        model = LlamaForCausalLM(config, cpu_dev, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids, positions, seq_lens, past_key_values, slot_mapping, _ = get_inputs(
            num_query_token, num_seq, config, sep_embed=sep_embed
        )

        with bb.dataflow():
            params = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
            ]

            inputs = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
                None,  # block_tables
            ]

            if config.sliding_window:
                num_inputs += 1
                # The value of num_cached_total is between
                # num_query_token (if seq_len < sliding_window for all seq) and
                # num_seq * config.sliding_window (if seq_len > sliding_window for all seq)
                num_cached_total = tvm.tir.Var("num_cached_total", "int64")
                indices_within_window = nn.Placeholder(
                    (num_cached_total,), dtype="int32", name="indices_within_window"
                )
                inputs.append(indices_within_window)
                params.append(indices_within_window)
            else:
                inputs.append(None)

            inputs += [None, None, None]

            logits, new_kvs = model(*inputs)
            gv = bb.emit_output((logits, relax.Tuple(new_kvs)))

        bb.emit_func_output(gv, params + model.parameters())

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", num_inputs))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
) -> None:
    """Batched decoding with vLLM paged KV cache."""
    func_name = "decode"

    num_seq = tvm.tir.SizeVar("num_seq", "int64")
    max_num_blocks_per_seq = tvm.tir.SizeVar("max_num_blocks_per_seq", "int64")

    with bb.function(func_name):
        inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables = get_inputs(
            num_seq, num_seq, config, max_num_blocks_per_seq
        )

        with bb.dataflow():
            model = LlamaForCausalLM(config, cpu_dev, tvm.tir.SizeVar("vocab_size", "int64"))
            param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

            logits, new_kvs = model(
                inputs,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
                block_tables,
                None,
                None,
                None,
                None,
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


def create_evaluate_multi_query_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "evaluate_multi_query"

    num_query_token = tvm.tir.SizeVar("num_query_token", "int64")
    num_past_token = tvm.tir.SizeVar("num_past_token", "int64")
    num_seq = tvm.tir.SizeVar("num_seq", "int64")
    seq_lens_sum = tvm.tir.SizeVar("seq_lens_sum", "int64")

    num_inputs = 8

    with bb.function(func_name):
        model = LlamaForCausalLM(config, cpu_dev, tvm.tir.Var("vocab_size", "int64"), False)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids, positions, seq_lens, past_key_values, slot_mapping, _ = get_inputs(
            num_query_token, num_seq, config, sep_embed=False
        )

        query_lens = nn.Placeholder((num_seq,), dtype="int32", name="query_lens")

        # Replace them with block_tables when a proper attention kernel becomes available.
        past_slot_mapping = nn.Placeholder(
            (num_past_token,), dtype="int32", name="past_slot_mapping"
        )
        permute_indices_after_concat = nn.Placeholder(
            (seq_lens_sum,), dtype="int32", name="permute_indices_after_concat"
        )

        with bb.dataflow():
            params = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
            ]

            inputs = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
                None,  # block_tables
                None,  # indices_within_window
            ]

            inputs += [query_lens, past_slot_mapping, permute_indices_after_concat]
            params += [query_lens, past_slot_mapping, permute_indices_after_concat]

            logits, new_kvs = model(*inputs)
            gv = bb.emit_output((logits, relax.Tuple(new_kvs)))

        bb.emit_func_output(gv, params + model.parameters())

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", num_inputs))


def get_model(args, hf_config):
    dtype = args.quantization.model_dtype
    sep_embed = False

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

    # The CPU device to copy the result of relax.op.max(seq_lens) to CPU.
    cpu_dev = VDevice("llvm", 0, "global")

    create_evaluate_func(bb, param_manager, config, cpu_dev, args.quantization, sep_embed)
    create_encoding_func(bb, param_manager, config, cpu_dev, args.quantization, sep_embed)
    create_decoding_func(bb, param_manager, config, cpu_dev, args.quantization)
    create_evaluate_multi_query_func(bb, param_manager, config, cpu_dev, args.quantization)

    mod = bb.get()

    mod.update_global_info("vdevice", [cpu_dev])

    if args.build_model_only:
        return mod, param_manager, None, config

    return setup_params(mod, param_manager, dtype, config, args)
