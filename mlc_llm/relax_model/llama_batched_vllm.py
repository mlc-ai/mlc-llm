from typing import Optional, Tuple

import numpy as np
import tvm
from tvm import relax, te
from tvm.ir import VDevice
from tvm.relax.op import ccl, concat, expand_dims, repeat, reshape, take, zeros
from tvm.relax.op.nn import attention_var_len
from tvm.relax.testing import nn
from tvm.script import relax as R
from tvm.script.ir_builder import tir as T

from ..quantization import QuantizationScheme
from .llama import (
    Embedding,
    Linear,
    LlamaAttentionBase,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    get_param_quant_kind,
    rotary_modulate_by_freq,
    setup_params,
)
from .modules import ModuleList
from .param_manager import ParamManager


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


class LlamaAttentionBatched(LlamaAttentionBase):
    def __init__(self, config: LlamaConfig, head_mapping: relax.Constant):
        super().__init__(config)
        self.head_mapping = head_mapping  # (num_heads,), used by vLLM for multi-query attention
        self.sliding_window = None

        if config.sliding_window:
            self.sliding_window = T.IntImm("int32", config.sliding_window)

    def forward(
        self,
        hidden_states: relax.Expr,  # (num_token, hidden_size)
        positions: relax.Expr,  # (num_token,), for batched RoPE
        seq_lens: relax.Expr,  # (num_seq,)
        kv_cache: Optional[Tuple[relax.Expr, relax.Expr]],
        slot_mapping: Optional[relax.Expr],  # (num_token,)
        max_seqlen: Optional[relax.Expr],  # (), must be on CPU
        seqstart: Optional[relax.Expr],  # (num_seq + 1,), for prefill
        block_tables: Optional[relax.Expr],  # (num_seq, max_num_blocks_per_seq), for decode
        indices_within_window: Optional[
            relax.Expr
        ],  # (num_cached_total,), for prefill with sliding-window attention
    ):
        num_tokens, _ = hidden_states.struct_info.shape

        queries, keys, values = self.project_qkv(
            hidden_states,
            (num_tokens, self.num_query_heads, self.head_dim),
            (num_tokens, self.num_key_value_heads, self.head_dim),
        )

        queries, keys = apply_rotary_pos_emb(queries, keys, positions, self.position_embedding_base)

        if kv_cache:
            # Paged KV cache update
            k_cache, v_cache = kv_cache

            if self.sliding_window is None or block_tables:
                # For decode or prefill without sliding window, cache all keys / values.
                keys_to_cache = keys
                values_to_cache = values
            else:
                # Cache only the most recent keys and values within the window.
                keys_to_cache = nn.emit(take(keys, indices_within_window, axis=0))
                values_to_cache = nn.emit(take(values, indices_within_window, axis=0))
                slot_mapping = nn.emit(take(slot_mapping, indices_within_window, axis=0))

            # kv caches are updated inplace, takes ownership of the arguments
            kv = nn.emit(
                relax.op.call_inplace_packed(
                    "tvm.contrib.vllm.reshape_and_cache",
                    keys_to_cache,
                    values_to_cache,
                    k_cache,
                    v_cache,
                    slot_mapping,
                    inplace_indices=[2, 3],
                    sinfo_args=[k_cache.struct_info, v_cache.struct_info],
                )
            )

            k_cache, v_cache = kv[0], kv[1]
        else:
            k_cache = v_cache = None

        if seqstart:
            # Prefill, batched attention over variable sequence lengths
            attn_output = nn.emit(
                attention_var_len(
                    nn.emit(expand_dims(queries, axis=0)),
                    nn.emit(expand_dims(keys, axis=0)),
                    nn.emit(expand_dims(values, axis=0)),
                    seqstart_q=seqstart,
                    max_seqlen_q=max_seqlen,
                    causal_mask="BottomRight",
                    window_size=self.sliding_window,
                )
            )
        else:
            # Decode, using vLLM kernel
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

        attn_output = nn.emit(
            reshape(attn_output, (num_tokens, self.num_query_heads * self.head_dim))
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, (k_cache, v_cache)


class LlamaDecoderLayerBatched(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, head_mapping: relax.Constant):
        super().__init__(config, False)
        self.self_attn = LlamaAttentionBatched(config, head_mapping)

    def forward(
        self,
        hidden_states: relax.Expr,
        positions: relax.Expr,
        seq_lens: relax.Expr,
        kv_cache: Optional[Tuple[relax.Expr, relax.Expr]],
        slot_mapping: Optional[relax.Expr],
        max_seqlen: Optional[relax.Expr],
        seqstart: Optional[relax.Expr],
        block_tables: Optional[relax.Expr],
        indices_within_window: Optional[relax.Expr],
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
            indices_within_window=indices_within_window,
        )

        hidden_states = self.post_self_attn(hidden_states, residual)

        return hidden_states, new_kv


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        cpu_device: VDevice,
        vocab_size_var: tvm.tir.SizeVar,
        sep_embed: bool = False,
    ):
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

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
                LlamaDecoderLayerBatched(config, head_mapping)
                for _ in range(config.num_hidden_layers)
            ]
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
        seqstart: Optional[relax.Expr],
        block_tables: Optional[relax.Expr],
        indices_within_window: Optional[relax.Expr],
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
                seqstart,
                block_tables,
                indices_within_window,
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
        cache_len = te.var("cached_rotary_embedding_len", "int64")
        self.cos_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="cos_cached")
        self.sin_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="sin_cached")
        ############ End ############

    def forward(
        self,
        input_ids: relax.Expr,  # (num_token,)
        positions: relax.Expr,  # (num_token,), for batched RoPE
        seq_lens: relax.Expr,  # (num_seq,)
        kv_caches: Optional[relax.Expr],  # For prefill and decode, not needed for evaluate
        slot_mapping: Optional[
            relax.Expr
        ],  # (num_token,), for prefill and decode, not needed for evaluate
        block_tables: Optional[relax.Expr],  # (num_seq, max_num_blocks_per_seq), for decode
        indices_within_window: Optional[
            relax.Expr
        ],  # (num_cached_total,), for prefill with sliding-window attention
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
        The last input, indices_within_window, tells which tokens among (num_token,) need to have
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

        is_prompt = block_tables is None

        if is_prompt:  # prefill and evaluate
            # https://github.com/apache/tvm/issues/15851 for why we need to use Thrust
            cumsum = nn.emit(
                relax.op.call_dps_packed(
                    "tvm.contrib.thrust.sum_scan", seq_lens, out_sinfo=seq_lens.struct_info
                )
            )
            seqstart = nn.emit(concat([zeros((1,), "int32"), cumsum]))
        else:
            seqstart = None

        hidden_states, new_kvs = self.model(
            input_ids,
            positions,
            seq_lens,
            kv_caches,
            slot_mapping,
            seqstart,
            block_tables,
            indices_within_window,
        )

        if is_prompt:
            # Extract logits for the last token in each sequence

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


def get_inputs(
    num_token, num_seq, config, max_num_blocks_per_seq=None, sep_embed=False, need_cache=True
):
    hidden_size = config.hidden_size

    inputs = (
        nn.Placeholder((num_token, hidden_size), dtype=config.dtype, name="inputs_embeds")
        if sep_embed
        else nn.Placeholder((num_token,), dtype="int32", name="input_ids")
    )

    seq_lens = nn.Placeholder((num_seq,), dtype="int32", name="seq_lens")
    positions = nn.Placeholder((num_token,), dtype="int32", name="positions")

    if need_cache:
        num_blocks = tvm.tir.SizeVar("num_blocks", "int64")
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
        slot_mapping = nn.Placeholder((num_token,), dtype="int32", name="slot_mapping")
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

    num_token = tvm.tir.SizeVar("num_tokens_excluding_cache", "int64")
    num_seq = tvm.tir.SizeVar("batch_size", "int64")

    with bb.function(func_name):
        model = LlamaForCausalLM(config, cpu_dev, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs, positions, seq_lens, _, _, _ = get_inputs(
            num_token, num_seq, config, sep_embed=sep_embed
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

    num_token = tvm.tir.SizeVar("num_tokens_excluding_cache", "int64")
    num_seq = tvm.tir.SizeVar("batch_size", "int64")

    num_inputs = 5

    with bb.function(func_name):
        model = LlamaForCausalLM(config, cpu_dev, tvm.tir.SizeVar("vocab_size", "int64"), sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids, positions, seq_lens, past_key_values, slot_mapping, _ = get_inputs(
            num_token, num_seq, config, sep_embed=sep_embed
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
                # num_token (if seq_len < sliding_window for all seq) and
                # num_seq * config.sliding_window (if seq_len > sliding_window for all seq)
                num_cached_total = tvm.tir.SizeVar("num_cached_total", "int64")
                indices_within_window = nn.Placeholder(
                    (num_cached_total,), dtype="int32", name="indices_within_window"
                )
                inputs.append(indices_within_window)
                params.append(indices_within_window)
            else:
                inputs.append(None)

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

    num_seq = tvm.tir.SizeVar("batch_size", "int64")
    max_num_blocks_per_seq = tvm.tir.SizeVar("max_num_blocks_per_seq", "int64")

    with bb.function(func_name):
        inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables = get_inputs(
            num_seq, num_seq, config, max_num_blocks_per_seq
        )

        with bb.dataflow():
            model = LlamaForCausalLM(config, cpu_dev, tvm.tir.SizeVar("vocab_size", "int64"))
            param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

            logits, new_kvs = model(
                inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables, None
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
    dtype = args.quantization.model_dtype
    sep_embed = False

    position_embedding_base = 10000

    if "rope_theta" in hf_config:
        position_embedding_base = hf_config["rope_theta"]

    # Llama-2 variants use `max_position_embeddings` to encode maximum sequence length in their hf model cards,
    # while Llama-1 variants use `max_sequence_length`.
    # Thus, use `max_sequence_length` if defined. Otherwise, use `max_position_embeddings`.
    # If none of them is defined, throw an error.
    if "max_sequence_length" in hf_config:
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

    param_manager = ParamManager()
    bb = relax.BlockBuilder()

    # The CPU device to copy the result of relax.op.max(seq_lens) to CPU.
    cpu_dev = VDevice("llvm", 0, "global")

    create_evaluate_func(bb, param_manager, config, cpu_dev, args.quantization, sep_embed)
    create_encoding_func(bb, param_manager, config, cpu_dev, args.quantization, sep_embed)
    create_decoding_func(bb, param_manager, config, cpu_dev, args.quantization)

    mod = bb.get()

    mod.update_global_info("vdevice", [cpu_dev])

    if args.build_model_only:
        return mod, param_manager, None, config

    return setup_params(mod, param_manager, dtype, config, args)
