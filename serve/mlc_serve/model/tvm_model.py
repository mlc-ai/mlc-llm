import math
import os
from typing import List, Union, Tuple, Sequence

import structlog
import numpy as np
import torch
import tvm
from tvm import relax
from tvm.runtime import disco as di

from .base import ModelArtifactConfig
from .paged_cache_manager import KVCache, CacheManager
from .model_common import (
    sample,
    prepare_inputs,
    get_num_cache_blocks,
)

from ..engine import (
    SequenceId,
    PROMPT_SEQEUNCE_INDEX,
    get_prompt_sequence_id,
    MLCServeEngineConfig,
)
from ..engine.model_module import (
    DecodeRequest,
    PrefillRequest,
    TextGenerationResult,
    TextGenerator,
)

LOG = structlog.stdlib.get_logger(__name__)


def load_disco_module(artifact_path, lib_path, num_shards):
    sess = di.ProcessSession(num_workers=num_shards, entrypoint="tvm.exec.disco_worker")
    devices = range(num_shards)
    sess.init_ccl("nccl", *devices)
    module = sess.load_vm_module(lib_path)

    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    metadata_path = os.path.join(artifact_path, "params", "ndarray-cache.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        ndarray_cache_metadata = f.read()

    loader = loader_create(metadata_path, ndarray_cache_metadata, "", module)
    loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAllPresharded")
    params = loader_load(loader)

    return module, params, sess


def copy_to_worker_0(sess: di.Session, host_array):
    x_array = sess.empty(host_array.shape, host_array.dtype)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def broadcast_from_worker_0(sess: di.Session, src, shape, dtype):
    dst = sess.empty(shape, dtype)
    sess.broadcast_from_worker0(src, dst)
    return dst


def get_tvm_model(config, dev):
    LOG.info(f"Loading parameters from {config.model_artifact_path}.")
    lib_path = os.path.join(config.model_artifact_path, config.library_name)

    if config.num_shards == 1:
        ex = tvm.runtime.load_module(lib_path)
        vm = relax.VirtualMachine(ex, dev)

        from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

        _params, _meta = tvmjs.load_ndarray_cache(
            f"{config.model_artifact_path}/params", dev
        )
        params = []
        for i in range(_meta["ParamSize"]):
            params.append(_params[f"param_{i}"])

        return vm.module, params, None

    return load_disco_module(config.model_artifact_path, lib_path, config.num_shards)


def _prepare_inputs(
    sequence_ids,
    all_token_ids,
    prompt_lens,
    all_slot_mappings,
    all_decode_block_tables,
    sliding_window,
    is_prefill,
):
    (
        input_ids,
        positions,
        seq_lens,
        slot_mapping,
        indices_within_window,
        block_tables,
    ) = prepare_inputs(
        sequence_ids,
        all_token_ids,
        prompt_lens,
        all_slot_mappings,
        all_decode_block_tables,
        sliding_window,
        is_prefill,
    )

    if block_tables is not None:
        block_tables = tvm.nd.from_dlpack(block_tables)
    if indices_within_window is not None:
        indices_within_window = tvm.nd.from_dlpack(indices_within_window)

    return (
        tvm.nd.from_dlpack(input_ids),
        tvm.nd.from_dlpack(positions),
        tvm.nd.from_dlpack(seq_lens),
        tvm.nd.from_dlpack(slot_mapping),
        indices_within_window,
        block_tables,
    )


class Model:
    def __init__(
        self,
        config,
        dev,
    ):
        self.mod, self.params, self.disco_session = get_tvm_model(config, dev)
        self.dev = dev
        self.vocab_size = config.vocab_size
        self.sliding_window = config.sliding_window
        self.num_shards = config.num_shards

        if self.sliding_window:
            self.block_sliding_window = self.sliding_window // CacheManager.block_size
        else:
            self.block_sliding_window = None

        if self.disco_session:
            self.copy_cache_blocks_func = self.disco_session.get_global_func(
                "tvm.contrib.vllm.copy_blocks"
            )
        else:
            self.copy_cache_blocks_func = tvm.get_global_func(
                "tvm.contrib.vllm.copy_blocks"
            )

    def get_used_memory(self):
        if self.disco_session:
            params = self.params.debug_get_from_remote(0)

            get_used_memory_func = self.disco_session.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            # For Disco, we explicitly query the device 0.
            peak_memory = get_used_memory_func(
                tvm.device("cuda", 0)
            ).debug_get_from_remote(0)

            # TODO: temp hack to switch the VM allocator to eager recycling mode on all devices
            for i in range(1, self.num_shards):
                get_used_memory_func(tvm.device("cuda", i)).debug_get_from_remote(i)
        else:
            params = self.params

            get_used_memory_func = tvm.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            peak_memory = get_used_memory_func(self.dev)

        param_bytes = sum(
            math.prod(param.shape) * np.dtype(param.dtype).itemsize for param in params
        )

        return peak_memory + param_bytes

    def profile_memory_usage(self, seq_lens):
        input_ids = [0] * sum(seq_lens)
        positions = []

        for s in seq_lens:
            positions += range(s)

        input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), self.dev)
        positions = tvm.nd.array(np.array(positions, dtype="int32"), self.dev)
        seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), self.dev)

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)

        self.mod["evaluate"](input_ids, positions, seq_lens, self.params)

        return self.get_used_memory()

    def generate(
        self,
        requests: Sequence[Union[PrefillRequest, DecodeRequest]],
        cache: KVCache,
    ) -> List[TextGenerationResult]:
        if len(requests) == 0:
            return []

        is_prefill = isinstance(requests[0], PrefillRequest)

        all_token_ids = []
        sampling_params = []
        sequence_ids = []
        prompt_lens = []
        num_sequences = []

        for request in requests:
            if isinstance(request, PrefillRequest):
                sequence_ids.append(get_prompt_sequence_id(request.request_id))
                num_sequences.append(request.num_sequence)
            else:
                sequence_ids.append(request.sequence_id)
                prompt_lens.append(request.prompt_token_counts)

            all_token_ids.append(request.token_ids)
            sampling_params.append(request.sampling_params)

        (
            input_ids,
            positions,
            seq_lens,
            slot_mapping,
            indices_within_window,
            block_tables,
        ) = _prepare_inputs(
            sequence_ids,
            all_token_ids,
            prompt_lens,
            cache.slot_mappings,
            cache.decode_block_tables,
            self.sliding_window,
            is_prefill,
        )

        input_shape = input_ids.shape

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)
            slot_mapping = copy_to_worker_0(self.disco_session, slot_mapping)

        if is_prefill:
            torch.cuda.nvtx.range_push(f"forward prefill {input_shape}")

            if self.sliding_window:
                if self.disco_session:
                    indices_within_window = copy_to_worker_0(
                        self.disco_session, indices_within_window
                    )

                out = self.mod["prefill"](
                    input_ids,
                    positions,
                    seq_lens,
                    cache.cache_blocks,
                    slot_mapping,
                    indices_within_window,
                    self.params,
                )
            else:
                out = self.mod["prefill"](
                    input_ids,
                    positions,
                    seq_lens,
                    cache.cache_blocks,
                    slot_mapping,
                    self.params,
                )

            if self.disco_session:
                logits, _ = out.debug_get_from_remote(0)
            else:
                logits = out[
                    0
                ]  # Ignore returned KV cache since it is updated in-place anyway.
        else:
            torch.cuda.nvtx.range_push(f"forward decode {input_shape}")

            if self.disco_session:
                block_tables = copy_to_worker_0(self.disco_session, block_tables)

            out = self.mod["decode"](
                input_ids,
                positions,
                seq_lens,
                cache.cache_blocks,
                slot_mapping,
                block_tables,
                self.params,
            )

            if self.disco_session:
                logits, _ = out.debug_get_from_remote(0)
            else:
                logits = out[0]

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        if is_prefill and cache.pending_copy_from_to:
            block_mapping = tvm.nd.array(
                np.array(cache.pending_copy_from_to, dtype="int64")
            )

            if self.disco_session:
                block_mapping = broadcast_from_worker_0(
                    self.disco_session,
                    copy_to_worker_0(self.disco_session, block_mapping),
                    block_mapping.shape,
                    "int64",
                )

            self.copy_cache_blocks_func(cache.cache_blocks, block_mapping)
            cache.pending_copy_from_to = []

        try:
            next_tokens = sample(logits, sampling_params, self.vocab_size)
            assert next_tokens is not None
            outputs = []
            for i, (sequence_id, new_token) in enumerate(
                zip(sequence_ids, next_tokens)
            ):
                if not new_token in requests[i].sampling_params.appeared_tokens_freq:
                    requests[i].sampling_params.appeared_tokens_freq[new_token] = 0
                requests[i].sampling_params.appeared_tokens_freq[new_token] += 1
                if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
                    for seq_id in range(num_sequences[i]):
                        outputs.append(
                            TextGenerationResult(
                                sequence_id=SequenceId(sequence_id.request_id, seq_id),
                                generated_tokens=[new_token],
                                error=None,
                            )
                        )
                else:
                    outputs.append(
                        TextGenerationResult(
                            sequence_id=sequence_id,
                            generated_tokens=[new_token],
                            error=None,
                        )
                    )

            return outputs
        except RuntimeError:
            # Fallback to per-token sampling in case some logits values are corrupted.
            outputs = []
            err_msg = (
                "Error from sampling: probability tensor contains either `inf`, `nan`"
                " or element < 0"
            )

            for i, (sequence_id, logits_per_token, sampling_param) in enumerate(
                zip(sequence_ids, torch.from_dlpack(logits), sampling_params)
            ):
                maybe_new_token = sample(
                    torch.unsqueeze(logits_per_token, 0),
                    [sampling_param],
                    self.vocab_size,
                    check_safety=True,
                )

                if maybe_new_token is not None:
                    new_token = maybe_new_token[0]
                    if (
                        not new_token
                        in requests[i].sampling_params.appeared_tokens_freq
                    ):
                        requests[i].sampling_params.appeared_tokens_freq[new_token] = 0
                    requests[i].sampling_params.appeared_tokens_freq[new_token] += 1
                    if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
                        for seq_id in range(num_sequences[i]):
                            outputs.append(
                                TextGenerationResult(
                                    sequence_id=SequenceId(
                                        sequence_id.request_id, seq_id
                                    ),
                                    generated_tokens=[new_token],  # type: ignore
                                    error=None,
                                )
                            )
                    else:
                        outputs.append(
                            TextGenerationResult(
                                sequence_id=sequence_id,
                                generated_tokens=[new_token],  # type: ignore
                                error=None,
                            )
                        )
                else:
                    if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
                        for seq_id in range(num_sequences[i]):
                            outputs.append(
                                TextGenerationResult(
                                    sequence_id=SequenceId(
                                        sequence_id.request_id, seq_id
                                    ),
                                    generated_tokens=[],
                                    error=err_msg,
                                )
                            )
                    else:
                        outputs.append(
                            TextGenerationResult(
                                sequence_id=sequence_id,
                                generated_tokens=[],
                                error=err_msg,
                            )
                        )

            return outputs


def init_tvm_model(
    model_artifact_config: ModelArtifactConfig, engine_config: MLCServeEngineConfig
) -> Tuple[TextGenerator, CacheManager]:
    dev = tvm.device("cuda", 0)

    model = Model(model_artifact_config, dev)

    if model_artifact_config.num_shards > 1:
        model.disco_session.sync_worker_0()

    num_kv_heads = (
        model_artifact_config.num_key_value_heads // model_artifact_config.num_shards
    )
    head_size = (
        model_artifact_config.hidden_size // model_artifact_config.num_attention_heads
    )

    if engine_config.max_num_batched_tokens > 0:
        LOG.info("Running memory profiling.")
        num_blocks = get_num_cache_blocks(
            model,
            [engine_config.max_input_len] * engine_config.max_num_sequences,
            model_artifact_config.num_hidden_layers,
            num_kv_heads,
            head_size,
        )
    else:
        num_blocks = 500

    num_cache_slots = num_blocks * CacheManager.block_size

    if num_cache_slots <= engine_config.max_num_batched_tokens:
        raise RuntimeError(
            f"max_num_batched_tokens = {engine_config.max_num_batched_tokens} but"
            f" only {num_blocks} cache blocks can be allocated. The number of"
            f" available cache slots is {num_cache_slots}, not enough for"
            f" {engine_config.max_num_batched_tokens} tokens. Try reducing"
            " --max_input_len or --max_num_sequences."
        )

    LOG.info(f"Using {num_blocks} cache blocks.")

    if model.disco_session:
        init_cache_func = model.disco_session.get_global_func(
            "tvm.contrib.vllm.allocate_kv_cache"
        )
    else:
        init_cache_func = tvm.get_global_func("tvm.contrib.vllm.allocate_kv_cache")

    cache_blocks = init_cache_func(
        head_size,
        model_artifact_config.num_hidden_layers,
        num_kv_heads,
        CacheManager.block_size,
        num_blocks,
    )

    cache_manager = CacheManager(
        cache_blocks,
        num_blocks,
        model_artifact_config.sliding_window,
    )

    LOG.info("Allocated KV cache blocks.")

    return model, cache_manager
