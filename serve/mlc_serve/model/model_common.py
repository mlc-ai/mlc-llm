from typing import List, Union, Optional

import structlog
import numpy as np
import torch
import tvm

from .paged_cache_manager import CacheManager
from ..engine import (
    SamplingType,
    SamplingParams,
)

LOG = structlog.stdlib.get_logger(__name__)


def get_gpu_memory(gpu: int = 0) -> int:
    return torch.cuda.get_device_properties(gpu).total_memory


def get_num_cache_blocks(
    model,
    seq_lens,
    num_layers,
    num_kv_heads,
    head_size,
    gpu_memory_utilization=0.9,  # the default used by vllm
):
    used_memory_bytes = model.profile_memory_usage(seq_lens)
    cache_block_size = CacheManager.get_cache_block_size(
        num_layers, num_kv_heads, head_size
    )
    total_vram = get_gpu_memory()
    return int(
        (total_vram * gpu_memory_utilization - used_memory_bytes) // cache_block_size
    )


def _apply_top_p_top_k(logits, top_ps, top_ks):
    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort, dim=-1, index=torch.argsort(logits_idx, dim=-1))
    return logits


def sample(
    logits: Union[tvm.nd.NDArray, torch.Tensor],
    sampling_params: List[SamplingParams],
    vocab_size: int,
    check_safety=False,
) -> Optional[np.ndarray]:
    def _is_safe_to_sample(prob_like):
        return (
            torch.sum(torch.isnan(prob_like) | torch.isinf(prob_like) | (prob_like < 0))
            == 0
        )

    torch.cuda.nvtx.range_push(f"sample {logits.shape}")
    logits = torch.from_dlpack(logits)
    num_seq = len(sampling_params)

    mask_random = torch.tensor(
        [p.sampling_type == SamplingType.RANDOM for p in sampling_params],
        dtype=torch.bool,
    )
    mask_greedy = torch.logical_not(mask_random)

    logits_greedy = logits[mask_greedy]

    if logits_greedy.shape[0] > 0:
        res_greedy = torch.argmax(logits_greedy, -1).cpu().numpy()

        if logits_greedy.shape[0] == num_seq:
            torch.cuda.nvtx.range_pop()
            return res_greedy

    temperatures = []
    top_ps = []
    top_ks = []
    divide_by_temperature = False
    do_top_p = False
    do_top_k = False

    for i in range(num_seq):
        param = sampling_params[i]
        freq = param.appeared_tokens_freq

        if param.sampling_type == SamplingType.RANDOM:
            temperatures.append(param.temperature)
            top_ps.append(param.top_p)
            top_ks.append(param.top_k if param.top_k != -1 else vocab_size)

            divide_by_temperature |= temperatures[-1] != 1.0
            do_top_p |= top_ps[-1] < 1.0
            do_top_k |= top_ks[-1] != vocab_size

            # TODO(vvchernov): need to strictly define order of using penalties and logit bias or
            # prohibit simultaneous using of them. At the latter case it can be LogitProcessor
            if (
                not param.presence_penalty == 0.0 or not param.frequency_penalty == 0
            ) and bool(freq):
                index = torch.from_numpy(np.array(list(freq.keys()))).to(
                    device=logits.device
                )
                src = (
                    torch.from_numpy(np.array(list(freq.values())))
                    .type_as(logits)
                    .to(device=logits.device)
                )
                logits[i][index] -= (
                    src * param.frequency_penalty + param.presence_penalty
                )

            if not param.repetition_penalty == 1.0 and bool(freq):
                index = torch.from_numpy(np.array(list(freq.keys()))).to(
                    device=logits.device
                )
                logits[i][index] /= param.repetition_penalty

            if param.logit_bias:
                logits[i][param.logit_bias_index] += (
                    torch.Tensor(param.logit_bias_value)
                    .type_as(logits)
                    .to(device=logits.device)
                )

    logits_random = logits[mask_random]

    if divide_by_temperature:
        t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
        logits_random.div_(t.unsqueeze(dim=1))

    if do_top_p or do_top_k:
        logits_random = _apply_top_p_top_k(logits_random, top_ps, top_ks)

    probs = torch.softmax(logits_random, dim=-1)

    if check_safety and not _is_safe_to_sample(probs):
        torch.cuda.nvtx.range_pop()
        return None

    res_random = torch.multinomial(probs, 1, True).cpu().numpy()[:, 0]

    if logits_random.shape[0] == num_seq:
        torch.cuda.nvtx.range_pop()
        return res_random

    res = np.empty((num_seq,), dtype=np.int32)
    res[mask_random] = res_random

    if logits_greedy.shape[0] > 0:
        res[mask_greedy] = res_greedy

    torch.cuda.nvtx.range_pop()
    return res


def prepare_inputs(
    sequence_ids,
    all_token_ids,
    prompt_lens,
    all_slot_mappings,
    all_decode_block_tables,
    sliding_window,
    is_prefill,
):
    block_tables = []
    seq_lens = []
    input_ids = []
    slot_mapping = []
    positions = []
    max_num_blocks_per_seq = 0
    indices_within_window = []
    start_idx = 0

    for i, (sequence_id, token_ids) in enumerate(zip(sequence_ids, all_token_ids)):
        if is_prefill:
            input_ids += token_ids
            prompt_len = len(token_ids)
            seq_lens.append(prompt_len)
            positions += range(prompt_len)
            slot_mapping += all_slot_mappings[sequence_id]

            if sliding_window:
                indices_within_window += range(
                    start_idx + max(0, prompt_len - sliding_window),
                    start_idx + prompt_len,
                )
                start_idx += prompt_len

        else:
            input_ids.append(token_ids[-1])
            seq_len = prompt_lens[i] + len(token_ids)
            positions.append(seq_len - 1)
            block_table = all_decode_block_tables[sequence_id]
            max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
            block_tables.append(block_table.get_blocks())
            slot_mapping.append(all_slot_mappings[sequence_id][-1])

            if sliding_window:
                seq_lens.append(min(seq_len, sliding_window))
            else:
                seq_lens.append(seq_len)

    def to_torch(arr, torch_dtype):
        return torch.tensor(arr, dtype=torch_dtype, device="cuda")

    input_ids = to_torch(input_ids, torch.int)
    positions = to_torch(positions, torch.int)
    seq_lens = to_torch(seq_lens, torch.int)
    slot_mapping = to_torch(slot_mapping, torch.int)

    if is_prefill and sliding_window:
        indices_within_window = to_torch(indices_within_window, torch.int)
    else:
        indices_within_window = None

    if not is_prefill:

        def _pad_to_max(x: List[int], max_len: int) -> List[int]:
            return x + [0] * (max_len - len(x))

        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in block_tables
        ]
        block_tables = to_torch(padded_block_tables, torch.int)
    else:
        block_tables = None

    return (
        input_ids,
        positions,
        seq_lens,
        slot_mapping,
        indices_within_window,
        block_tables,
    )
