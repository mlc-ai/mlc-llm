# pylint: disable=invalid-name,missing-docstring
# Used as reference

import argparse
import json
import os

import numpy as np
import torch
import tvm
from transformers import LlamaTokenizer  # type: ignore[import]
from tvm import relax
from tvm.runtime import ShapeTuple

from mlc_llm import utils

##############################################################
# Test file for e2e Llama with batching enabled by directly
# calling functions in VM.
#
# NOTE: the test will not be runnable until the attention
# compute function is integrated to Llama. This is left as
# an item that we will work on shortly in the future.
##############################################################


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, default="Llama-2-7b-chat-hf-q4f16_1")
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--prompt", type=str, default="What's the meaning of life?")
    args.add_argument("--profile", action="store_true", default=False)
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )
    return parsed


def sample_from_logits(vm, logits, device):
    temperature = 0.7
    top_p = 0.95

    num_sequence = logits.shape[0]
    temperature_arr = tvm.nd.array(np.full((num_sequence,), temperature, dtype="float32"), device)
    probs = vm["softmax_with_temperature"](logits, temperature_arr).numpy()

    sampled_tokens = []
    fsample_top_p_from_prob = tvm.get_global_func("vm.builtin.sample_top_p_from_prob")
    for seq_id in range(num_sequence):
        token = fsample_top_p_from_prob(tvm.nd.array(probs[seq_id]), top_p, np.random.sample())
        sampled_tokens.append(token)
    return sampled_tokens


def deploy_to_pipeline(args) -> None:  # pylint: disable=too-many-locals
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    ex = tvm.runtime.load_module(
        os.path.join(
            args.artifact_path,
            f"{args.model}-{args.quantization.name}-{args.device_name}.so",
        )
    )
    vm = relax.VirtualMachine(ex, device)

    with open(
        os.path.join(args.artifact_path, "params", "mlc-chat-config.json"),
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    assert config["model_category"] == "llama"
    tokenizer = LlamaTokenizer.from_pretrained(
        os.path.join(args.artifact_path, "params"), trust_remote_code=True
    )

    num_sequences = 4
    generated_tokens = [[], [], [], []]
    prompts = [
        "What's the meaning of life?",
        "Introduce the history of Pittsburgh to me.",
        "Write a three-day Seattle travel plan.",
        "What is Alaska famous of?",
    ]
    num_decode_steps = 256

    print("Create KV cache...")
    max_total_seq_len = 16384
    page_size = 16
    kv_cache = vm["create_kv_cache"](ShapeTuple([num_sequences, max_total_seq_len, page_size]))

    fadd_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_add_sequence")
    freset_append_length = tvm.get_global_func(
        "vm.builtin.paged_attention_kv_cache_reset_append_lengths"
    )
    freserve = tvm.get_global_func(
        "vm.builtin.paged_attention_kv_cache_reserve_extra_length_for_append"
    )
    fsync = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_sync_aux_array_to_device")

    for seq_id in range(num_sequences):
        print(f"Process seq {seq_id} for prefill...")
        inputs = tvm.nd.array(
            tokenizer(prompts[seq_id], return_tensors="pt").input_ids.to(torch.int32).numpy(),
            device,
        )
        seq_length = inputs.shape[1]
        embedding = vm["embed"](inputs, const_params)

        seq_id_in_cache = fadd_sequence(kv_cache)
        assert seq_id_in_cache == seq_id

        freset_append_length(kv_cache)
        freserve(kv_cache, seq_id, seq_length)
        fsync(kv_cache)

        print(f"Prefilling seq {seq_id}...")
        logits, _ = vm["prefill_with_embed"](embedding, kv_cache, const_params)

        tokens = sample_from_logits(vm, logits, device)
        assert len(tokens) == 1
        generated_tokens[seq_id].append(tokens[0])

    print("Decoding...")
    for step in range(num_decode_steps):
        inputs = tvm.nd.array(
            np.array(
                [[generated_tokens[seq_id][-1]] for seq_id in range(num_sequences)], dtype="int32"
            ),
            device,
        )
        embedding = vm["embed"](inputs, const_params)
        freset_append_length(kv_cache)
        for seq_id in range(num_sequences):
            freserve(kv_cache, seq_id, 1)
        fsync(kv_cache)

        logits, _ = vm["decode_with_embed"](embedding, kv_cache, const_params)
        tokens = sample_from_logits(vm, logits, device)
        assert len(tokens) == num_sequences

        for seq_id in range(num_sequences):
            generated_tokens[seq_id].append(tokens[seq_id])

    for seq_id in range(num_sequences):
        output = tokenizer.decode(generated_tokens[seq_id])
        print("====================================================================")
        print(f"Prompt {seq_id}: {prompts[seq_id]}")
        print(f"Output: {output}")
        print("\n\n")


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy_to_pipeline(ARGS)
