# pylint: skip-file
import argparse
import json
import os

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer
from tvm import relax
from tvm.relax.frontend.nn import IOEffect
from tvm.runtime import disco as di

import mlc_chat


def load_params(artifact_path: str, device, param_names):
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(artifact_path, device)
    plist = []
    size = meta["ParamSize"]
    for pname in param_names:
        plist.append(params[pname])
    return plist


def CopyToWorker0(sess: di.ProcessSession, host_array):
    array = sess.empty(host_array.shape, host_array.dtype)
    sess.copy_to_worker_0(host_array, array)
    return array


def deploy_to_pipeline(args) -> None:
    model_config = open(os.path.join(args.params, "mlc-chat-config.json")).read()

    devices = list(range(args.tensor_parallel_shards))
    sess = di.ProcessSession(
        num_workers=args.tensor_parallel_shards, entrypoint="mlc_chat.cli.worker"
    )
    sess.init_ccl("nccl", *devices)
    print(args.model_lib)
    mod = sess.load_vm_module(args.model_lib)
    create_kv_cache = mod["_initialize_effect"]
    reset_kv = sess.get_global_func("vm.builtin.attention_kv_cache_array_clear")

    prefill_func = mod["prefill"]

    kv_caches = create_kv_cache()

    # local_kv = kv_caches.debug_get_from_remote(0)
    # print(type(local_kv))
    # print(len(local_kv))
    # exit(0)

    loader = sess.get_global_func("mlc.loader.LoadMultiGPU")
    params = loader(args.params, mod, model_config)

    input_tokens_decode = sess.empty((1, 1), "int32")

    tokenizer = AutoTokenizer.from_pretrained(args.params, trust_remote_code=True)
    prompt_tokens = tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    print(prompt_tokens)
    total_seq_len = prompt_tokens.shape[1]
    prompt_tokens = tvm.nd.array(prompt_tokens)

    sess.sync_worker_0()
    # ForwardTokens
    input_data = CopyToWorker0(sess, prompt_tokens)
    seq_len_shape = tvm.runtime.ShapeTuple([total_seq_len])
    ret = prefill_func(input_data, seq_len_shape, kv_caches, params)

    logits_on_devices = ret.debug_get_from_remote(0)[0]
    print(logits_on_devices)
    total_seq_len += 1
    exit(0)
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])

    print("Running inference...")
    print("======================= Starts Encoding =======================")

    try:
        prefill_func = state.vm["prefill"]
    except AttributeError:
        prefill_func = None

    if inputs.shape[1] > 1 and prefill_func:
        inputs = tvm.nd.array(inputs, device=primary_device)
        logits, kv_caches = prefill_func(inputs, seq_len_shape, kv_caches, const_params)
    else:
        for i in range(inputs.shape[1]):
            input_slice = tvm.nd.array(inputs[:, i : i + 1], device=primary_device)
            logits, kv_caches = state.vm["decode"](
                input_slice, seq_len_shape, kv_caches, const_params
            )

    exit(0)
    print("======================= Starts Decoding =======================")
    logits, kv_caches = state.vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )


def _parse_args():
    # @tvm.register_func("vm.builtin.debug_print")
    # def _print(lineo: str, array) -> None:
    #     print(f"{lineo}: shape = {array.shape}, dtype = {array.dtype}, data =\n{array}")

    args = argparse.ArgumentParser()
    args.add_argument("--model-lib", type=str, required=True)
    args.add_argument("--params", type=str, required=True)
    args.add_argument("--tensor-parallel-shards", type=int, default=4)
    args.add_argument("--primary-device", type=str, default="auto")
    args.add_argument("--prompt", type=str, default="What is the meaning of life?")
    args.add_argument("--time-eval", default=False, action="store_true")
    args.add_argument("--skip-rounds", type=int, default=0)
    parsed = args.parse_args()

    if parsed.primary_device == "auto":
        if tvm.cuda().exist:
            parsed.primary_device = "cuda"
        elif tvm.metal().exist:
            parsed.primary_device = "metal"
        else:
            raise ValueError("Cannot auto deduce device-name, please set it")
    return parsed


if __name__ == "__main__":
    args = _parse_args()
    deploy_to_pipeline(args)
    deploy_to_pipeline(args)
