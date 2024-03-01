import argparse
import os
import pickle

import numpy as np
import torch
import tvm
from mlc_llm import utils
from transformers import AutoTokenizer
from tvm import relax


def load_params(artifact_path: str, device, param_names):
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(artifact_path, device)
    plist = []
    size = meta["ParamSize"]
    for pname in param_names:
        plist.append(params[pname])
    return plist


class TestState:
    def __init__(self, args):
        self.primary_device = tvm.device(args.primary_device)
        ex = tvm.runtime.load_module(args.model_lib)
        self.vm = relax.VirtualMachine(ex, self.primary_device)
        self.sess = None


def deploy_to_pipeline(args) -> None:
    state = TestState(args)
    x = state.vm["_metadata"]()
    metadata = eval(x)
    params = metadata["params"]
    param_names = []
    for param in params:
        param_names.append(param["name"])

    primary_device = tvm.device(args.primary_device)
    const_params = load_params(args.params, primary_device, param_names)
    tokenizer = AutoTokenizer.from_pretrained(args.params, trust_remote_code=True)

    print("Tokenizing...")
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    print(inputs)
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), primary_device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    kv_caches = state.vm["_initialize_effect"]()

    print("Running inference...")
    print("======================= Starts Encoding =======================")

    prefill_func = state.vm["prefill"]

    inputs = tvm.nd.array(inputs, device=primary_device)
    logits, kv_caches = prefill_func(inputs, seq_len_shape, kv_caches, const_params)
    print(logits)

    exit(0)
    print("======================= Starts Decoding =======================")
    logits, kv_caches = state.vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model-lib", type=str, required=True)
    args.add_argument("--params", type=str, required=True)
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
