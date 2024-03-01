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

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    plist = []
    size = meta["ParamSize"]
    for pname in param_names:
        plist.append(params[pname])
    return plist


class TestState:
    def __init__(self, args):
        self.primary_device = tvm.device(args.primary_device)
        ex = tvm.runtime.load_module(os.path.join(args.artifact_path, f"model.so"))
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
    const_params = load_params(args.artifact_path, primary_device, param_names)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.artifact_path, "params"), trust_remote_code=True
    )

    print("Tokenizing...")
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    print(inputs)
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), primary_device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    kv_caches = state.vm["_initialize_effect"]()

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
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, default="")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--primary-device", type=str, default="auto")
    args.add_argument("--prompt", type=str, default="What is the meaning of life?")
    args.add_argument("--time-eval", default=False, action="store_true")
    args.add_argument("--skip-rounds", type=int, default=0)
    parsed = args.parse_args()
    # parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    parsed.model = "Phi-2"
    parsed.quantization = "q0f16"
    utils.argparse_postproc_common(parsed)

    # parsed.artifact_path = os.path.join(
    #     parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    # )

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
