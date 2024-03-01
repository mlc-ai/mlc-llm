import argparse
import os
import pickle

import numpy as np
import torch
import tvm
from mlc_llm import utils
from transformers import AutoTokenizer
from tvm import relax


class TestState:
    def __init__(self, args):
        self.primary_device = tvm.device(args.primary_device)
        ex = tvm.runtime.load_module(
            os.path.join(
                args.artifact_path,
                f"{args.model}-{args.quantization.name}-{args.primary_device}.so",
            )
        )
        self.vm = relax.VirtualMachine(ex, self.primary_device)
        self.sess = None


def deploy_to_pipeline(args) -> None:
    @tvm.register_func("debug_print")
    def debug_print(dummy_object, info, a):
        print(a.numpy())
        return dummy_object

    primary_device = tvm.device(args.primary_device)
    const_params = utils.load_params(args.artifact_path, primary_device)
    state = TestState(args)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.artifact_path, "params"), trust_remote_code=True
    )

    print("Tokenizing...")
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), primary_device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    dummy_obj = tvm.nd.array(np.array([[6234]]).astype("int32"), primary_device)
    kv_caches = state.vm["create_kv_cache"]()

    print("Running inference...")
    print("======================= Starts Encoding =======================")

    try:
        prefill_func = state.vm["prefill"]
    except AttributeError:
        prefill_func = None

    if inputs.shape[1] > 1 and prefill_func:
        inputs = tvm.nd.array(inputs, device=primary_device)
        logits, kv_caches, dummy_obj = prefill_func(
            inputs, seq_len_shape, kv_caches, dummy_obj, const_params
        )
    else:
        for i in range(inputs.shape[1]):
            input_slice = tvm.nd.array(inputs[:, i : i + 1], device=primary_device)
            logits, kv_caches = state.vm["decode"](
                input_slice, seq_len_shape, kv_caches, const_params
            )

    exit(0)
    print("======================= Starts Decoding =======================")
    logits, kv_caches, dummy_obj = state.vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, dummy_obj, const_params
    )


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--primary-device", type=str, default="auto")
    args.add_argument("--prompt", type=str, default="What is the meaning of life?")
    args.add_argument("--time-eval", default=False, action="store_true")
    args.add_argument("--skip-rounds", type=int, default=0)
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)

    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )

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
