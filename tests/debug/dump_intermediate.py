import argparse
import os

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer
from tvm import relax
import pickle

from mlc_llm import utils


class DumpInstrument:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.counter = 0

    def __call__(self, func, name, before_run, ret_val, *args):
        if before_run:
            return
        if name.startswith("vm.builtin."):
            return
        if any(not isinstance(x, tvm.nd.NDArray) for x in args):
            return

        print(f"[{self.counter}][{name}]")
        print(*args, sep="\n")
        if self.counter == 6:
            for i, arg in enumerate( args):
                arg = arg.numpy().dump(f"tmp/{i}.pkl")
        self.counter += 1


def print_as_table(sorted_list):
    print(
        "Name".ljust(50)
        + "Time (ms)".ljust(12)
        + "Count".ljust(8)
        + "Total time (ms)".ljust(18)
        + "Percentage (%)"
    )
    total_time = sum([record[1][0] * record[1][1] for record in sorted_list]) * 1000
    for record in sorted_list:
        time = record[1][0] * 1000
        weighted_time = time * record[1][1]
        percentage = weighted_time / total_time * 100
        print(
            record[0].ljust(50)
            + "{:.4f}".format(time).ljust(12)
            + str(record[1][1]).ljust(8)
            + "{:.4f}".format(weighted_time).ljust(18)
            + "{:.2f}".format(percentage)
        )
    print("Total time: {:.4f} ms".format(total_time))
    print()


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
        self.instrument = DumpInstrument(verbose=True)
        self.vm.set_instrument(self.instrument)


def deploy_to_pipeline(args) -> None:
    primary_device = tvm.device(args.primary_device)
    const_params = utils.load_params(args.artifact_path, primary_device)
    state = TestState(args)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.artifact_path, "params"), trust_remote_code=True
    )

    print("Tokenizing...")
    inputs = (
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    )
    first_sampled_token = tvm.nd.array(
        np.array([[6234]]).astype("int32"), primary_device
    )
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    kv_caches = state.vm["create_kv_cache"]()

    print("Running inference...")
    print("======================= Starts Encoding =======================")

    try:
        prefill_func = state.vm["prefill"]
    except AttributeError:
        prefill_func = None

    if inputs.shape[1] > 1 and prefill_func:
        inputs = tvm.nd.array(inputs.numpy(), device=primary_device)
        logits, kv_caches = prefill_func(inputs, seq_len_shape, kv_caches, const_params)
    else:
        for i in range(inputs.shape[1]):
            input_slice = tvm.nd.array(inputs[:, i : i + 1], device=primary_device)
            logits, kv_caches = state.vm["decode"](
                input_slice, seq_len_shape, kv_caches, const_params
            )

    print("======================= Starts Decoding =======================")
    logits, kv_caches = state.vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--primary-device", type=str, default="auto")
    args.add_argument("--prompt", type=str, default="The capital of Canada is")
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
