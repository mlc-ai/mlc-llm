"""Debug a model by printing out argument information before and after each function."""

import argparse
import json
import os

import numpy as np
import torch
import tvm
from mlc_llm import utils
from transformers import AutoTokenizer
from tvm import relax
from tvm.runtime import ShapeTuple

# pylint: disable=redefined-outer-name


def _extract_metadata(model_lib):
    # pylint: disable=import-outside-toplevel
    from tvm.runtime import device, load_module
    from tvm.runtime.relax_vm import VirtualMachine

    # pylint: enable=import-outside-toplevel

    return json.loads(VirtualMachine(load_module(model_lib), device("cpu"))["_metadata"]())


class DumpInstrument:  # pylint: disable=too-few-public-methods
    """Defines what to do before and after each function."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.counter = 0
        self.first_nan_occurred = False
        self.first_inf_occurred = False

    def __call__(self, func, name, before_run, ret_val, *args):
        # Determine what functions to look at
        if before_run:  # Whether before the function is called or after
            return
        # if self.first_nan_occurred:
        #     return
        # if self.first_inf_occurred:
        #     return
        if name.startswith("vm.builtin."):
            return
        if any(not isinstance(x, tvm.nd.NDArray) for x in args):
            return

        # Decide what to print or save about the function's arguments (where args[-1] is the
        # buffer we write the result to)
        func_name = (
            f"f{self.counter}_before_{name}" if before_run else f"f{self.counter}_after_{name}"
        )
        print(func_name)

        # Write your own behavior below. For example, we can count the number of INF/NaN in args[-1]
        num_nans = np.sum(np.isnan(args[-1].numpy()))
        num_infs = np.sum(np.isinf(args[-1].numpy()))
        if num_nans > 0:
            print(f"has NaN: {num_nans}")
            self.first_nan_occurred = True
        if num_infs > 0:
            print(f"has INF: {num_infs}")
            self.first_inf_occurred = True

        # You can also save the the arguments to experiment offline
        # if self.counter == 769:
        #     for i, ndarray in enumerate(args):
        #         save_name = func_name + f"_arg{i}"
        #         np.save(f"./debug/{save_name}.npy", ndarray.numpy())

        self.counter += 1


def print_as_table(sorted_list):  # pylint: disable=missing-function-docstring
    # pylint: disable=consider-using-f-string
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
    """Embodies the virtual machine and instrument."""

    def __init__(self, args):
        self.primary_device = tvm.device(args.primary_device)
        ex = tvm.runtime.load_module(args.model_lib_path)
        self.vm = relax.VirtualMachine(ex, self.primary_device)
        self.sess = None
        self.instrument = DumpInstrument(verbose=True)
        self.vm.set_instrument(self.instrument)


def deploy_to_pipeline(args) -> None:
    """Main pipeline forst testing; can be modified for specific testing purposes."""
    primary_device = tvm.device(args.primary_device)
    model_metadata = _extract_metadata(args.model_lib_path)
    const_params = utils.load_params_SLM(args.model, primary_device, model_metadata)
    state = TestState(args)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model), trust_remote_code=True)

    print("Tokenizing...")
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    inputs = tvm.nd.array(inputs, device=primary_device)
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), primary_device)

    kv_cache_method: str
    if state.vm.module.implements_function(
        "create_tir_paged_kv_cache"
    ) or state.vm.module.implements_function("create_flashinfer_paged_kv_cache"):
        kv_cache_method = "paged_kv_cache"
        raise NotImplementedError()
    elif state.vm.module.implements_function("create_rnn_state"):
        kv_cache_method = "rnn_state"
        max_num_seq, history = ShapeTuple([1]), ShapeTuple([1])
        kv_caches = state.vm.module["create_rnn_state"](max_num_seq, history)
        f_add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        f_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        f_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    elif state.vm.module.implements_function("_initialize_effect"):
        kv_cache_method = "effect"
        kv_caches = state.vm.module["_initialize_effect"]()
    else:
        raise ValueError("Unknown how to create KVCache")

    def forward(inputs, kv_caches, total_seq_len):
        hidden = state.vm["embed"](inputs, const_params)
        if inputs.shape[1] > 1:
            f_forward = state.vm["prefill"]
        else:
            f_forward = state.vm["decode"]
        if kv_cache_method == "effect":
            logits, kv_caches = f_forward(
                hidden, ShapeTuple([total_seq_len]), kv_caches, const_params
            )
        else:
            seq_ids, input_shape = ShapeTuple([0]), ShapeTuple([inputs.shape[1]])
            f_begin_forward(kv_caches, seq_ids, input_shape)
            logits, kv_caches = f_forward(hidden, kv_caches, const_params)
            f_end_forward(kv_caches)

        return logits, kv_caches

    print("Running inference...")

    print("======================= Starts Prefilling ======================")

    if kv_cache_method != "effect":
        f_add_seq(kv_caches, 0)
    logits, kv_caches = forward(inputs, kv_caches, inputs.shape[1])

    print("======================= Starts Decoding =======================")

    logits, kv_caches = forward(first_sampled_token, kv_caches, inputs.shape[1] + 1)


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, required=True)  # The model weight folder
    args.add_argument("--model-lib-path", type=str, required=True)  # Path to the model library
    args.add_argument("--primary-device", type=str, default="auto")  # Device to run on
    args.add_argument("--prompt", type=str, default="The capital of Canada is")
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
