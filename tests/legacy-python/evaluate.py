# pylint: disable=invalid-name,missing-docstring
# Used as reference

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer, LlamaTokenizer  # type: ignore[import]
from tvm import relax
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument
from tvm.runtime import ShapeTuple

from mlc_llm import utils


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--prompt", type=str, default="The capital of Canada is")
    args.add_argument("--profile", action="store_true", default=False)
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )
    return parsed


class LibCompare(LibCompareVMInstrument):
    def __init__(self, mod, device):
        super().__init__(mod, device, verbose=False)
        self.time_eval_results = {}

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ):
        if name.startswith("shape_func"):
            return
        if name not in self.time_eval_results:
            super().compare(name, ref_args, new_args, ret_indices)
            res = self.mod.time_evaluator(
                name,
                dev=self.device,
                number=100,
                repeat=3,
            )(*new_args).mean
            shapes = [arg.shape for arg in new_args]
            total_bytes = sum(arg.numpy().size * arg.numpy().itemsize for arg in new_args)
            self.time_eval_results[name] = (res, 1, shapes, total_bytes)
        else:
            record = self.time_eval_results[name]
            self.time_eval_results[name] = (
                record[0],
                record[1] + 1,
                record[2],
                record[3],
            )


def print_as_table(sorted_list: List[Tuple[str, Tuple[float, int]]]):
    print(
        "Name".ljust(50)
        + "Time (ms)".ljust(12)
        + "Count".ljust(8)
        + "Total time (ms)".ljust(18)
        + "Pct (%)".ljust(10)
        + "Memory (MB)".ljust(16)
        + "Bandwidth (GB/s)".ljust(18)
        + "Shape"
    )
    total_time = sum(record[1][0] * record[1][1] for record in sorted_list) * 1000
    for record in sorted_list:
        time_used = record[1][0] * 1000
        weighted_time = time_used * record[1][1]
        percentage = weighted_time / total_time * 100
        total_bytes = record[1][3]
        bandwidth = total_bytes / record[1][0] / (1024**3)

        print(
            record[0].ljust(50)
            + f"{time_used:.4f}".ljust(12)
            + str(record[1][1]).ljust(8)
            + f"{weighted_time:.4f}".ljust(18)
            + f"{percentage:.2f}".ljust(10)
            + f"{total_bytes / (1024 * 1024):.2f}".ljust(16)
            + f"{bandwidth:.4f}".format(bandwidth).ljust(18)
            + ", ".join(str(s) for s in record[1][2])
        )
    print(f"Total time: {total_time:.4f} ms")
    print()


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

    if config["model_category"] == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(
            os.path.join(args.artifact_path, "params"), trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.artifact_path, "params"), trust_remote_code=True
        )

    print("Tokenizing...")
    inputs = tvm.nd.array(
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy(),
        device,
    )
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    kv_caches = vm["create_kv_cache"]()
    # skip warm up

    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    logits, kv_caches = vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    device.sync()

    kv_caches = vm["create_kv_cache"]()
    print("Running inference...")
    start = time.time()
    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    device.sync()
    encoding_end = time.time()
    logits, kv_caches = vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    device.sync()
    end = time.time()
    if args.debug_dump:
        fcache_view = tvm.get_global_func("vm.builtin.attention_kv_cache_view")
        first_k_cache = fcache_view(kv_caches[0], ShapeTuple([7, 32, 128]))
        print(f"output kv_cache[0]:\n{first_k_cache.numpy().transpose(1, 0, 2)}")
        print(f"output logits:\n{logits.numpy()}")
    print(
        f"Time elapsed: encoding {(encoding_end - start)} seconds, "
        f"decoding {end - encoding_end} secs"
    )

    if args.profile:
        cmp_instrument = LibCompare(ex, device)
        vm.set_instrument(cmp_instrument)

        print("Profiling...")
        kv_caches = vm["create_kv_cache"]()

        logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
        print("======================= Encoding Profiling =======================")
        print_as_table(
            sorted(
                cmp_instrument.time_eval_results.items(),
                key=lambda x: -(x[1][0] * x[1][1]),
            )
        )
        cmp_instrument.time_eval_results.clear()

        logits, kv_caches = vm["decode"](
            first_sampled_token, second_seq_len_shape, kv_caches, const_params
        )
        print("======================= Decoding Profiling =======================")
        print_as_table(
            sorted(
                cmp_instrument.time_eval_results.items(),
                key=lambda x: -(x[1][0] * x[1][1]),
            )
        )


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy_to_pipeline(ARGS)
