from typing import List

import argparse
import os

import tvm
from tvm import relax
from tvm.runtime import ShapeTuple
from tvm import rpc
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument
import numpy as np

import torch
from transformers import AutoTokenizer

from mlc_llm import utils


class LibCompare(LibCompareVMInstrument):
    def __init__(self, mod, device, time_eval, skip_rounds=0):
        super().__init__(mod, device, True)
        self.time_eval = time_eval
        self.time_eval_results = {}
        self.visited = set([])
        self.skip_rounds = skip_rounds
        self.atol = 1e-5
        self.rtol = 1e-5

    def skip_instrument(self, func, name, before_run, ret_val, *args):
        print(f"run {name}")
        if name.startswith("shape_func"):
            return True
        if self.counter < self.skip_rounds:
            self.counter += 1
            print(f"[{self.counter}] Skip validating {name}..")
            return True
        if name in self.visited:
            return True
        self.visited.add(name)
        return False

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ):

        super().compare(name, ref_args, new_args, ret_indices)

        if self.time_eval and name not in self.time_eval_results:
            res = self.mod.time_evaluator(name, self.device)(*new_args)
            self.time_eval_results[name] = res
            print(f"Time-eval result {name} on {self.device}: {res}")


class TestState:
    def __init__(self, args):
        self.primary_device = tvm.device(args.primary_device)
        ex = tvm.runtime.load_module(
            os.path.join(
                args.artifact_path,
                f"{args.model}_{args.primary_device}_{args.dtype}.so",
            )
        )
        self.vm = relax.VirtualMachine(ex, self.primary_device)
        if args.cmp_device == "iphone":
            lib_name = f"{args.model}_{args.cmp_device}_{args.dtype}.dylib"
            local_lib_path = os.path.join(args.artifact_path, lib_name)
            proxy_host = os.environ.get("TVM_RPC_PROXY_HOST", "127.0.0.1")
            proxy_port = int(os.environ.get("TVM_RPC_PROXY_PORT", "9090"))
            self.sess = rpc.connect(proxy_host, proxy_port, "iphone")
            self.sess.upload(local_lib_path)
            self.lib = self.sess.load_module(lib_name)
            self.cmp_device = self.sess.metal()
        else:
            self.sess = None
            self.lib = tvm.runtime.load_module(
                os.path.join(
                    args.artifact_path,
                    f"{args.model}_{args.cmp_device}_{args.dtype}.so",
                )
            )
            self.cmp_device = tvm.device(args.cmp_device)
        self.const_params_dict = utils.load_params(
            args.artifact_path, self.primary_device
        )
        self.cmp_instrument = LibCompare(
            self.lib,
            self.cmp_device,
            time_eval=args.time_eval,
            skip_rounds=args.skip_rounds,
        )
        self.vm.set_instrument(self.cmp_instrument)


def deploy_to_pipeline(args) -> None:
    primary_device = tvm.device(args.primary_device)
    const_params = utils.load_params(args.artifact_path, primary_device)
    state = TestState(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Tokenizing...")
    inputs = tvm.nd.array(
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy(),
        primary_device,
    )
    first_sampled_token = tvm.nd.array(
        np.array([[6234]]).astype("int32"), primary_device
    )
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    kv_caches = state.vm["create_kv_cache"]()

    print("Running inference...")
    print("======================= Starts Encoding =======================")
    logits, kv_caches = state.vm["encoding"](
        inputs, seq_len_shape, kv_caches, const_params
    )
    print("======================= Starts Decoding =======================")
    logits, kv_caches = state.vm["decoding"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--primary-device", type=str, default="auto")
    args.add_argument("--cmp-device", type=str, required=True)
    args.add_argument("--prompt", type=str, default="The capital of Canada is")
    args.add_argument("--model", type=str, default="vicuna-v1-7b")
    args.add_argument(
        "--dtype", type=str, choices=["float32", "float16"], default="float16"
    )
    args.add_argument("--time-eval", default=False, action="store_true")
    args.add_argument("--skip-rounds", type=int, default=0)
    parsed = args.parse_args()

    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, parsed.model, parsed.dtype
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
