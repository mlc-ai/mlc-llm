# pylint: skip-file
import argparse

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer
from tvm import relax
from tvm.relax.frontend.nn import IOEffect
from tvm.runtime import disco as di

import mlc_chat


def deploy_to_pipeline(args) -> None:
    devices = list(range(args.tensor_parallel_shards))
    print(devices)
    sess = di.ProcessSession(
        num_workers=args.tensor_parallel_shards, entrypoint="mlc_chat.cli.worker"
    )
    sess.init_ccl("nccl", *devices)
    print(args.model_lib)
    mod = sess.load_vm_module(args.model_lib)
    create_kv_cache = mod["create_kv_cache"]

    loader = sess.get_global_func("mlc.loader.LoadMultiGPU")
    params = loader(args.params, mod)
    exit(0)


def _parse_args():
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
