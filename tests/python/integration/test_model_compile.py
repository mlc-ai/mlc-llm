# pylint: disable=missing-docstring
import concurrent.futures as cf
import os
import shlex
import subprocess
import sys
import tempfile
from itertools import product

import tvm

from mlc_llm.model import MODEL_PRESETS
from mlc_llm.model import MODELS as SUPPORTED_MODELS
from mlc_llm.quantization import QUANTIZATION as SUPPORTED_QUANTS
from mlc_llm.support.constants import MLC_TEMP_DIR

OPT_LEVEL = "O2"
DEVICE2TARGET = {
    "cuda": {
        "kind": "cuda",
        "arch": "sm_86",
        "max_threads_per_block": 1024,
        "max_num_threads": 1024,
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": 32,
    },
    "rocm": {
        "kind": "rocm",
        "mtriple": "amdgcn-amd-amdhsa-hcc",
        "mcpu": "gfx1100",
        "thread_warp_size": 32,
        "max_threads_per_block": 1024,
        "max_num_threads": 256,
        "max_shared_memory_per_block": 65536,
    },
    "vulkan": {
        "kind": "vulkan",
        "max_threads_per_block": 1024,
        "max_num_threads": 256,
        "max_shared_memory_per_block": 32768,
        "thread_warp_size": 1,
        "supports_float32": 1,
        "supports_float16": 1,
        "supports_int64": 1,
        "supports_int32": 1,
        "supports_int16": 1,
        "supports_int8": 1,
        "supports_16bit_buffer": 1,
    },
    "metal": "metal",
    "wasm": "webgpu",
    "android": "android",
    "ios": "iphone",
}
DEVICE2SUFFIX = {
    "cuda": "so",
    "rocm": "so",
    "vulkan": "so",
    "metal": "dylib",
    "wasm": "wasm",
    "android": "tar",
    "ios": "tar",
}
MODELS = list(MODEL_PRESETS.keys())
QUANTS = [  # TODO(@junrushao): use `list(mlc_llm.quantization.QUANTIZATION.keys())`
    "q0f16",
    "q0f32",
    "q3f16_1",
    "q4f16_1",
    "q4f32_1",
    "q4f16_ft",
]
TENSOR_PARALLEL_SHARDS = [
    1,
]


def run_command(log_file, cmd):
    with open(log_file, "w", encoding="utf-8") as file:
        subprocess.check_call(
            cmd,
            stdout=file,
            stderr=subprocess.STDOUT,
        )


def test_model_compile():  # pylint: disable=too-many-locals
    device = sys.argv[1]
    num_workers = int(sys.argv[2])
    target = DEVICE2TARGET[device]
    if not isinstance(target, str):
        target = str(tvm.target.Target(target))
    suffix = DEVICE2SUFFIX[device]

    passed_cmds = []
    failed_cmds = []
    with tempfile.TemporaryDirectory(dir=MLC_TEMP_DIR) as tmp_dir:
        with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
            log_files = []
            cmds = []
            futures = []
            for idx, (model, quant, tp_shard) in enumerate(
                product(
                    MODELS,
                    QUANTS,
                    TENSOR_PARALLEL_SHARDS,
                )
            ):
                if (
                    SUPPORTED_QUANTS[quant].kind
                    not in SUPPORTED_MODELS[MODEL_PRESETS[model]["model_type"]].quantize
                ):
                    continue
                if not target.startswith("cuda") and quant == "q4f16_ft":
                    # FasterTransformer only works with cuda
                    continue
                if "deepseek_v2" in model and "32" in quant:
                    # Skip f32 for deepseek v2 model for now.
                    continue
                log_file = os.path.join(tmp_dir, f"lib{idx}.log")
                cmd = [
                    sys.executable,
                    "-m",
                    "mlc_llm",
                    "compile",
                    model,
                    "--quantization",
                    quant,
                    "--overrides",
                    f"tensor_parallel_shards={tp_shard}",
                    "--device",
                    target,
                    "--opt",
                    OPT_LEVEL,
                    "-o",
                    os.path.join(tmp_dir, f"lib{idx}.{suffix}"),
                ]
                future = executor.submit(run_command, log_file, cmd)
                log_files.append(log_file)
                cmds.append(cmd)
                futures.append(future)
            for log_file, cmd, future in zip(log_files, cmds, futures):
                cmd = shlex.join(cmd)
                try:
                    future.result()
                    passed_cmds.append(cmd)
                    print(f"[PASS] {cmd}")
                except Exception:  # pylint: disable=broad-except
                    failed_cmds.append(cmd)
                    print("-------------------------------")
                    print(f"[FAIL] {cmd}")
                    with open(log_file, "r", encoding="utf-8") as file:
                        print(file.read())
                    print("-------------------------------")
    print("-------------------------------")
    print(f"Total {len(passed_cmds)} passed, {len(failed_cmds)} failed.")
    print("-------------------------------")
    print("Passed commands:")
    for cmd in passed_cmds:
        print(cmd)
    if failed_cmds:
        print("-------------------------------")
        print("Failed commands:")
        for cmd in failed_cmds:
            print(cmd)
        sys.exit(1)


if __name__ == "__main__":
    test_model_compile()
