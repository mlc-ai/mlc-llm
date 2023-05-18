# pylint: disable=missing-docstring,invalid-name
import argparse
import os
import shutil
from dataclasses import dataclass
from platform import system
from typing import List, Tuple

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax


@dataclass
class Quantization:
    """Class denoting the quantization options"""

    name: str
    mode: str
    sym: bool
    storage_nbit: int
    model_dtype: str


# Preset compilation modes (configuring quantization and model dtype).
# The value tuple denotes
# (quantization_mode, quantization_sym, quantization_storage_nbit, model_dtype)
quantization_dict = {
    "q3f16_0": Quantization(
        name="q3f16_0", mode="int3", sym=True, storage_nbit=16, model_dtype="float16"
    ),
    "q4f16_0": Quantization(
        name="q4f16_0", mode="int4", sym=True, storage_nbit=32, model_dtype="float16"
    ),
    "q4f32_0": Quantization(
        name="q4f32_0", mode="int4", sym=False, storage_nbit=32, model_dtype="float32"
    ),
    "q0f32": Quantization(
        name="q0f32", mode="no", sym=False, storage_nbit=-1, model_dtype="float32"
    ),
    "q0f16": Quantization(
        name="q0f16", mode="no", sym=False, storage_nbit=-1, model_dtype="float16"
    ),
}

supported_model_types = set(["llama", "gpt_neox", "moss"])


def argparse_postproc_common(args: argparse.Namespace) -> None:
    if hasattr(args, "device_name"):
        if args.device_name == "auto":
            if tvm.cuda().exist:
                args.device_name = "cuda"
            elif tvm.metal().exist:
                args.device_name = "metal"
            else:
                raise ValueError("Cannot auto deduce device-name, please set it")
    if args.model.startswith("vicuna-") or args.model.startswith("llama-"):
        args.conv_template = "vicuna_v1.1"
        args.model_category = "llama"
    elif args.model.startswith("dolly-"):
        args.conv_template = "dolly"
        args.model_category = "gpt_neox"
    elif args.model.startswith("stablelm-"):
        args.conv_template = "stablelm"
        args.model_category = "gpt_neox"
    elif args.model.startswith("RedPajama-"):
        args.conv_template = "redpajama_chat"
        args.model_category = "gpt_neox"
    elif args.model.startswith("moss-"):
        args.conv_template = "moss"
        args.model_category = "moss"
    else:
        raise ValueError(f"Model {args.model} not supported")
    args.quantization = quantization_dict[args.quantization]


def split_transform_deploy_mod(
    mod: tvm.IRModule, model_names: List[str]
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule()

    transform_func_name = None
    gv_names = [gv.name_hint for gv in mod.get_global_vars()]
    for name in model_names:
        if name + "_transform_params" in gv_names:
            transform_func_name = name + "_transform_params"
    assert transform_func_name is not None

    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_deploy[gv] = func
        elif gv.name_hint == transform_func_name:
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    mod_transform = relax.transform.DeadCodeElimination([transform_func_name])(
        mod_transform
    )
    mod_deploy = relax.transform.DeadCodeElimination(model_names)(mod_deploy)

    return mod_transform, mod_deploy


def transform_params(
    mod_transform: tvm.IRModule, model_params: List[tvm.nd.NDArray]
) -> List[tvm.nd.NDArray]:
    transform_func_name = None
    for gv, func in mod_transform.functions.items():
        if isinstance(func, relax.Function):
            transform_func_name = gv.name_hint
    assert transform_func_name is not None

    ex = relax.build(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm.cpu())
    res = vm[transform_func_name](model_params)
    return res


def save_params(params: List[tvm.nd.NDArray], artifact_path: str) -> None:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    meta_data = {}
    param_dict = {}
    meta_data["ParamSize"] = len(params)
    total_size = 0.0
    for i, nd in enumerate(params):
        param_dict[f"param_{i}"] = nd
        np_nd = nd.numpy()
        total_size += np_nd.size * np_nd.dtype.itemsize
    total_size = total_size / 1024.0 / 1024.0 / 1024.0
    print(f"Total param size: {total_size} GB")
    tvmjs.dump_ndarray_cache(
        param_dict, f"{artifact_path}/params", meta_data=meta_data, encode_format="raw"
    )


def load_params(artifact_path: str, device) -> List[tvm.nd.NDArray]:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


def build_model_from_log(relax_mod, target, log_dir):
    db = ms.database.create(work_dir=log_dir)
    with target, db, tvm.transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)
    return relax_mod


def split_static_dynamic_tir(mod: tvm.IRModule):
    def _is_static_shape_buffer(buffer: tvm.tir.Buffer):
        for dim in buffer.shape:
            if not isinstance(dim, tvm.tir.IntImm):
                return False
        return True

    def _is_static_shape_func(func: tvm.tir.PrimFunc):
        for buffer in func.buffer_map.values():
            if not _is_static_shape_buffer(buffer):
                return False
        return True

    mod_dynamic = {}
    mod_static = {}
    for k, v in mod.functions.items():
        if isinstance(v, tvm.tir.PrimFunc):
            if _is_static_shape_func(v):
                mod_static[k] = v
            else:
                mod_dynamic[k] = v
    mod_static = tvm.IRModule(mod_static)
    mod_dynamic = tvm.IRModule(mod_dynamic)
    return mod_static, mod_dynamic


def copy_tokenizer(args: argparse.Namespace) -> None:
    for filename in os.listdir(args.model_path):
        if filename in [
            "tokenizer.model",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ]:
            shutil.copy(
                os.path.join(args.model_path, filename),
                os.path.join(args.artifact_path, "params"),
            )


def get_tokenizer_files(path) -> List[str]:
    tokenizer_set = {
        "tokenizer.model",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    }
    return [x for x in os.listdir(path) if x in tokenizer_set]


def parse_target(args: argparse.Namespace) -> None:
    if not hasattr(args, "target"):
        return
    if args.target == "auto":
        if system() == "Darwin":
            target = tvm.target.Target("apple/m1-gpu")
        else:
            has_gpu = tvm.cuda().exist
            target = tvm.target.Target(
                "cuda"  # TODO: cuda details are required, for example, max shared memory
                if has_gpu
                else "llvm"
            )
        print(f"Automatically configuring target: {target}")
        args.target = tvm.target.Target(target, host="llvm")
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "webgpu":
        args.target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )
        args.target_kind = "webgpu"
        args.lib_format = "wasm"
        args.system_lib = True
    elif args.target.startswith("iphone"):
        from tvm.contrib import cc, xcode  # pylint: disable=import-outside-toplevel

        # override
        @tvm.register_func("tvm_callback_metal_compile")
        def compile_metal(src):
            return xcode.compile_metal(src, sdk="iphoneos")

        dylib = args.target == "iphone-dylib"

        args.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                }
            ),
            host="llvm -mtriple=arm64-apple-darwin",
        )
        args.target_kind = "iphone"
        args.export_kwargs = {
            "fcompile": cc.create_staticlib,
        }
        args.lib_format = "a"

        if dylib:
            args.export_kwargs = {
                "fcompile": xcode.create_dylib,
                "sdk": "iphoneos",
                "arch": "arm64",
            }
            args.lib_format = "dylib"
        else:
            args.system_lib = True
    elif args.target.startswith("android"):
        # android-opencl
        from tvm.contrib import ndk, cc

        args.target = tvm.target.Target(
            "opencl",
            host="llvm -mtriple=aarch64-linux-android",  # Only support arm64 for now
        )
        args.target_kind = "android"
        args.export_kwargs = {
            "fcompile": ndk.create_staticlib,
        }
        args.lib_format = "a"
        args.system_lib = True
    elif args.target == "vulkan":
        args.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "vulkan",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "supports_float16": 1,
                    "supports_int16": 1,
                    "supports_16bit_buffer": 1,
                }
            ),
            host="llvm",
        )
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "metal_x86_64":
        from tvm.contrib import xcode  # pylint: disable=import-outside-toplevel

        args.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                }
            ),
            host="llvm -mtriple=x86_64-apple-darwin",
        )
        args.target_kind = "metal_x86_64"
        args.export_kwargs = {
            "fcompile": xcode.create_dylib,
            "sdk": "macosx",
            "arch": "x86_64",
        }
        args.lib_format = "dylib"
    else:
        args.target = tvm.target.Target(args.target, host="llvm")
        args.target_kind = args.target.kind.default_keys[0]

    # use mingw to cross compile windows
    if hasattr(args, "llvm_mingw") and args.llvm_mingw != "":
        from tvm.contrib.cc import (  # pylint: disable=import-outside-toplevel
            cross_compiler,
        )

        args.export_kwargs = {
            "fcompile": cross_compiler(
                os.path.join(args.llvm_mingw, "bin", "x86_64-w64-mingw32-clang++"),
                output_format="dll",
            ),
        }
        args.target = args.target.with_host("llvm -mtriple=x86_64-w64-windows-gnu")
        args.lib_format = "dll"
