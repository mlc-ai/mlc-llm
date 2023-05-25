# pylint: disable=missing-docstring,invalid-name
import argparse
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple

import tvm
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
            elif tvm.vulkan().exist:
                args.device_name = "vulkan"
            elif tvm.opencl().exist:
                args.device_name = "opencl"
            else:
                raise ValueError("Cannot auto deduce device-name, please set it")
    supported_model_prefix = {
        "vicuna-": ("vicuna_v1.1", "llama"),
        "dolly-": ("dolly", "gpt_neox"),
        "stablelm-": ("stablelm", "gpt_neox"),
        "redpajama-": ("redpajama_chat", "gpt_neox"),
        "moss-": ("moss", "moss"),
        "open-llama-": ("LM", "llama"),
    }
    model = args.model.lower()
    for prefix, (conv_template, model_category) in supported_model_prefix.items():
        if model.startswith(prefix):
            args.conv_template = conv_template
            args.model_category = model_category
            break
    else:
        raise ValueError(
            f'Cannot recognize model "{args.model}". '
            f'Supported ones: {", ".join(supported_model_prefix.keys())}'
        )
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
    mod_transform: tvm.IRModule,
    model_params: List[tvm.nd.NDArray],
) -> List[tvm.nd.NDArray]:
    # Remove the dataflow block inside the param transform function,
    # so that the LazyTransformParams pass can be applied.
    mod_transform = relax.transform.ToNonDataflow()(mod_transform)
    mod_transform = relax.transform.LazyTransformParams()(mod_transform)

    transform_func_name = None
    for gv, func in mod_transform.functions.items():
        if isinstance(func, relax.Function):
            transform_func_name = gv.name_hint
    assert transform_func_name is not None

    target = detect_local_target()
    print(f"Automatically using target for weight quantization: {target}")
    device = tvm.device(target.kind.default_keys[0])

    @tvm.register_func("get_item", override=True)
    def get_item(i):
        gpu_input = tvm.nd.array(model_params[i], device=device)
        return gpu_input

    res = []

    @tvm.register_func("set_item", override=True)
    def set_item(i, value):
        if len(res) <= i:
            res.extend([None for _ in range(i - len(res) + 1)])
        res[i] = tvm.nd.array(value, device=tvm.cpu())
        return tvm.nd.empty((1,), device=device)

    if target.kind.name != "llvm":
        with tvm.target.Target(target):
            mod_transform = tvm.tir.transform.DefaultGPUSchedule()(mod_transform)

    ex = relax.build(mod_transform, target=target)
    vm = relax.vm.VirtualMachine(ex, device)
    vm[transform_func_name]()
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


def get_database(db_paths: str) -> ms.Database:
    db = ms.database.MemoryDatabase()  # pylint: disable=invalid-name
    for db_path in db_paths:
        model_db = ms.database.create(kind="json", work_dir=db_path)
        for record in model_db.get_all_tuning_records():
            db.commit_workload(record.workload.mod)
            db.commit_tuning_record(record)
    return db


def _detect_local_metal():
    dev = tvm.metal()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "metal",
            "max_shared_memory_per_block": 32768,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": 32,
        },
        host=tvm.target.Target(  # TODO: assuming ARM mac for now
            {
                "kind": "llvm",
                "mtriple": "arm64-apple-macos",
                "mcpu": "apple-latest",
            }
        ),
    )


def _detect_local_cuda():
    dev = tvm.cuda()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": 65536,
            "arch": "sm_" + tvm.cuda().compute_version.replace(".", ""),
        }
    )


def _detect_local_vulkan():
    dev = tvm.vulkan()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "vulkan",
            "max_threads_per_block": dev.max_threads_per_block,
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "thread_warp_size": dev.warp_size,
            "supports_float16": 1,
            "supports_int16": 1,
            "supports_16bit_buffer": 1,
        }
    )


def detect_local_target():
    dev = tvm.metal()
    if dev.exist:
        return tvm.target.Target("apple/m1-gpu")

    for method in [
        _detect_local_metal,
        _detect_local_cuda,
        _detect_local_vulkan,
    ]:
        target = method()
        if target is not None:
            return target

    print("Failed to detect local GPU, falling back to CPU as a target")
    return tvm.target.Target("llvm")


def parse_target(args: argparse.Namespace) -> None:
    if not hasattr(args, "target"):
        return
    if args.target == "auto":
        target = detect_local_target()
        if target.host is None:
            target = tvm.target.Target(
                target,
                host="llvm",  # TODO: detect host CPU
            )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "metal":
        target = _detect_local_metal()
        if target is None:
            print("Cannot detect local Apple Metal GPU target! Falling back...")
            target = tvm.target.Target(
                tvm.target.Target(
                    {
                        "kind": "metal",
                        "max_threads_per_block": 256,
                        "max_shared_memory_per_block": 32768,
                        "thread_warp_size": 1,
                    }
                ),
                host=tvm.target.Target(  # TODO: assuming ARM mac for now
                    {
                        "kind": "llvm",
                        "mtriple": "arm64-apple-macos",
                        "mcpu": "apple-latest",
                    }
                ),
            )
        args.target = target
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
    elif args.target in ["iphone", "iphone-dylib", "iphone-tar"]:
        from tvm.contrib import tar, xcode  # pylint: disable=import-outside-toplevel

        if args.target == "iphone-dylib":
            args.export_kwargs = {
                "fcompile": xcode.create_dylib,
                "sdk": "iphoneos",
                "arch": "arm64",
            }
            args.lib_format = "dylib"
        else:
            args.export_kwargs = {"fcompile": tar.tar}
            args.lib_format = "tar"
            args.system_lib = True
            args.system_lib_prefix = f"{args.model}_{args.quantization}_".replace(
                "-", "_"
            )

        @tvm.register_func("tvm_callback_metal_compile")
        def compile_metal(src, target):
            if target.libs:
                return xcode.compile_metal(src, sdk=target.libs[0])
            return xcode.compile_metal(src)

        target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "libs": ["iphoneos"],
                }
            ),
            host="llvm -mtriple=arm64-apple-darwin",
        )
        args.target = target
        args.target_kind = "iphone"
    elif args.target == "vulkan":
        target = _detect_local_vulkan()
        if target is None:
            print("Cannot detect local Vulkan GPU target! Falling back...")
            target = tvm.target.Target(
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
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "webgpu":
        args.target = tvm.target.Target(
            "webgpu",
            host="llvm -mtriple=wasm32-unknown-unknown-wasm",
        )
        args.target_kind = "webgpu"
        args.lib_format = "wasm"
        args.system_lib = True
    elif args.target in ["android", "android-dylib"]:  # android-opencl
        from tvm.contrib import cc, ndk

        if args.target == "android-dylib":
            args.export_kwargs = {
                "fcompile": ndk.create_shared,
            }
            args.lib_format = "so"
        else:
            args.export_kwargs = {
                "fcompile": ndk.create_staticlib,
            }
            args.lib_format = "a"
            args.system_lib = True
        args.target = tvm.target.Target(
            "opencl",
            host="llvm -mtriple=aarch64-linux-android",  # TODO: Only support arm64 for now
        )
        args.target_kind = "android"
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

    print(f"Target configured: {args.target}")
