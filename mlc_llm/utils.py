# pylint: disable=missing-docstring,invalid-name
import argparse
import functools
import json
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Set

import numpy as np
import tvm
from tvm import relax

from .quantization import quantization_schemes
from .relax_model import param_manager

supported_model_types = set(
    [
        "llama",
        "gpt_neox",
        "gpt_bigcode",
        "minigpt",
        "moss",
        "rwkv",
        "gptj",
        "chatglm",
        "mistral",
        "mixtral",
        "stablelm_epoch",
    ]
)


def wrap_tqdm_counter(func, **tqdm_kwargs):
    # tqdm isn't a hard requirement, so return the original function
    # if it isn't available.
    try:
        from tqdm import tqdm
    except ImportError:
        return func

    pbar = tqdm(**tqdm_kwargs)

    @functools.wraps(func)
    def inner(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return inner


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

    model_category_override = {
        "moss-moon-003-sft": "gptj",
        "moss-moon-003-base": "gptj",
        "rwkv-": "rwkv",
        "rwkv_world": "rwkv_world",
        "minigpt": "minigpt",
    }
    try:
        with open(os.path.join(args.model_path, "config.json"), encoding="utf-8") as i_f:
            config = json.load(i_f)
            args.model_category = config["model_type"]
        model_path_lower = args.model_path.lower()
        if "rwkv" in model_path_lower and "world" in model_path_lower:
            args.model_category = "rwkv_world"
    except Exception:
        args.model_category = ""
    model = args.model.lower()
    if "rwkv" in model and "world" in model:
        model = "rwkv_world"
    for prefix, override_category in model_category_override.items():
        if model.startswith(prefix):
            args.model_category = override_category
            break
    assert args.model_category is not None

    model_conv_templates = {
        "llama-2": "llama-2",
        "codellama-7b-instruct": "codellama_instruct",
        "codellama-13b-instruct": "codellama_instruct",
        "codellama-34b-instruct": "codellama_instruct",
        "codellama": "codellama_completion",
        "gpt2": "gpt2",
        "vicuna-": "vicuna_v1.1",
        "dolly-": "dolly",
        "stablelm-3b-": "stablelm-3b",
        "stablelm-": "stablelm",
        "redpajama-": "redpajama_chat",
        "minigpt": "minigpt",
        "moss-moon-003-sft": "moss",
        "moss-moon-003-base": "LM",
        "gpt-j-": "LM",
        "open_llama": "LM",
        "rwkv-": "rwkv",
        "rwkv_world": "rwkv_world",
        "gorilla-": "gorilla",
        "guanaco": "guanaco",
        "wizardlm-7b": "wizardlm_7b",  # first get rid of 7b
        "wizardlm-": "vicuna_v1.1",  # all others use vicuna template
        "wizardmath-": "wizard_coder_or_math",
        "wizardcoder-": "wizard_coder_or_math",
        "starcoder": "gpt_bigcode",
        "gpt_bigcode-santacoder": "gpt_bigcode",
        "stablecode-completion": "stablecode_completion",
        "stablecode-instruct": "stablecode_instruct",
        "chatglm2": "glm",
        "chatglm3": "glm",
        "codegeex2": "glm",
        "tinyllama": "chatml",
        "openhermes-2.5-mistral": "open_hermes_mistral",
        "neuralhermes-2.5-mistral": "neural_hermes_mistral",
    }

    for prefix, conv_template in model_conv_templates.items():
        if model.startswith(prefix):
            args.conv_template = conv_template
            break
    else:
        args.conv_template = f"{args.model_category}_default"

    if args.quantization not in quantization_schemes:
        raise ValueError(f'Quantization "{args.quantization}" is not supported.')

    use_ft_quant = args.quantization in ["q4f16_ft", "q8f16_ft", "q4f16_ft_group", "q8f16_ft_group"]
    args.quantization = quantization_schemes[args.quantization]

    if use_ft_quant and args.num_shards > 1:
        # Preprocess is done after sharding for this case.
        args.quantization.linear_weight.do_preprocess = False
        args.quantization.final_fc_weight.do_preprocess = False


def debug_dump_script(mod, name, args: argparse.Namespace, show_meta=True):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(mod.script(show_meta=show_meta))
    print(f"Dump mod to {dump_path}")


def debug_dump_benchmark_script(
    mod: tvm.ir.IRModule,
    name: str,
    args: argparse.Namespace,
) -> None:
    """Extract model level benchmark workloads from relax model."""
    if not args.debug_dump:
        return

    from tvm.dlight.benchmark import (  # pylint: disable=import-error,import-outside-toplevel
        extract_all_func_info_from_relax,
    )

    dump_path = os.path.join(args.artifact_path, "debug", name + ".py")
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(
            "# Please save this file to dlight_bench/models and add\n"
            + f"# `from .{name} import *` to dlight_bench/models/__init__.py\n"
            + "from dlight_bench import DlightBench\n"
            + "from tvm.script import tir as T\n\n"
        )

        stmt = []
        try:
            relax_funcs, _ = extract_all_func_info_from_relax(mod)
        except NotImplementedError:
            return
        tvm_script_prefix = "# from tvm.script import tir as T"
        for relax_func_gv in relax_funcs:  # pylint: disable=consider-using-dict-items
            for prim_func_gv in relax_funcs[relax_func_gv]:
                # add global_symbol
                func_body = (
                    mod[prim_func_gv]
                    .with_attr("global_symbol", prim_func_gv.name_hint)
                    .script(name=prim_func_gv.name_hint)
                )
                # remove prefix
                if func_body.startswith(tvm_script_prefix + "\n"):
                    func_body = func_body[len(tvm_script_prefix) :]
                # print out
                outfile.write(func_body + "\n")
                # register
                stmt.append(
                    f"DlightBench.register_bench_workload({prim_func_gv.name_hint}, "
                    f"'{name}', '{prim_func_gv.name_hint}')"
                )
        outfile.write("\n" + "\n".join(stmt) + "\n")
    print(f"Dump benchmarking script to {dump_path}.")


def debug_load_script(name: str, args: argparse.Namespace):
    input_path = os.path.join(args.artifact_path, "debug", name)
    lib = {"__file__": input_path}
    with open(input_path, "rb") as i_f:
        exec(compile(i_f.read(), input_path, "exec"), lib, lib)  # pylint: disable=exec-used
    return lib["Module"]


def debug_dump_shader(ex: tvm.relax.Executable, name: str, args: argparse.Namespace):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    target_kind = args.target.kind.default_keys[0]
    suffix_map = {
        "webgpu": ".wgsl",
        "cuda": ".cu",
        "metal": ".mtl",
        "opencl": ".cl",
    }
    suffix = suffix_map.get(target_kind, ".txt")
    dump_path = os.path.join(args.artifact_path, "debug", name + suffix)
    source = ex.mod.imported_modules[0].imported_modules[0].get_source()
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def convert_weights(
    mod_transform: tvm.IRModule,
    param_mgr: param_manager.ParamManager,
    model_params: List[Optional[tvm.nd.NDArray]],
    args: argparse.Namespace,
):
    # Save the number of parameters before we lower mod_transform, so
    # we can use them in the progress bar.
    transform_func = mod_transform["transform_params"]
    num_original_params = len(transform_func.params[0].struct_info.fields)
    num_transformed_params = len(transform_func.struct_info.ret.fields)

    # Remove the dataflow block inside the param transform function,
    # so that the LazyTransformParams pass can be applied.
    mod_transform = relax.transform.ToNonDataflow()(mod_transform)
    mod_transform = relax.transform.LazyTransformParams()(mod_transform)
    mod_transform = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_transform)
    mod_transform = relax.transform.LegalizeOps()(mod_transform)

    debug_dump_script(mod_transform, "mod_convert_weights.py", args)

    target = detect_local_target()
    print(f"Automatically using target for weight quantization: {target}")
    device = tvm.device(target.kind.default_keys[0])
    device_cpu = tvm.cpu()

    loaded_params: List[tvm.nd.NDArray] = []
    loaded_idx_set: Set[int] = set()
    loaded_torch_bins: Set[str] = set()
    cached_relax_params: Dict[int, tvm.nd.NDArray] = {}
    cached_torch_params: Dict[str, Any] = {}

    get_item, set_item = param_mgr.get_param_loading_functions(
        model_params,
        loaded_params,
        loaded_idx_set,
        loaded_torch_bins,
        cached_relax_params,
        cached_torch_params,
        device,
        device_cpu,
    )

    get_item = wrap_tqdm_counter(
        get_item, desc="Get old param", position=0, unit="tensors", total=num_original_params
    )
    set_item = wrap_tqdm_counter(
        set_item, desc="Set new param", position=1, unit="tensors", total=num_transformed_params
    )

    tvm.register_func(func_name="get_item", f=get_item, override=True)
    tvm.register_func(func_name="set_item", f=set_item, override=True)

    if target.kind.name != "llvm":
        with tvm.target.Target(target):
            mod_transform = tvm.tir.transform.DefaultGPUSchedule()(mod_transform)

    ex = relax.build(mod_transform, target=target)
    vm = relax.vm.VirtualMachine(ex, device)
    print("Start computing and quantizing weights... This may take a while.")
    vm["transform_params"]()
    print("Finish computing and quantizing weights.")
    return loaded_params


def save_params(params: List[tvm.nd.NDArray], artifact_path: str, num_presharded: int = 1) -> None:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    assert len(params) % num_presharded == 0
    num_weights = len(params) // num_presharded

    meta_data = {}
    param_dict = {}
    meta_data["ParamSize"] = len(params)
    for i, nd in enumerate(params):
        if num_presharded == 1:
            param_name = f"param_{i}"
        else:
            expected_worker_id = i // num_weights
            orig_param_id = i % num_weights
            param_name = f"param_{orig_param_id}_shard-{expected_worker_id+1}-of-{num_presharded}"

        param_dict[param_name] = nd

    total_size_bytes = sum(
        math.prod(param.shape) * np.dtype(param.dtype).itemsize for param in params
    )
    total_size_gb = total_size_bytes / (1024**3)
    print(f"Total param size: {total_size_gb} GB")
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


def copy_tokenizer(args: argparse.Namespace) -> None:
    for filename in os.listdir(args.model_path):
        if filename in [
            "tokenizer.model",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "tokenizer_config.json",
        ]:
            shutil.copy(
                os.path.join(args.model_path, filename),
                os.path.join(args.artifact_path, "model") if args.enable_batching else os.path.join(args.artifact_path, "params"),
            )

    # If we have `tokenizer.model` but not `tokenizer.json`, try convert it to
    # `tokenizer.json` with `transformers`.
    tokenizer_json_path = os.path.join(args.model_path, "tokenizer.json")
    tokenizer_model_path = os.path.join(args.model_path, "tokenizer.model")
    if os.path.exists(tokenizer_model_path) and (not os.path.exists(tokenizer_json_path)):
        print("Attempting to convert `tokenizer.model` to `tokenizer.json`.")
        try:
            # pylint: disable=import-outside-toplevel
            from transformers import AutoTokenizer

            tokenizer_json_save_dest = os.path.join(args.artifact_path, "params/tokenizer.json")
            fast_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
            fast_tokenizer.backend_tokenizer.save(tokenizer_json_save_dest)
            print(f"Succesfully converted `tokenizer.model` to: {tokenizer_json_save_dest}")
        except ImportError:
            print(
                "WARNING: The model has `tokenizer.model` but not `tokenizer.json`. It is"
                + "recommended to use `tokenizer.json`, so we try convert it with `transformers`.\n"
                + "However, we were unable to import `transformers`, hence skipping this step."
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            print(
                "WARNING: The model has `tokenizer.model` but not `tokenizer.json`. It is"
                + "recommended to use `tokenizer.json`, so we try convert it with `transformers`.\n"
                + "However, we are skipping this due to an error:\n",
                error,
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


def _detect_local_metal_host():
    target_triple = tvm._ffi.get_global_func("tvm.codegen.llvm.GetDefaultTargetTriple")()
    process_triple = tvm._ffi.get_global_func("tvm.codegen.llvm.GetProcessTriple")()
    host_cpu = tvm._ffi.get_global_func("tvm.codegen.llvm.GetHostCPUName")()
    print(
        f"Host CPU dection:\n  Target triple: {target_triple}\n  Process triple: {process_triple}\n  Host CPU: {host_cpu}"
    )
    if target_triple.startswith("x86_64-"):
        return tvm.target.Target(
            {
                "kind": "llvm",
                "mtriple": "x86_64-apple-macos",
                "mcpu": host_cpu,
            }
        )
    # should start with "arm64-"
    return tvm.target.Target(
        {
            "kind": "llvm",
            "mtriple": "arm64-apple-macos",
            "mcpu": host_cpu,
        }
    )


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
        host=_detect_local_metal_host(),
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
            "arch": "sm_" + dev.compute_version.replace(".", ""),
        }
    )


def _detect_local_rocm():
    dev = tvm.rocm()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "rocm",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
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
            "supports_int8": 1,
            "supports_16bit_buffer": 1,
        }
    )


def _detect_local_opencl():
    dev = tvm.opencl()
    if not dev.exist:
        return None
    return tvm.target.Target("opencl")


def detect_local_target():
    for method in [
        _detect_local_metal,
        _detect_local_rocm,
        _detect_local_cuda,
        _detect_local_vulkan,
        _detect_local_opencl,
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
    elif args.target == "cuda" or args.target == "cuda-multiarch":
        target = _detect_local_cuda()
        if target is None:
            raise ValueError("Cannot detect local CUDA GPU target!")
        multiarch = args.target == "cuda-multiarch"
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
        if multiarch:
            args.target_kind += "-multiarch"
    elif args.target.startswith("nvidia/jetson"):
        try:
            args.target = tvm.target.Target(args.target)
        except ValueError:
            raise ValueError("Cannot find configuration of given nvidia/jetson board target!")
        if not hasattr(args, "cc_path") or args.cc_path == "":
            args.cc_path = "/usr/bin/aarch64-linux-gnu-g++"
        from tvm.contrib.cc import (  # pylint: disable=import-outside-toplevel
            cross_compiler,
        )

        args.export_kwargs = {
            "fcompile": cross_compiler(
                args.cc_path,
            ),
        }
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
                host=_detect_local_metal_host(),
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
            args.system_lib_prefix = f"{args.model}_{args.quantization}_".replace("-", "_")

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
        target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "vulkan",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "supports_float16": 1,
                    "supports_int16": 1,
                    "supports_int8": 1,
                    "supports_8bit_buffer": 1,
                    "supports_16bit_buffer": 1,
                    "supports_storage_buffer_storage_class": 1,
                }
            ),
            host="llvm",
        )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "opencl":
        target = tvm.target.Target(
            "opencl",
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
        if os.environ.get("TVM_HOME", "") == "":
            raise RuntimeError(
                "Please set TVM_HOME for webgpu build following scripts/prep_emcc_deps.sh"
            )
    elif args.target in ["android", "android-dylib"]:  # android-opencl
        from tvm.contrib import ndk, tar

        if args.target == "android-dylib":
            args.export_kwargs = {
                "fcompile": ndk.create_shared,
            }
            args.lib_format = "so"
        else:
            args.export_kwargs = {
                "fcompile": tar.tar,
            }
            args.lib_format = "tar"
            args.system_lib = True
            args.system_lib_prefix = f"{args.model}_{args.quantization}_".replace("-", "_")
        args.target = tvm.target.Target(
            "opencl",
            host="llvm -mtriple=aarch64-linux-android",  # TODO: Only support arm64 for now
        )
        args.target_kind = "android"
    elif args.target in ["mali"]:
        if "TVM_NDK_CC" in os.environ:
            from tvm.contrib import ndk

            args.export_kwargs = {
                "fcompile": ndk.create_shared,
            }
        target = tvm.target.Target(
            "opencl -device=mali",
            host="llvm -mtriple=aarch64-linux-gnu",
        )
        args.target = target
        args.target_kind = "mali"
    else:
        args.target = tvm.target.Target(args.target, host="llvm")
        args.target_kind = args.target.kind.default_keys[0]

    if args.target_kind == "cuda-multiarch":
        from tvm.contrib import nvcc

        assert args.target.arch[3:] != ""
        arch_list = os.getenv("CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST")
        if arch_list:
            compute_versions = [int(v) for v in arch_list.replace(" ", ";").split(";")]
        elif int(args.target.arch[3:]) >= 70:
            compute_versions = [70, 72, 75, 80, 86, 87, 89, 90]
        else:
            compute_versions = [60, 61, 62]

        args.target_kind = "cuda"

        @tvm.register_func("tvm_callback_cuda_compile", override=True)
        def tvm_callback_cuda_compile(code, target):  # pylint: disable=unused-argument
            """use nvcc to generate fatbin code for better optimization"""
            arch = []
            for compute_version in compute_versions:
                arch += ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]
            ptx = nvcc.compile_cuda(code, target_format="fatbin", arch=arch)
            return ptx

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
