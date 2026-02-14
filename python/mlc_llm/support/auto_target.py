"""Helper functions for target auto-detection."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from tvm import IRModule, relax
from tvm.contrib import ndk, tar, xcode
from tvm.ir.transform import Pass
from tvm.target import Target
from tvm_ffi import get_global_func, register_global_func

from . import logging
from .auto_device import AUTO_DETECT_DEVICES, detect_device, device2str
from .constants import MLC_MULTI_ARCH
from .style import bold, green, red

if TYPE_CHECKING:
    from mlc_llm.compiler.compile import CompileArgs


logger = logging.getLogger(__name__)

# TODO: add help message on how to specify the target manually # pylint: disable=fixme
HELP_MSG = """TBD"""
FOUND = green("Found")
NOT_FOUND = red("Not found")
BuildFunc = Callable[[IRModule, "CompileArgs", Pass], None]


def detect_target_and_host(target_hint: str, host_hint: str = "auto") -> Tuple[Target, BuildFunc]:
    """Detect the configuration for the target device and its host, for example, target GPU and
    the host CPU.

    Parameters
    ----------
    target_hint : str
        The hint for the target device.

    host_hint : str
        The hint for the host CPU, default is "auto".
    """
    target, build_func = _detect_target_gpu(target_hint)
    if target.host is None:
        target = Target(target, host=_detect_target_host(host_hint))
    if target.kind.name == "cuda":
        # Enable thrust for CUDA
        target_dict = dict(target.export())
        target_dict["libs"] = (
            (target_dict["libs"] + ["thrust"]) if "libs" in target_dict else ["thrust"]
        )
        target = Target(target_dict)
        _register_cuda_hook(target)
    elif target.kind.name == "rocm":
        target_dict = dict(target.export())
        extra_libs = ["thrust", "rocblas", "miopen", "hipblas"]
        target_dict["libs"] = (
            (target_dict["libs"] + extra_libs) if "libs" in target_dict else extra_libs
        )
        target = Target(target_dict)
    return target, build_func


def _detect_target_gpu(hint: str) -> Tuple[Target, BuildFunc]:
    if hint in ["iphone", "macabi", "android", "webgpu", "mali", "opencl"]:
        hint += ":generic"
    if hint == "auto" or hint in AUTO_DETECT_DEVICES:
        target: Optional[Target] = None
        device = detect_device(hint)
        if device is not None:
            device_str = device2str(device)
            try:
                target = Target.from_device(device)
            except ValueError:
                logger.info("%s: Cannot detect target from device: %s", NOT_FOUND, device_str)
        if target is None:
            raise ValueError(f"No target detected from device: {hint}. Please specify explicitly")
        logger.info(
            '%s configuration of target device "%s": %s',
            FOUND,
            bold(device_str),
            target.export(),
        )
        return target, _build_default()
    if hint in PRESET:
        preset = PRESET[hint]
        target = Target(preset["target"])  # type: ignore[index]
        build = preset.get("build", _build_default)  # type: ignore[attr-defined]
        return target, build()
    if _is_device(hint):
        logger.info("Detecting target device: %s", hint)
        target = Target.from_device(hint)
        logger.info("%s target: %s", FOUND, target.export())
        return target, _build_default()
    try:
        logger.info("Try creating device target from string: %s", hint)
        target = Target(hint)
        logger.info("%s target: %s", FOUND, target.export())
        return target, _build_default()
    except Exception as err:
        logger.info("%s: Failed to create target", NOT_FOUND)
        raise ValueError(f"Invalid target: {hint}") from err


def _detect_target_host(hint: str) -> Target:
    """Detect the host CPU architecture."""
    if hint == "auto":
        target_triple = get_global_func("tvm.codegen.llvm.GetDefaultTargetTriple")()
        target = Target.from_device("cpu")
        logger.info("%s host LLVM triple: %s", FOUND, bold(target.attrs["mtriple"]))
        logger.info("%s host LLVM CPU: %s", FOUND, bold(target.attrs["mcpu"]))
        return target
    target_triple = hint
    logger.info("Using LLVM triple specified by --host: %s", bold(target_triple))
    return Target({"kind": "llvm", "mtriple": target_triple})


def _is_device(device: str):
    if " " in device:
        return False
    if device.count(":") != 1:
        return False
    return True


def _add_system_lib_prefix(mod: IRModule, prefix: str, is_system_lib: bool) -> IRModule:
    if is_system_lib and prefix:
        mod = mod.with_attrs({"system_lib_prefix": prefix})  # type: ignore[dict-item]
    elif is_system_lib:
        logger.warning(
            "%s is not specified when building a static library",
            bold("--system-lib-prefix"),
        )
    elif prefix:
        logger.warning(
            "--system-lib-prefix is specified, but it will not take any effect "
            "when building the shared library"
        )
    return mod


def _build_metal_x86_64():
    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=False)
        assert output.suffix == ".dylib"
        relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
        ).export_library(
            str(output),
            fcompile=xcode.create_dylib,
            sdk="macosx",
            arch="x86_64",
        )

    return build


def _build_iphone():
    @register_global_func("tvm_callback_metal_compile", override=True)
    def compile_metal(src, target):
        if target.libs:
            return xcode.compile_metal(src, sdk=target.libs[0])
        return xcode.compile_metal(src)

    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=True)
        assert output.suffix == ".tar"
        relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
            system_lib=True,
        ).export_library(
            str(output),
            fcompile=tar.tar,
        )

    return build


def _build_android():
    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=True)
        assert output.suffix == ".tar"
        ex = relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
            system_lib=True,
        )
        ex.export_library(
            str(output),
            fcompile=tar.tar,
        )
        if args.debug_dump is not None:
            source = ex.mod.imports[0].imports[0].inspect_source()
            with open(args.debug_dump / "kernel.cl", "w", encoding="utf-8") as f:
                f.write(source)

    return build


def _build_android_so():
    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=False)
        assert output.suffix == ".so"
        ex = relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
            system_lib=False,
        )
        ex.export_library(
            str(output),
            fcompile=ndk.create_shared,
        )
        if args.debug_dump is not None:
            source = ex.mod.imports[0].imports[0].inspect_source()
            with open(args.debug_dump / "kernel.cl", "w", encoding="utf-8") as f:
                f.write(source)

    return build


def _build_webgpu():
    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=True)
        assert output.suffix == ".wasm"

        # Try to locate `mlc_wasm_runtime.bc`
        bc_path = None
        bc_candidates = ["web/dist/wasm/mlc_wasm_runtime.bc"]
        if os.environ.get("MLC_LLM_SOURCE_DIR", None):
            mlc_source_home_dir = os.environ["MLC_LLM_SOURCE_DIR"]
            bc_candidates.append(
                os.path.join(mlc_source_home_dir, "web", "dist", "wasm", "mlc_wasm_runtime.bc")
            )
        error_info = (
            "Cannot find library: mlc_wasm_runtime.bc\n"
            + "Make sure you have run `./web/prep_emcc_deps.sh` and "
            + "`export MLC_LLM_SOURCE_DIR=/path/to/mlc-llm` so that we can locate the file. "
            + "We tried to look at candidate paths:\n"
        )
        for candidate in bc_candidates:
            error_info += candidate + "\n"
            if Path(candidate).exists():
                bc_path = candidate
        if not bc_path:
            raise RuntimeError(error_info)

        relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
            system_lib=True,
        ).export_library(
            str(output),
            libs=[bc_path],
        )

    return build


def _build_mali():
    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=True)
        assert output.suffix == ".so"
        mod = relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
            system_lib=True,
        )
        if "TVM_NDK_CC" in os.environ:
            mod.export_library(str(output), fcompile=ndk.create_shared)
        else:
            mod.export_library(str(output))

    return build


def _build_default():
    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        if output.suffix in [".tar", ".lib"]:
            system_lib = True
        elif output.suffix in [".so", ".dylib", ".dll"]:
            system_lib = False
        else:
            logger.warning("Unknown output suffix: %s. Assuming shared library.", output.suffix)
            system_lib = False
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=system_lib)
        relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
            system_lib=system_lib,
        ).export_library(
            str(output),
        )

    return build


def detect_cuda_arch_list(target: Target) -> List[int]:
    """Detect the CUDA architecture list from the target."""

    def convert_to_num(arch_str):
        arch_num_str = "".join(filter(str.isdigit, arch_str))
        assert arch_num_str, f"'{arch_str}' does not contain any digits"
        return int(arch_num_str)

    assert target.kind.name == "cuda", f"Expect target to be CUDA, but got {target}"
    if MLC_MULTI_ARCH is not None:
        multi_arch = [convert_to_num(x) for x in MLC_MULTI_ARCH.split(",")]
    else:
        assert target.arch.startswith("sm_")
        multi_arch = [convert_to_num(target.arch[3:])]
    multi_arch = list(set(multi_arch))
    return multi_arch


def _register_cuda_hook(target: Target):
    if MLC_MULTI_ARCH is None:
        default_arch = target.attrs.get("arch", None)
        logger.info("Generating code for CUDA architecture: %s", bold(default_arch))
        logger.info(
            "To produce multi-arch fatbin, set environment variable %s. "
            "Example: MLC_MULTI_ARCH=70,72,75,80,86,87,89,90a",
            bold("MLC_MULTI_ARCH"),
        )
        multi_arch = None
    else:
        logger.info("%s %s: %s", FOUND, bold("MLC_MULTI_ARCH"), MLC_MULTI_ARCH)
        multi_arch = [x.strip() for x in MLC_MULTI_ARCH.split(",")]
        logger.info("Generating code for CUDA architecture: %s", multi_arch)

    @register_global_func("tvm_callback_cuda_compile", override=True)
    def tvm_callback_cuda_compile(code, target):  # pylint: disable=unused-argument
        """use nvcc to generate fatbin code for better optimization"""
        from tvm.contrib import nvcc  # pylint: disable=import-outside-toplevel

        if multi_arch is None:
            ptx = nvcc.compile_cuda(code, target_format="fatbin")
        else:
            arch = []
            for compute_version in multi_arch:
                arch += [
                    "-gencode",
                    f"arch=compute_{compute_version},code=sm_{compute_version}",
                ]
            ptx = nvcc.compile_cuda(code, target_format="fatbin", arch=arch)
        return ptx


def detect_system_lib_prefix(
    target_hint: str, prefix_hint: str, model_name: str, quantization: str
) -> str:
    """Detect the iOS / Android system lib prefix to identify the library needed to load the app.

    Parameters
    ----------
    target_hint : str
        The hint for the target device.

    prefix_hint : str
        The hint for the system lib prefix.
    """
    if prefix_hint == "auto" and (
        target_hint.startswith("iphone")
        or target_hint.startswith("macabi")
        or target_hint.startswith("android")
    ):
        prefix = f"{model_name}_{quantization}_".replace("-", "_")
        logger.warning(
            "%s is automatically picked from the filename, %s, this allows us to use the filename "
            "as the model_lib in android/iOS builds. Please avoid renaming the .tar file when "
            "uploading the prebuilt.",
            bold("--system-lib-prefix"),
            bold(prefix),
        )
        return prefix
    if target_hint not in ["iphone", "macabi", "android"]:
        return ""
    return prefix_hint


_MACABI_ARCH = os.environ.get("MLC_MACABI_ARCH", "").strip() or "arm64"
if _MACABI_ARCH not in ["arm64", "x86_64"]:
    _MACABI_ARCH = "arm64"
_MACABI_MTRIPLE = (
    "x86_64-apple-ios18.0-macabi" if _MACABI_ARCH == "x86_64" else "arm64-apple-ios18.0-macabi"
)

PRESET = {
    "iphone:generic": {
        "target": {
            "kind": "metal",
            "max_threads_per_block": 256,
            "max_shared_memory_per_block": 32768,
            "thread_warp_size": 1,
            "libs": ["iphoneos"],
            "host": {
                "kind": "llvm",
                "mtriple": "arm64-apple-darwin",
            },
        },
        "build": _build_iphone,
    },
    "macabi:generic": {
        "target": {
            "kind": "metal",
            "max_threads_per_block": 256,
            "max_shared_memory_per_block": 32768,
            "thread_warp_size": 1,
            "libs": ["macosx"],
            "host": {
                "kind": "llvm",
                "mtriple": _MACABI_MTRIPLE,
            },
        },
        "build": _build_iphone,
    },
    "android:generic": {
        "target": {
            "kind": "opencl",
            "host": {
                "kind": "llvm",
                "mtriple": "aarch64-linux-android",
            },
        },
        "build": _build_android,
    },
    "android:adreno": {
        "target": {
            "kind": "opencl",
            "device": "adreno",
            "max_threads_per_block": 512,
            "host": {
                "kind": "llvm",
                "mtriple": "aarch64-linux-android",
            },
        },
        "build": _build_android,
    },
    "android:adreno-so": {
        "target": {
            "kind": "opencl",
            "device": "adreno",
            "max_threads_per_block": 512,
            "host": {
                "kind": "llvm",
                "mtriple": "aarch64-linux-android",
            },
        },
        "build": _build_android_so,
    },
    "windows:adreno_x86": {
        "target": {
            "kind": "opencl",
            "device": "adreno",
            "max_threads_per_block": 512,
            "host": {
                "kind": "llvm",
                "mtriple": "x86_64-pc-windows-msvc",
            },
        },
    },
    "metal:x86-64": {
        "target": {
            "kind": "metal",
            "max_threads_per_block": 256,
            "max_shared_memory_per_block": 32768,
            "thread_warp_size": 1,
        },
        "build": _build_metal_x86_64,
    },
    "webgpu:generic": {
        "target": {
            "kind": "webgpu",
            "host": {
                "kind": "llvm",
                "mtriple": "wasm32-unknown-unknown-wasm",
            },
        },
        "build": _build_webgpu,
    },
    "opencl:generic": {
        "target": {
            "kind": "opencl",
        },
    },
    "mali:generic": {
        "target": {
            "kind": "opencl",
            "host": {
                "kind": "llvm",
                "mtriple": "aarch64-linux-gnu",
            },
        },
        "build": _build_mali,
    },
    "metal:generic": {
        "target": {
            "kind": "metal",
            "max_threads_per_block": 256,
            "max_shared_memory_per_block": 32768,
            "thread_warp_size": 1,
        },
    },
    "vulkan:generic": {
        "target": {
            "kind": "vulkan",
            "max_threads_per_block": 256,
            "max_shared_memory_per_block": 32768,
            "thread_warp_size": 1,
            "supports_float16": 1,
            "supports_int64": 1,
            "supports_int16": 1,
            "supports_int8": 1,
            "supports_8bit_buffer": 1,
            "supports_16bit_buffer": 1,
            "supports_storage_buffer_storage_class": 1,
        },
    },
}
