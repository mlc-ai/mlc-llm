"""Helper functioms for target auto-detection."""
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from tvm import IRModule, relax
from tvm._ffi import get_global_func, register_func
from tvm.contrib import tar, xcode
from tvm.ir.transform import Pass
from tvm.target import Target

from . import logging
from .auto_device import AUTO_DETECT_DEVICES, detect_device, device2str
from .constants import MLC_MULTI_ARCH
from .style import bold, green, red

if TYPE_CHECKING:
    from mlc_chat.compiler.compile import CompileArgs


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
        _register_cuda_hook(target)
    return target, build_func


def _detect_target_gpu(hint: str) -> Tuple[Target, BuildFunc]:
    if hint in ["iphone", "android", "webgpu", "mali", "opencl"]:
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
            pipeline=pipeline,
        ).export_library(
            str(output),
            fcompile=xcode.create_dylib,
            sdk="macosx",
            arch="x86_64",
        )

    return build


def _build_iphone():
    @register_func("tvm_callback_metal_compile", override=True)
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
            pipeline=pipeline,
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
        relax.build(
            mod,
            target=args.target,
            pipeline=pipeline,
            system_lib=True,
        ).export_library(
            str(output),
            fcompile=tar.tar,
        )

    return build


def _build_webgpu():
    def build(mod: IRModule, args: "CompileArgs", pipeline=None):
        output = args.output
        mod = _add_system_lib_prefix(mod, args.system_lib_prefix, is_system_lib=True)
        assert output.suffix == ".wasm"
        relax.build(
            mod,
            target=args.target,
            pipeline=pipeline,
            system_lib=True,
        ).export_library(
            str(output),
        )

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
            pipeline=pipeline,
            system_lib=system_lib,
        ).export_library(
            str(output),
        )

    return build


def detect_cuda_arch_list(target: Target) -> List[int]:
    """Detect the CUDA architecture list from the target."""
    assert target.kind.name == "cuda", f"Expect target to be CUDA, but got {target}"
    if MLC_MULTI_ARCH is not None:
        multi_arch = [int(x.strip()) for x in MLC_MULTI_ARCH.split(",")]
    else:
        assert target.arch.startswith("sm_")
        multi_arch = [int(target.arch[3:])]
    multi_arch = list(set(multi_arch))
    return multi_arch


def _register_cuda_hook(target: Target):
    if MLC_MULTI_ARCH is None:
        default_arch = target.attrs.get("arch", None)
        logger.info("Generating code for CUDA architecture: %s", bold(default_arch))
        logger.info(
            "To produce multi-arch fatbin, set environment variable %s. "
            "Example: MLC_MULTI_ARCH=70,72,75,80,86,87,89,90",
            bold("MLC_MULTI_ARCH"),
        )
        multi_arch = None
    else:
        logger.info("%s %s: %s", FOUND, bold("MLC_MULTI_ARCH"), MLC_MULTI_ARCH)
        multi_arch = [int(x.strip()) for x in MLC_MULTI_ARCH.split(",")]
        logger.info("Generating code for CUDA architecture: %s", multi_arch)

    @register_func("tvm_callback_cuda_compile", override=True)
    def tvm_callback_cuda_compile(code, target):  # pylint: disable=unused-argument
        """use nvcc to generate fatbin code for better optimization"""
        from tvm.contrib import nvcc  # pylint: disable=import-outside-toplevel

        if multi_arch is None:
            ptx = nvcc.compile_cuda(code, target_format="fatbin")
        else:
            arch = []
            for compute_version in multi_arch:
                arch += ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]
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
    if prefix_hint == "auto" and target_hint in ["iphone", "android"]:
        prefix = f"{model_name}_{quantization}_".replace("-", "_")
        logger.warning(
            "%s is automatically picked from the filename, %s, this allows us to use the filename "
            "as the model_lib in android/iOS builds. Please avoid renaming the .tar file when "
            "uploading the prebuilt.",
            bold("--system-lib-prefix"),
            bold(prefix),
        )
        return prefix
    if not target_hint in ["iphone", "android"]:
        return ""
    return prefix_hint


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
            "supports_int16": 1,
            "supports_int8": 1,
            "supports_8bit_buffer": 1,
            "supports_16bit_buffer": 1,
            "supports_storage_buffer_storage_class": 1,
        },
    },
}
