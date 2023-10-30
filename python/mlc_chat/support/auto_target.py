"""Helper functioms for target auto-detection."""
import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from tvm import IRModule, relax
from tvm._ffi import register_func
from tvm.contrib import tar, xcode
from tvm.target import Target

from .style import green, red

if TYPE_CHECKING:
    from mlc_chat.compiler.compile import CompileArgs


logger = logging.getLogger(__name__)

# TODO: add help message on how to specify the target manually # pylint: disable=fixme
# TODO: include host detection logic below after the new TVM build is done. # pylint: disable=fixme
HELP_MSG = """TBD"""
FOUND = green("Found")
NOT_FOUND = red("Not found")
BuildFunc = Callable[[IRModule, "CompileArgs"], None]


def detect_target_and_host(target_hint: str, host_hint: str) -> Tuple[Target, BuildFunc]:
    """Detect the configuration for the target device and its host, for example, target GPU and
    the host CPU.

    Parameters
    ----------
    target_hint : str
        The hint for the target device.

    host_hint : str
        The hint for the host CPU.
    """
    target, build_func = _detect_target_gpu(target_hint)
    if target.host is None:
        target = Target(target, host=_detect_target_host(host_hint))
    return target, build_func


def _detect_target_gpu(hint: str) -> Tuple[Target, BuildFunc]:
    if hint in ["iphone", "android", "webgpu", "mali", "opencl"]:
        hint += ":generic"
    if hint == "auto":
        logger.info("Detecting potential target devices: %s", ", ".join(AUTO_DETECT_DEVICES))
        target: Optional[Target] = None
        for device in AUTO_DETECT_DEVICES:
            device_target = _detect_target_from_device(device + ":0")
            if device_target is not None and target is None:
                target = device_target
        if target is None:
            raise ValueError("No GPU target detected. Please specify explicitly")
        return target, _build_default()
    if hint in AUTO_DETECT_DEVICES:
        target = _detect_target_from_device(hint + ":0")
        if target is None:
            raise ValueError(f"No GPU target detected from device: {hint}")
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
    # cpu = codegen.llvm_get_system_cpu()
    # triple = codegen.llvm_get_system_triple()
    # vendor = codegen.llvm_get_system_x86_vendor()
    if hint == "auto":
        hint = "x86-64"
    if hint == "x86-64":
        hint = "x86_64"
    return Target({"kind": "llvm", "mtriple": f"{hint}-unknown-unknown"})


def _is_device(device: str):
    if " " in device:
        return False
    if device.count(":") != 1:
        return False
    return True


def _add_prefix_symbol(mod: IRModule, prefix: str, is_system_lib: bool) -> IRModule:
    if is_system_lib and prefix:
        mod = mod.with_attr("system_lib_prefix", prefix)
    elif is_system_lib:
        logger.warning("--prefix-symbols is not specified when building a static library")
    elif prefix:
        logger.warning(
            "--prefix-symbols is specified, but it will not take any effect "
            "when building the shared library"
        )
    return mod


def _detect_target_from_device(device: str) -> Optional[Target]:
    try:
        target = Target.from_device(device)
    except ValueError:
        logger.info("%s: target device: %s", NOT_FOUND, device)
        return None
    logger.info(
        '%s configuration of target device "%s": %s',
        FOUND,
        device,
        target.export(),
    )
    return target


def _build_metal_x86_64():
    def build(mod: IRModule, args: "CompileArgs"):
        output = args.output
        mod = _add_prefix_symbol(mod, args.prefix_symbols, is_system_lib=False)
        assert output.suffix == ".dylib"
        relax.build(
            mod,
            target=args.target,
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

    def build(mod: IRModule, args: "CompileArgs"):
        output = args.output
        mod = _add_prefix_symbol(mod, args.prefix_symbols, is_system_lib=True)
        assert output.suffix == ".tar"
        relax.build(
            mod,
            target=args.target,
            system_lib=True,
        ).export_library(
            str(output),
            fcompile=tar.tar,
        )

    return build


def _build_android():
    def build(mod: IRModule, args: "CompileArgs"):
        output = args.output
        mod = _add_prefix_symbol(mod, args.prefix_symbols, is_system_lib=True)
        assert output.suffix == ".tar"
        relax.build(
            mod,
            target=args.target,
            system_lib=True,
        ).export_library(
            str(output),
            fcompile=tar.tar,
        )

    return build


def _build_webgpu():
    def build(mod: IRModule, args: "CompileArgs"):
        output = args.output
        mod = _add_prefix_symbol(mod, args.prefix_symbols, is_system_lib=True)
        assert output.suffix == ".wasm"
        relax.build(
            mod,
            target=args.target,
            system_lib=True,
        ).export_library(
            str(output),
        )

    return build


def _build_default():
    def build(mod: IRModule, args: "CompileArgs"):
        output = args.output
        if output.suffix in [".tar", ".lib"]:
            system_lib = True
        elif output.suffix in [".so", ".dylib", ".dll"]:
            system_lib = False
        else:
            logger.warning("Unknown output suffix: %s. Assuming shared library.", output.suffix)
            system_lib = False
        mod = _add_prefix_symbol(mod, args.prefix_symbols, is_system_lib=system_lib)
        relax.build(
            mod,
            target=args.target,
            system_lib=system_lib,
        ).export_library(
            str(output),
        )

    return build


AUTO_DETECT_DEVICES = ["cuda", "rocm", "metal", "vulkan"]

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
