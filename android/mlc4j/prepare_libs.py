"""The build script for mlc4j (MLC LLM and tvm4j)"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mlc_llm.support import logging

logging.enable_logging()
logger = logging.getLogger(__name__)


def run_cmake(mlc4j_path: Path):
    if "ANDROID_NDK" not in os.environ:
        raise ValueError(
            f'Environment variable "ANDROID_NDK" is required but not found.'
            "Please follow https://llm.mlc.ai/docs/deploy/android.html to properly "
            'specify "ANDROID_NDK".'
        )
    logger.info("Running cmake")
    # use pathlib so it is cross platform
    android_ndk_path = (
        Path(os.environ["ANDROID_NDK"]) / "build" / "cmake" / "android.toolchain.cmake"
    )
    cmd = [
        "cmake",
        str(mlc4j_path),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_TOOLCHAIN_FILE={str(android_ndk_path)}",
        "-DCMAKE_INSTALL_PREFIX=.",
        '-DCMAKE_CXX_FLAGS="-O3"',
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_NATIVE_API_LEVEL=android-24",
        "-DANDROID_PLATFORM=android-24",
        "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ON",
        "-DANDROID_STL=c++_static",
        "-DUSE_HEXAGON_SDK=OFF",
        "-DMLC_LLM_INSTALL_STATIC_LIB=ON",
        "-DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON",
        "-DUSE_OPENCL=ON",
        "-DUSE_OPENCL_ENABLE_HOST_PTR=ON",
        "-DUSE_CUSTOM_LOGGING=ON",
        "-DTVM_FFI_USE_LIBBACKTRACE=OFF",
        "-DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF",
    ]

    if sys.platform == "win32":
        logger.info("Using ninja in windows, make sure you installed ninja in conda")
        cmd += ["-G", "Ninja"]
    subprocess.run(cmd, check=True, env=os.environ)


def run_cmake_build():
    logger.info("Running cmake build")
    cmd = [
        "cmake",
        "--build",
        ".",
        "--target",
        "tvm4j_runtime_packed",
        "--config",
        "release",
        f"-j{os.cpu_count()}",
    ]
    subprocess.run(cmd, check=True, env=os.environ)


def run_cmake_install():
    logger.info("Running cmake install")
    cmd = [
        "cmake",
        "--build",
        ".",
        "--target",
        "install",
        "--config",
        "release",
        f"-j{os.cpu_count()}",
    ]
    subprocess.run(cmd, check=True, env=os.environ)


def main(mlc_llm_source_dir: Path):
    # - Setup rust.
    subprocess.run(["rustup", "target", "add", "aarch64-linux-android"], check=True, env=os.environ)

    # - Build MLC LLM and tvm4j.
    build_path = Path("build")
    os.makedirs(build_path / "lib", exist_ok=True)
    logger.info('Entering "%s" for MLC LLM and tvm4j build.', os.path.abspath(build_path))
    os.chdir(build_path)
    # Generate config.cmake if TVM Home is set.
    if "TVM_SOURCE_DIR" in os.environ:
        logger.info('Set TVM_SOURCE_DIR to "%s"', os.environ["TVM_SOURCE_DIR"])
        with open("config.cmake", "w", encoding="utf-8") as file:
            # We use "json.dumps" to escape backslashes and quotation marks
            tvm_source_dir_str_with_escape = json.dumps(os.environ["TVM_SOURCE_DIR"])
            print("set(TVM_SOURCE_DIR %s)" % tvm_source_dir_str_with_escape, file=file)

    # - Run cmake, build and install
    run_cmake(mlc_llm_source_dir / "android" / "mlc4j")
    run_cmake_build()
    run_cmake_install()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLC LLM Android Lib Preparation")

    parser.add_argument(
        "--mlc-llm-source-dir",
        type=Path,
        default=os.environ.get("MLC_LLM_SOURCE_DIR", None),
        help="The path to MLC LLM source",
    )
    parsed = parser.parse_args()
    if parsed.mlc_llm_source_dir is None:
        parsed.mlc_llm_source_dir = Path(os.path.abspath(os.path.curdir)).parent.parent
    os.environ["MLC_LLM_SOURCE_DIR"] = str(parsed.mlc_llm_source_dir)
    main(parsed.mlc_llm_source_dir)
