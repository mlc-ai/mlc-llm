"""Python entrypoint of package."""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal

from mlc_llm.interface import jit
from mlc_llm.support import download_cache, logging, style

logging.enable_logging()
logger = logging.getLogger(__name__)

SUPPORTED_DEVICES = ["iphone", "macabi", "android"]


def build_model_library(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    package_config: Dict[str, Any], device: str, bundle_dir: Path, app_config_path: Path
) -> Dict[str, str]:
    """Build model libraries. Return the dictionary of "library prefix to lib path"."""
    # - Create the bundle directory.
    os.makedirs(bundle_dir, exist_ok=True)
    # Clean up all the directories in `output/bundle`.
    logger.info('Clean up all directories under "%s"', str(bundle_dir))
    for content_path in bundle_dir.iterdir():
        if content_path.is_dir():
            shutil.rmtree(content_path)

    # - Process each model, and prepare the app config.
    app_config_model_list = []

    model_entries = package_config.get("model_list", [])
    if not isinstance(model_entries, list):
        raise ValueError('The "model_list" in "mlc-package-config.json" is expected to be a list.')
    model_lib_path_for_prepare_libs = package_config.get("model_lib_path_for_prepare_libs", {})
    if not isinstance(model_lib_path_for_prepare_libs, dict):
        raise ValueError(
            'The "model_lib_path_for_prepare_libs" in "mlc-package-config.json" is expected to be '
            "a dict."
        )

    jit.log_jit_policy()

    for model_entry in package_config.get("model_list", []):
        # - Parse model entry.
        if not isinstance(model_entry, dict):
            raise ValueError('The element of "model_list" is expected to be a dict.')
        model = model_entry["model"]
        model_id = model_entry["model_id"]
        bundle_weight = model_entry.get("bundle_weight", False)
        overrides = model_entry.get("overrides", {})
        model_lib = model_entry.get("model_lib", None)

        estimated_vram_bytes = model_entry["estimated_vram_bytes"]
        if not isinstance(model, str):
            raise ValueError('The value of "model" in "model_list" is expected to be a string.')
        if not isinstance(model_id, str):
            raise ValueError('The value of "model_id" in "model_list" is expected to be a string.')
        if not isinstance(bundle_weight, bool):
            raise ValueError(
                'The value of "bundle_weight" in "model_list" is expected to be a boolean.'
            )
        if not isinstance(overrides, dict):
            raise ValueError('The value of "overrides" in "model_list" is expected to be a dict.')
        if model_lib is not None and not isinstance(model_lib, str):
            raise ValueError('The value of "model_lib" in "model_list" is expected to be string.')

        # - Load model config. Download happens when needed.
        model_path = download_cache.get_or_download_model(model)

        # - Jit compile if the model lib path is not specified.
        model_lib_path = (
            model_lib_path_for_prepare_libs.get(model_lib, None) if model_lib is not None else None
        )
        if model_lib_path is None:
            if model_lib is None:
                logger.info(
                    'Model lib is not specified for model "%s". Now jit compile the model library.',
                    model_id,
                )
            else:
                logger.info(
                    'Model lib path for "%s" is not specified in "model_lib_path_for_prepare_libs".'
                    "Now jit compile the model library.",
                    model_lib,
                )
            model_lib_path, model_lib = dataclasses.astuple(
                jit.jit(
                    model_path=model_path,
                    overrides=overrides,
                    device=device,
                    system_lib_prefix=model_lib,
                    skip_log_jit_policy=True,
                )
            )
            assert model_lib is not None
            model_lib_path_for_prepare_libs[model_lib] = model_lib_path

        # - Set "model_url"/"model_path" and "model_id"
        app_config_model_entry = {}
        is_local_model = not model.startswith("HF://") and not model.startswith("https://")
        app_config_model_entry["model_id"] = model_id
        app_config_model_entry["model_lib"] = model_lib

        # - Bundle weight
        if is_local_model and not bundle_weight:
            raise ValueError(
                f'Model "{model}" in "model_list" is a local path.'
                f'Please set \'"bundle_weight": true\' in the entry of model "{model}".'
            )
        if bundle_weight:
            if not os.path.isfile(model_path / "tensor-cache.json"):
                raise ValueError(
                    f'Bundle weight is set for model "{model}". However, model weights are not'
                    f'found under the directory "{model}". '
                    + (
                        "Please follow https://llm.mlc.ai/docs/compilation/convert_weights.html to "
                        "convert model weights."
                        if is_local_model
                        else "Please report this issue to https://github.com/mlc-ai/mlc-llm/issues."
                    )
                )
            # Overwrite the model weight directory in bundle.
            bundle_model_weight_path = bundle_dir / model_id
            logger.info(
                "Bundle weight for %s, copy into %s",
                style.bold(model_id),
                style.bold(str(bundle_model_weight_path)),
            )
            if bundle_model_weight_path.exists():
                shutil.rmtree(bundle_model_weight_path)
            shutil.copytree(model_path, bundle_model_weight_path)
        if bundle_weight and device in ["iphone", "macabi"]:
            app_config_model_entry["model_path"] = model_id
        else:
            app_config_model_entry["model_url"] = model.replace("HF://", "https://huggingface.co/")

        # - estimated_vram_bytes
        app_config_model_entry["estimated_vram_bytes"] = estimated_vram_bytes

        app_config_model_list.append(app_config_model_entry)

    # - Dump "mlc-app-config.json".
    app_config_json_str = json.dumps(
        {"model_list": app_config_model_list},
        indent=2,
    )
    with open(app_config_path, "w", encoding="utf-8") as file:
        print(app_config_json_str, file=file)
        logger.info(
            'Dump the app config below to "%s":\n%s',
            str(app_config_path),
            style.green(app_config_json_str),
        )
    return model_lib_path_for_prepare_libs


def validate_model_lib(  # pylint: disable=too-many-locals
    app_config_path: Path,
    package_config_path: Path,
    model_lib_path_for_prepare_libs: dict,
    device: Literal["iphone", "macabi", "android"],
    output: Path,
) -> None:
    """Validate the model lib prefixes of model libraries."""
    # pylint: disable=import-outside-toplevel,redefined-outer-name,shadowed-import,reimported
    if device == "android":
        from tvm.contrib import ndk as cc
    else:
        from tvm.contrib import cc
    # pylint: enable=import-outside-toplevel,redefined-outer-name,shadowed-import,reimported

    with open(app_config_path, "r", encoding="utf-8") as file:
        app_config = json.load(file)

    tar_list = []
    model_set = set()

    for model, model_lib_path in model_lib_path_for_prepare_libs.items():
        model_lib_path = os.path.join(model_lib_path)
        lib_path_valid = os.path.isfile(model_lib_path)
        if not lib_path_valid:
            raise RuntimeError(f"Cannot find file {model_lib_path} as an {device} model library")
        tar_list.append(model_lib_path)
        model_set.add(model)

    os.makedirs(output / "lib", exist_ok=True)
    if device in ["iphone", "macabi"]:
        lib_name = "libmodel_iphone.a"
    else:
        lib_name = "libmodel_android.a"
    lib_path = output / "lib" / lib_name

    def _get_model_libs(lib_path: Path) -> List[str]:
        """Get the model lib prefixes in the given static lib path."""
        global_symbol_map = cc.get_global_symbol_section_map(lib_path)
        libs = []
        suffix = "___tvm_ffi__library_bin"
        for name, _ in global_symbol_map.items():
            if name.endswith(suffix):
                model_lib = name[: -len(suffix)]
                if model_lib.startswith("_"):
                    model_lib = model_lib[1:]
                libs.append(model_lib)
        return libs

    cc.create_staticlib(lib_path, tar_list)
    available_model_libs = _get_model_libs(lib_path)
    logger.info("Creating lib from %s", str(tar_list))
    logger.info("Validating the library %s", str(lib_path))
    logger.info(
        "List of available model libs packaged: %s,"
        " if we have '-' in the model_lib string, it will be turned into '_'",
        str(available_model_libs),
    )
    global_symbol_map = cc.get_global_symbol_section_map(lib_path)
    error_happened = False

    for item in app_config["model_list"]:
        model_lib = item["model_lib"]
        model_id = item["model_id"]
        if model_lib not in model_set:
            # NOTE: this cannot happen under new setting
            # since if model_lib is not included, it will be jitted
            raise RuntimeError(
                f"ValidationError: model_lib={model_lib} specified for model_id={model_id} "
                "is not included in model_lib_path_for_prepare_libs argument, "
                "This will cause the specific model not being able to load, "
                f"model_lib_path_for_prepare_libs={model_lib_path_for_prepare_libs}"
            )

        model_prefix_pattern = model_lib.replace("-", "_") + "___tvm_ffi__library_bin"
        if (
            model_prefix_pattern not in global_symbol_map
            and "_" + model_prefix_pattern not in global_symbol_map
        ):
            # NOTE: no lazy format is ok since this is a slow pass
            model_lib_path = model_lib_path_for_prepare_libs[model_lib]
            log_msg = (
                "ValidationError:\n"
                f"\tmodel_lib {model_lib} requested in {str(app_config_path)}"
                f" is not found in {str(lib_path)}\n"
                f"\tspecifically the model_lib for {model_lib_path}.\n"
                f"\tcurrent available model_libs in {str(lib_path)}: {available_model_libs}\n"
                f"\tThis can happen when we manually specified model_lib_path_for_prepare_libs"
                f" in {str(package_config_path)}\n"
                f"\tConsider remove model_lib_path_for_prepare_libs (so library can be jitted)"
                "or check the compile command"
            )
            logger.info(log_msg)
            error_happened = True

    if not error_happened:
        logger.info(style.green("Validation pass"))
    else:
        logger.info(style.red("Validation failed"))
        sys.exit(255)


def build_android_binding(mlc_llm_source_dir: Path, output: Path) -> None:
    """Build android binding in MLC LLM"""
    mlc4j_path = mlc_llm_source_dir / "android" / "mlc4j"

    # Move the model libraries to "build/lib/" for linking
    os.makedirs(Path("build") / "lib", exist_ok=True)
    src_path = str(output / "lib" / "libmodel_android.a")
    dst_path = str(Path("build") / "lib" / "libmodel_android.a")
    logger.info('Moving "%s" to "%s"', src_path, dst_path)
    shutil.move(src_path, dst_path)

    # Build mlc4j
    logger.info("Building mlc4j")
    subprocess.run([sys.executable, mlc4j_path / "prepare_libs.py"], check=True, env=os.environ)
    # Copy built files back to output directory.
    lib_path = output / "lib" / "mlc4j"
    os.makedirs(lib_path, exist_ok=True)
    logger.info('Clean up all directories under "%s"', str(lib_path))
    for content_path in lib_path.iterdir():
        if content_path.is_dir():
            shutil.rmtree(content_path)

    src_path = str(mlc4j_path / "src")
    dst_path = str(lib_path / "src")
    logger.info('Copying "%s" to "%s"', src_path, dst_path)
    shutil.copytree(src_path, dst_path)

    src_path = str(mlc4j_path / "build.gradle")
    dst_path = str(lib_path / "build.gradle")
    logger.info('Copying "%s" to "%s"', src_path, dst_path)
    shutil.copy(src_path, dst_path)

    src_path = str(Path("build") / "output")
    dst_path = str(lib_path / "output")
    logger.info('Copying "%s" to "%s"', src_path, dst_path)
    shutil.copytree(src_path, dst_path)

    os.makedirs(lib_path / "src" / "main" / "assets")
    src_path = str(output / "bundle" / "mlc-app-config.json")
    dst_path = str(lib_path / "src" / "main" / "assets" / "mlc-app-config.json")
    logger.info('Moving "%s" to "%s"', src_path, dst_path)
    shutil.move(src_path, dst_path)


def build_iphone_binding(mlc_llm_source_dir: Path, output: Path) -> None:
    """Build iOS binding in MLC LLM"""
    # Build iphone binding
    logger.info("Build iphone binding")
    subprocess.run(
        ["bash", mlc_llm_source_dir / "ios" / "prepare_libs.sh"],
        check=True,
        env=os.environ,
    )

    # Copy built libraries back to output directory.
    for static_library in (Path("build") / "lib").iterdir():
        dst_path = str(output / "lib" / static_library.name)
        logger.info('Copying "%s" to "%s"', static_library, dst_path)
        shutil.copy(static_library, dst_path)


def build_macabi_binding(mlc_llm_source_dir: Path, output: Path) -> None:
    """Build Mac Catalyst binding in MLC LLM"""
    deployment_target = os.environ.get("MLC_MACABI_DEPLOYMENT_TARGET", "18.0")
    macabi_arch = os.environ.get("MLC_MACABI_ARCH", "").strip() or "arm64"
    logger.info("Build macabi binding (deployment target %s)", deployment_target)
    cmd = [
        "bash",
        mlc_llm_source_dir / "ios" / "prepare_libs.sh",
        "--catalyst",
        "--deployment-target",
        deployment_target,
    ]
    if macabi_arch:
        cmd += ["--arch", macabi_arch]
    subprocess.run(cmd, check=True, env=os.environ)

    # Copy built libraries back to output directory.
    build_dir = Path(f"build-maccatalyst-{macabi_arch}")
    for static_library in (build_dir / "lib").iterdir():
        dst_path = str(output / "lib" / static_library.name)
        logger.info('Copying "%s" to "%s"', static_library, dst_path)
        shutil.copy(static_library, dst_path)


def package(
    package_config_path: Path,
    mlc_llm_source_dir: Path,
    output: Path,
) -> None:
    """Python entrypoint of package."""
    logger.info('MLC LLM HOME: "%s"', mlc_llm_source_dir)

    # - Read package config.
    with open(package_config_path, "r", encoding="utf-8") as file:
        package_config = json.load(file)
    if not isinstance(package_config, dict):
        raise ValueError(
            "The content of MLC package config is expected to be a dict with "
            f'field "model_list". However, the content of "{package_config_path}" is not a dict.'
        )

    # - Read device.
    if "device" not in package_config:
        raise ValueError(f'JSON file "{package_config_path}" is required to have field "device".')
    device = package_config["device"]
    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f'The "device" field of JSON file {package_config_path} is expected to be one of '
            f'{SUPPORTED_DEVICES}, while "{device}" is given in the JSON.'
        )

    bundle_dir = output / "bundle"
    app_config_path = bundle_dir / "mlc-app-config.json"
    # - Build model libraries.
    model_lib_path_for_prepare_libs = build_model_library(
        package_config, device, bundle_dir, app_config_path
    )
    # - Validate model libraries.
    validate_model_lib(
        app_config_path,
        package_config_path,
        model_lib_path_for_prepare_libs,
        device,
        output,
    )

    # - Copy model libraries
    if device == "android":
        build_android_binding(mlc_llm_source_dir, output)
    elif device == "iphone":
        build_iphone_binding(mlc_llm_source_dir, output)
    elif device == "macabi":
        build_macabi_binding(mlc_llm_source_dir, output)
    else:
        assert False, "Cannot reach here"

    logger.info("All finished.")
