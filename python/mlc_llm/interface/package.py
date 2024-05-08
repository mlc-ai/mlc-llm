"""Python entrypoint of package."""

import dataclasses
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Literal

from tvm.contrib import cc

from mlc_llm.chat_module import ChatConfig, _get_chat_config, _get_model_path
from mlc_llm.interface import jit
from mlc_llm.support import logging, style

logging.enable_logging()
logger = logging.getLogger(__name__)


def _get_model_libs(lib_path: Path) -> List[str]:
    """Get the model lib prefixes in the given static lib path."""
    global_symbol_map = cc.get_global_symbol_section_map(lib_path)
    libs = []
    suffix = "___tvm_dev_mblob"
    for name, _ in global_symbol_map.items():
        if name.endswith(suffix):
            model_lib = name[: -len(suffix)]
            if model_lib.startswith("_"):
                model_lib = model_lib[1:]
            libs.append(model_lib)
    return libs


def validate_model_lib(  # pylint: disable=too-many-locals
    app_config_path: Path, device: Literal["iphone", "android"], output: Path
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

    for model, model_lib_path in app_config["model_lib_path_for_prepare_libs"].items():
        model_lib_path = os.path.join(model_lib_path)
        lib_path_valid = os.path.isfile(model_lib_path)
        if not lib_path_valid:
            raise RuntimeError(f"Cannot find file {model_lib_path} as an {device} model library")
        tar_list.append(model_lib_path)
        model_set.add(model)

    os.makedirs(output / "lib", exist_ok=True)
    lib_path = (
        output / "lib" / ("libmodel_iphone.a" if device == "iphone" else "libmodel_android.a")
    )

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
            logger.info(
                "ValidationError: model_lib=%s specified for model_id=%s "
                "is not included in model_lib_path_for_prepare_libs field, "
                "This will cause the specific model not being able to load, "
                "please check %s.",
                model_lib,
                model_id,
                str(app_config_path),
            )
            error_happened = True

        model_prefix_pattern = model_lib.replace("-", "_") + "___tvm_dev_mblob"
        if (
            model_prefix_pattern not in global_symbol_map
            and "_" + model_prefix_pattern not in global_symbol_map
        ):
            model_lib_path = app_config["model_lib_path_for_prepare_libs"][model_lib]
            logger.info(
                "ValidationError:\n"
                "\tmodel_lib %s requested in %s is not found in %s\n"
                "\tspecifically the model_lib for %s in model_lib_path_for_prepare_libs.\n"
                "\tcurrent available model_libs in %s: %s",
                model_lib,
                str(app_config_path),
                str(lib_path),
                model_lib_path,
                str(lib_path),
                str(available_model_libs),
            )
            error_happened = True

    if not error_happened:
        logger.info(style.green("Validation pass"))
    else:
        logger.info(style.red("Validation failed"))
        sys.exit(255)


def package(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    package_config_path: Path,
    device: Literal["iphone", "android"],
    output: Path,
) -> None:
    """Python entrypoint of package."""
    # - Read package config.
    with open(package_config_path, "r", encoding="utf-8") as file:
        package_config = json.load(file)
    if not isinstance(package_config, dict):
        raise ValueError(
            "The content of MLC package config is expected to be a dict with "
            f'field "model_list". However, the content of "{package_config_path}" is not a dict.'
        )

    # - Create the bundle directory.
    bundle_dir = output / "bundle"
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
        model_path_and_config_file_path = _get_model_path(model)
        model_path = Path(model_path_and_config_file_path[0])
        config_file_path = model_path_and_config_file_path[1]
        chat_config = _get_chat_config(
            config_file_path, user_chat_config=ChatConfig.from_dict(overrides)
        )
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
                    chat_config=asdict(chat_config),
                    device=device,
                    system_lib_prefix=model_lib,
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
            if not os.path.isfile(model_path / "ndarray-cache.json"):
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
            bundle_model_weight_path = bundle_dir / model_path.name
            logger.info(
                'Bundle weight for model "%s". Copying weights from "%s" to "%s".',
                model_id,
                model_path,
                bundle_model_weight_path,
            )
            if bundle_model_weight_path.exists():
                shutil.rmtree(bundle_model_weight_path)
            shutil.copytree(model_path, bundle_model_weight_path)
            app_config_model_entry["model_path"] = model_path.name
        else:
            app_config_model_entry["model_url"] = model.replace("HF://", "https://huggingface.co/")

        # - estimated_vram_bytes
        app_config_model_entry["estimated_vram_bytes"] = estimated_vram_bytes

        app_config_model_list.append(app_config_model_entry)

    # - Dump "mlc-app-config.json".
    app_config_json_str = json.dumps(
        {
            "model_list": app_config_model_list,
            "model_lib_path_for_prepare_libs": model_lib_path_for_prepare_libs,
        },
        indent=2,
    )
    app_config_path = bundle_dir / "mlc-app-config.json"
    with open(app_config_path, "w", encoding="utf-8") as file:
        print(app_config_json_str, file=file)
        logger.info(
            'Dump the app config below to "dist/bundle/mlc-app-config.json":\n%s',
            style.green(app_config_json_str),
        )

    # - Validate model libraries.
    validate_model_lib(app_config_path, device, output)
