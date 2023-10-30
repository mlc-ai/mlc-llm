"""Help function for detecting the model configuration file `config.json`"""
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .style import green

if TYPE_CHECKING:
    from mlc_chat.compiler import Model  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

FOUND = green("Found")


def detect_config(config: Union[str, Path]) -> Path:
    """Detect and return the path that points to config.json. If `config` is a directory,
    it looks for config.json below it.

    Parameters
    ---------
    config : Union[str, pathlib.Path]
        The preset name of the model, or the path to `config.json`, or the directory containing
        `config.json`.

    Returns
    -------
    config_json_path : pathlib.Path
        The path points to config.json.
    """
    from mlc_chat.compiler import (  # pylint: disable=import-outside-toplevel
        MODEL_PRESETS,
    )

    if isinstance(config, str) and config in MODEL_PRESETS:
        content = MODEL_PRESETS[config]
        temp_file = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
            suffix=".json",
            delete=False,
        )
        logger.info("%s preset model configuration: %s", FOUND, temp_file.name)
        config_path = Path(temp_file.name)
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(content, config_file, indent=2)
    else:
        config_path = Path(config)
    if not config_path.exists():
        raise ValueError(f"{config_path} does not exist.")

    if config_path.is_dir():
        # search config.json under config path
        config_json_path = config_path / "config.json"
        if not config_json_path.exists():
            raise ValueError(f"Fail to find config.json under {config_path}.")
    else:
        config_json_path = config_path

    logger.info("%s model configuration: %s", FOUND, config_json_path)
    return config_json_path


def detect_model_type(model_type: str, config: Path) -> "Model":
    """Detect the model type from the configuration file. If `model_type` is "auto", it will be
    inferred from the configuration file. Otherwise, it will be used as the model type, and sanity
    check will be performed.

    Parameters
    ----------
    model_type : str
        The model type, for example, "llama".

    config : pathlib.Path
        The path to config.json.

    Returns
    -------
    model : mlc_chat.compiler.Model
        The model type.
    """

    from mlc_chat.compiler import (  # pylint: disable=import-outside-toplevel
        MODELS,
        Model,
    )

    if model_type == "auto":
        with open(config, "r", encoding="utf-8") as config_file:
            cfg = json.load(config_file)
        if "model_type" not in cfg:
            raise ValueError(
                f"'model_type' not found in: {config}. "
                f"Please explicitly specify `--model-type` instead"
            )
        model_type = cfg["model_type"]
        logger.info("%s Model type: %s", FOUND, model_type)
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available ones: {list(MODELS.keys())}")
    return MODELS[model_type]
