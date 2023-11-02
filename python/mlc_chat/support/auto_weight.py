"""Help functions for detecting weight paths and weight formats."""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .style import green, red

logger = logging.getLogger(__name__)

FOUND = green("Found")
NOT_FOUND = red("Not found")


def detect_weight(
    weight_path: Path,
    config_json_path: Path,
    weight_format: str = "auto",
) -> Tuple[Path, str]:
    """Detect the weight directory, and detect the weight format.

    Parameters
    ---------
    weight_path : pathlib.Path
        The path to weight files. If `weight_path` is not None, check if it exists. Otherwise, find
        `weight_path` in `config.json` or use the same directory as `config.json`.

    config_json_path: pathlib.Path
        The path to `config.json`.

    weight_format : str
        The hint for the weight format. If it is "auto", guess the weight format.
        Otherwise, check the weights are in that format.
        Available weight formats:
            - auto (guess the weight format)
            - huggingface-torch (validate via checking pytorch_model.bin.index.json)
            - huggingface-safetensor (validate via checking model.safetensors.index.json)
            - awq
            - ggml
            - gguf

    Returns
    -------
    weight_config_path : pathlib.Path
        The path that points to the weights config file or the weights directory.

    weight_format : str
        The valid weight format.
    """
    if weight_path is None:
        assert (
            config_json_path is not None and config_json_path.exists()
        ), "Please provide config.json path."

        # 1. Find the weight_path in config.json
        with open(config_json_path, encoding="utf-8") as i_f:
            config = json.load(i_f)
        if "weight_path" in config:
            weight_path = Path(config["weight_path"])
            logger.info('Found "weight_path" in config.json: %s', weight_path)
            if not weight_path.exists():
                raise ValueError(f"weight_path doesn't exist: {weight_path}")
        else:
            # 2. Find the weights file in the same directory as config.json
            weight_path = config_json_path.parent
    else:
        if not weight_path.exists():
            raise ValueError(f"weight_path doesn't exist: {weight_path}")

    logger.info("%s weights from directory: %s", FOUND, weight_path)

    # check weight format
    # weight_format = "auto", guess the weight format.
    # otherwise, check the weight format is valid.
    if weight_format == "auto":
        return _guess_weight_format(weight_path)

    if weight_format not in AVAILABLE_WEIGHT_FORMAT:
        raise ValueError(
            f"Available weight format list: {AVAILABLE_WEIGHT_FORMAT}, but got {weight_format}"
        )
    if weight_format in CHECK_FORMAT_METHODS:
        check_func = CHECK_FORMAT_METHODS[weight_format]
        weight_config_path = check_func(weight_path)
        if not weight_config_path:
            raise ValueError(f"The weight is not in {weight_format} format.")
    return weight_config_path, weight_format


def _guess_weight_format(weight_path: Path) -> Tuple[Path, str]:
    possible_formats: List[Tuple[Path, str]] = []
    for weight_format, check_func in CHECK_FORMAT_METHODS.items():
        weight_config_path = check_func(weight_path)
        if weight_config_path:
            possible_formats.append((weight_config_path, weight_format))

    if len(possible_formats) == 0:
        raise ValueError(
            "Fail to detect weight format. Use `--weight-format` to manually specify the format."
        )

    weight_config_path, selected_format = possible_formats[0]
    logger.info(
        "Using %s format now. Use `--weight-format` to manually specify the format.",
        selected_format,
    )
    return weight_config_path, selected_format


def _check_pytorch(weight_path: Path) -> Optional[Path]:
    pytorch_json_path = weight_path / "pytorch_model.bin.index.json"
    if pytorch_json_path.exists():
        logger.info("%s Huggingface PyTorch: %s", FOUND, pytorch_json_path)
        return pytorch_json_path
    logger.info("%s Huggingface PyTorch", NOT_FOUND)
    return None


def _check_safetensor(weight_path: Path) -> Optional[Path]:
    safetensor_json_path = weight_path / "model.safetensors.index.json"
    if safetensor_json_path.exists():
        logger.info("%s Huggingface Safetensor: %s", FOUND, safetensor_json_path)
        return safetensor_json_path
    logger.info("%s Huggingface Safetensor", NOT_FOUND)
    return None


CHECK_FORMAT_METHODS = {
    "huggingface-torch": _check_pytorch,
    "huggingface-safetensor": _check_safetensor,
}

# "awq", "ggml", "gguf" are not supported yet.
AVAILABLE_WEIGHT_FORMAT = ["huggingface-torch", "huggingface-safetensor"]
