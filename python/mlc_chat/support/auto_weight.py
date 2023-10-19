"""Help functions for detecting weight paths and weight formats."""
import json
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def detect_weight(
    weight_path: Path, config_json_path: Path, weight_format: str = "auto"
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
            - PyTorch (validate via checking pytorch_model.bin.index.json)
            - SafeTensor (validate via checking model.safetensors.index.json)
            - AWQ
            - GGML/GGUF

    Returns
    -------
    weight_path : pathlib.Path
        The path that points to the weights.

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

    logger.info("Loading weights from directory: %s", weight_path)

    # check weight format
    # weight_format = "auto", guess the weight format.
    # otherwise, check the weight format is valid.
    if weight_format == "auto":
        weight_format = _guess_weight_format(weight_path)

    if weight_format not in AVAILABLE_WEIGHT_FORMAT:
        raise ValueError(
            f"Available weight format list: {AVAILABLE_WEIGHT_FORMAT}, but got {weight_format}"
        )
    if weight_format in CHECK_FORMAT_METHODS:
        check_func = CHECK_FORMAT_METHODS[weight_format]
        if not check_func(weight_path):
            raise ValueError(f"The weight is not in {weight_format} format.")
    return weight_path, weight_format


def _guess_weight_format(weight_path: Path):
    possible_formats = []
    for weight_format, check_func in CHECK_FORMAT_METHODS.items():
        if check_func(weight_path):
            possible_formats.append(weight_format)

    if len(possible_formats) == 0:
        raise ValueError(
            "Fail to detect weight format. Use `--weight-format` to manually specify the format."
        )

    selected_format = possible_formats[0]
    logging.info(
        "Using %s format now. Use `--weight-format` to manually specify the format.",
        selected_format,
    )
    return selected_format


def _check_pytorch(weight_path: Path):
    pytorch_json_path = weight_path / "pytorch_model.bin.index.json"
    result = pytorch_json_path.exists()
    if result:
        logger.info("[Y] Found Huggingface PyTorch: %s", pytorch_json_path)
    else:
        logger.info("[X] Not found: Huggingface PyTorch")
    return result


def _check_safetensor(weight_path: Path):
    safetensor_json_path = weight_path / "model.safetensors.index.json"
    result = safetensor_json_path.exists()
    if result:
        logger.info("[Y] Found SafeTensor: %s", safetensor_json_path)
    else:
        logger.info("[X] Not found: SafeTensor")
    return result


CHECK_FORMAT_METHODS = {
    "PyTorch": _check_pytorch,
    "SafeTensor": _check_safetensor,
}

AVAILABLE_WEIGHT_FORMAT = ["PyTorch", "SafeTensor", "GGML", "GGUF", "AWQ"]
