"""Command line entrypoint of weight conversion."""
import argparse
import logging
from pathlib import Path
from typing import Union

from mlc_chat.compiler import MODELS, QUANTIZATION, convert_weight

from ..support.auto_config import detect_config, detect_model_type
from ..support.auto_target import detect_device
from ..support.auto_weight import detect_weight

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def main():
    """Parse command line argumennts and apply quantization."""

    def _parse_config(path: Union[str, Path]) -> Path:
        try:
            return detect_config(path)
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"No valid config.json in: {path}. Error: {err}")

    def _parse_source(path: Union[str, Path], config_path: Path) -> Path:
        if path == "auto":
            return config_path.parent
        path = Path(path)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"Model source does not exist: {path}")
        return path

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    parser = argparse.ArgumentParser("MLC AutoLLM Quantization Framework")
    parser.add_argument(
        "--config",
        type=_parse_config,
        required=True,
        help="Path to config.json file or to the directory that contains config.json, which is "
        "a HuggingFace standard that defines model architecture, for example, "
        "https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        help="The path to original model weight, infer from `config` if missing. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--source-format",
        type=str,
        choices=["auto", "huggingface-torch", "huggingface-safetensor", "awq"],
        default="auto",
        help="The format of source model weight, infer from `config` if missing. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=True,
        choices=list(QUANTIZATION.keys()),
        help="Quantization format, for example `q4f16_1`.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto"] + list(MODELS.keys()),
        help="Model architecture, for example, llama. If not set, it is inferred "
        "from the config.json file. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=detect_device,
        help="The device used to do quantization, for example, / `cuda:0`. "
        "Detect from local environment if not specified. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help="The output directory to save the quantized model weight, "
        "will contain `params_shard_*.bin` and `ndarray-cache.json`.",
    )

    parsed = parser.parse_args()
    parsed.source, parsed.source_format = detect_weight(
        weight_path=_parse_source(parsed.source, parsed.config),
        config_json_path=parsed.config,
        weight_format=parsed.source_format,
    )
    model = detect_model_type(parsed.model_type, parsed.config)
    convert_weight(
        config=parsed.config,
        quantization=QUANTIZATION[parsed.quantization],
        model=model,
        device=parsed.device,
        source=parsed.source,
        source_format=parsed.source_format,
        output=parsed.output,
    )


if __name__ == "__main__":
    main()
