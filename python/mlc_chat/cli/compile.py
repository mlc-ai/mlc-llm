"""Command line entrypoint of compilation."""
import argparse
import json
import logging
from pathlib import Path
from typing import Union

from mlc_chat.compiler.compile import compile  # pylint: disable=redefined-builtin
from mlc_chat.compiler.model import MODELS, Model

from ..support.auto_config import detect_config
from ..support.auto_target import detect_target_and_host

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def _parse_config(path: Union[str, Path]) -> Path:
    try:
        return detect_config(Path(path))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"No valid config.json in: {path}. Error: {err}")


def _parse_output(path: Union[str, Path]) -> Path:
    path = Path(path)
    parent = path.parent
    if not parent.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {parent}")
    return path


def _parse_model_type(model_type: str, config: Path) -> Model:
    if model_type == "auto":
        with open(config, "r", encoding="utf-8") as config_file:
            cfg = json.load(config_file)
        if "model_type" not in cfg:
            raise ValueError(
                f"'model_type' not found in: {config}. "
                f"Please explicitly specify `--model-type` instead"
            )
        model_type = cfg["model_type"]
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available ones: {list(MODELS.keys())}")
    return MODELS[model_type]


def main():
    """Parse command line argumennts and call `mlc_llm.compiler.compile`."""
    parser = argparse.ArgumentParser("MLC LLM Compiler")
    parser.add_argument(
        "--config",
        type=_parse_config,
        required=True,
        help="Path to config.json file or to the directory that contains config.json, which is "
        "a HuggingFace standard that defines model architecture, for example, "
        "https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=True,
        choices=[
            "q0f16",
            "q0f32",
            "q3f16_1",
            "q3f32_1",
            "q4f16_1",
            "q4f16_ft",
            "q4f32_1",
        ],
        help="The quantization format. TBD",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto"] + list(MODELS.keys()),
        help="Model architecture, for example, llama. If not set, it is inferred "
        "from the config.json file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="The GPU device to compile the model to. If not set, it is inferred from locally "
        "available GPUs.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="auto",
        choices=[
            "auto",
            "arm",
            "arm64",
            "aarch64",
            "x86-64",
        ],
        help="The host CPU ISA to compile the model to. If not set, it is inferred from the "
        "local CPU.",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="",
        help="Optimization flags.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help="The name of the output file. The suffix determines if the output file is a "
        "shared library or a static library. Available suffixes: "
        "1) Linux: .so (shared), .a (static); "
        "2) macOS: .dylib (shared), .a (static); "
        "3) Windows: .dll (shared), .lib (static); "
        "4) Android, iOS: .tar (static); "
        "5) Web: .wasm (web assembly)",
    )
    parsed = parser.parse_args()
    target, build_func = detect_target_and_host(parsed.device, parsed.host)
    parsed.model_type = _parse_model_type(parsed.model_type, parsed.config)
    compile(
        config=parsed.config,
        quantization=parsed.quantization,
        model_type=parsed.model_type,
        target=target,
        opt=parsed.opt,
        build_func=build_func,
        output=parsed.output,
    )


if __name__ == "__main__":
    main()
