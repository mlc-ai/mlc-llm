"""Command line entrypoint of compilation."""
import argparse
import json
import re
from pathlib import Path
from typing import Union

from mlc_chat.compiler import (  # pylint: disable=redefined-builtin
    HELP,
    MODELS,
    QUANTIZATION,
    ModelConfigOverride,
    OptimizationFlags,
    compile,
)

from ..support.argparse import ArgumentParser
from ..support.auto_config import detect_mlc_chat_config, detect_model_type
from ..support.auto_target import detect_target_and_host


def main(argv):
    """Parse command line argumennts and call `mlc_llm.compiler.compile`."""

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.is_dir():
            raise argparse.ArgumentTypeError(f"Output cannot be a directory: {path}")
        parent = path.parent
        if not parent.is_dir():
            raise argparse.ArgumentTypeError(f"Directory does not exist: {parent}")
        return path

    def _check_system_lib_prefix(prefix: str) -> str:
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        if prefix == "" or re.match(pattern, prefix):
            return prefix
        raise argparse.ArgumentTypeError(
            "Invalid prefix. It should only consist of "
            "numbers (0-9), alphabets (A-Z, a-z) and underscore (_)."
        )

    parser = ArgumentParser("MLC LLM Compiler")
    parser.add_argument(
        "--model",
        type=detect_mlc_chat_config,
        required=True,
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto"] + list(MODELS.keys()),
        help=HELP["model_type"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_compile"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--host",
        type=str,
        default="auto",
        help=HELP["host"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--opt",
        type=OptimizationFlags.from_str,
        default="O2",
        help=HELP["opt"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--system-lib-prefix",
        type=str,
        default="",
        help=HELP["system_lib_prefix"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help=HELP["output_compile"] + " (required)",
    )
    parser.add_argument(
        "--overrides",
        type=ModelConfigOverride.from_str,
        default="",
        help=HELP["overrides"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)
    target, build_func = detect_target_and_host(parsed.device, parsed.host)
    parsed.model_type = detect_model_type(parsed.model_type, parsed.model)
    with open(parsed.model, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    compile(
        config=config["model_config"],
        quantization=QUANTIZATION[config["quantization"]],
        model_type=parsed.model_type,
        target=target,
        opt=parsed.opt,
        build_func=build_func,
        system_lib_prefix=parsed.system_lib_prefix,
        output=parsed.output,
        overrides=parsed.overrides,
    )
