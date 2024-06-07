"""Command line entrypoint of compilation."""

import argparse
import json
import re
from functools import partial
from pathlib import Path
from typing import Union

from mlc_llm.interface.compile import (  # pylint: disable=redefined-builtin
    ModelConfigOverride,
    OptimizationFlags,
    compile,
)
from mlc_llm.interface.help import HELP
from mlc_llm.model import MODELS
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.auto_config import (
    detect_mlc_chat_config,
    detect_model_type,
    detect_quantization,
)
from mlc_llm.support.auto_target import detect_system_lib_prefix, detect_target_and_host


def main(argv):
    """Parse command line arguments and call `mlc_llm.compiler.compile`."""

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.is_dir():
            raise argparse.ArgumentTypeError(f"Output cannot be a directory: {path}")
        parent = path.parent
        if not parent.is_dir():
            raise argparse.ArgumentTypeError(f"Directory does not exist: {parent}")
        return path

    def _parse_dir(path: Union[str, Path], auto_create: bool = False) -> Path:
        path = Path(path)
        if not auto_create and not path.is_dir():
            raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
        if auto_create and not path.is_dir():
            path.mkdir(parents=True)
        return path

    def _check_system_lib_prefix(prefix: str) -> str:
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        if prefix == "" or re.match(pattern, prefix):
            return prefix
        raise argparse.ArgumentTypeError(
            "Invalid prefix. It should only consist of "
            "numbers (0-9), alphabets (A-Z, a-z) and underscore (_)."
        )

    parser = ArgumentParser("mlc_llm compile")
    parser.add_argument(
        "model",
        type=detect_mlc_chat_config,
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=list(QUANTIZATION.keys()),
        help=HELP["quantization"]
        + " (default: look up mlc-chat-config.json, choices: %(choices)s)",
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
        default="auto",
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
    parser.add_argument(
        "--debug-dump",
        type=partial(_parse_dir, auto_create=True),
        default=None,
        help=HELP["debug_dump"] + " (default: %(default)s)",
    )
    parsed = parser.parse_args(argv)
    target, build_func = detect_target_and_host(parsed.device, parsed.host)
    parsed.model_type = detect_model_type(parsed.model_type, parsed.model)
    parsed.quantization = detect_quantization(parsed.quantization, parsed.model)
    parsed.system_lib_prefix = detect_system_lib_prefix(
        parsed.device, parsed.system_lib_prefix, parsed.model_type.name, parsed.quantization.name
    )
    with open(parsed.model, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    compile(
        config=config,
        quantization=parsed.quantization,
        model_type=parsed.model_type,
        target=target,
        opt=parsed.opt,
        build_func=build_func,
        system_lib_prefix=parsed.system_lib_prefix,
        output=parsed.output,
        overrides=parsed.overrides,
        debug_dump=parsed.debug_dump,
    )
