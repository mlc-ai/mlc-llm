"""Command line entrypoint of package."""

import os
from pathlib import Path
from typing import Union

from mlc_llm.interface.help import HELP
from mlc_llm.interface.package import package
from mlc_llm.support.argparse import ArgumentParser


def main(argv):
    """Parse command line arguments and call `mlc_llm.interface.package`."""
    parser = ArgumentParser("MLC LLM Package CLI")

    def _parse_package_config(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.exists():
            raise ValueError(
                f"Path {str(path)} is expected to be a JSON file, but the file does not exist."
            )
        if not path.is_file():
            raise ValueError(f"Path {str(path)} is expected to be a JSON file.")
        return path

    def _parse_mlc_llm_source_dir(path: str) -> Path:
        os.environ["MLC_LLM_SOURCE_DIR"] = path
        return Path(path)

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    parser.add_argument(
        "--package-config",
        type=_parse_package_config,
        default="mlc-package-config.json",
        help=HELP["config_package"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--mlc-llm-source-dir",
        type=_parse_mlc_llm_source_dir,
        default=os.environ.get("MLC_LLM_SOURCE_DIR", None),
        help=HELP["mlc_llm_source_dir"]
        + " (default: the $MLC_LLM_SOURCE_DIR environment variable)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        default="dist",
        help=HELP["output_package"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)
    if parsed.mlc_llm_source_dir is None:
        raise ValueError(
            "MLC LLM home is not specified. "
            "Please obtain a copy of MLC LLM source code by "
            "cloning https://github.com/mlc-ai/mlc-llm, and set environment variable "
            '"MLC_LLM_SOURCE_DIR=path/to/mlc-llm"'
        )
    package(
        package_config_path=parsed.package_config,
        mlc_llm_source_dir=parsed.mlc_llm_source_dir,
        output=parsed.output,
    )
