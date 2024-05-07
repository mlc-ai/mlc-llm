"""Command line entrypoint of package."""

from pathlib import Path
from typing import Union

from mlc_llm.help import HELP
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

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    parser.add_argument(
        "package_config",
        type=_parse_package_config,
        help=HELP["config_package"] + " (required)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["iphone", "android"],
        required=True,
        help=HELP["device_package"] + " (required)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help=HELP["output_package"] + " (required)",
    )
    parsed = parser.parse_args(argv)
    package(
        package_config_path=parsed.package_config,
        device=parsed.device,
        output=parsed.output,
    )
