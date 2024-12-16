"""Entrypoint of all CLI commands from MLC LLM"""

import sys

from mlc_llm.support import logging
from mlc_llm.support.argparse import ArgumentParser

logging.enable_logging()


def main():
    """Entrypoint of all CLI commands from MLC LLM"""
    parser = ArgumentParser("MLC LLM Command Line Interface.")
    parser.add_argument(
        "subcommand",
        type=str,
        choices=[
            "compile",
            "convert_weight",
            "gen_config",
            "chat",
            "serve",
            "package",
            "calibrate",
            "router",
        ],
        help="Subcommand to to run. (choices: %(choices)s)",
    )
    parsed = parser.parse_args(sys.argv[1:2])
    # pylint: disable=import-outside-toplevel
    if parsed.subcommand == "compile":
        from mlc_llm.cli import compile as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "convert_weight":
        from mlc_llm.cli import convert_weight as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "gen_config":
        from mlc_llm.cli import gen_config as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "chat":
        from mlc_llm.cli import chat as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "serve":
        from mlc_llm.cli import serve as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "package":
        from mlc_llm.cli import package as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "calibrate":
        from mlc_llm.cli import calibrate as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "router":
        from mlc_llm.cli import router as cli

        cli.main(sys.argv[2:])
    else:
        raise ValueError(f"Unknown subcommand {parsed.subcommand}")
    # pylint: enable=import-outside-toplevel


if __name__ == "__main__":
    main()
