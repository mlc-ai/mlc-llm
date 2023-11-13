"""Entrypoint of all CLI commands from MLC LLM"""
import logging
import sys

from mlc_chat.support.argparse import ArgumentParser

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def main():
    """Entrypoint of all CLI commands from MLC LLM"""
    parser = ArgumentParser("MLC LLM Command Line Interface.")
    parser.add_argument(
        "subcommand",
        type=str,
        choices=["compile", "convert_weight", "gen_mlc_chat_config"],
        help="Subcommand to to run. (choices: %(choices)s)",
    )
    parsed = parser.parse_args([sys.argv[1]])
    # pylint: disable=import-outside-toplevel
    if parsed.subcommand == "compile":
        from mlc_chat.cli import compile as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "convert_weight":
        from mlc_chat.cli import convert_weight as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "gen_mlc_chat_config":
        from mlc_chat.cli import gen_mlc_chat_config as cli

        cli.main(sys.argv[2:])
    else:
        raise ValueError(f"Unknown subcommand {parsed.subcommand}")
    # pylint: enable=import-outside-toplevel


if __name__ == "__main__":
    main()
