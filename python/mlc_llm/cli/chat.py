"""Command line entrypoint of chat."""

from mlc_llm.help import HELP
from mlc_llm.interface.chat import ChatConfigOverride, chat
from mlc_llm.support.argparse import ArgumentParser


def main(argv):
    """Parse command line arguments and call `mlc_llm.interface.chat`."""
    parser = ArgumentParser("MLC LLM Chat CLI")

    parser.add_argument(
        "model",
        type=str,
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="O2",
        help=HELP["opt"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_deploy"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--overrides",
        type=ChatConfigOverride.from_str,
        default="",
        help=HELP["chatconfig_overrides"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        default=None,
        help=HELP["model_lib"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)
    chat(
        model=parsed.model,
        device=parsed.device,
        opt=parsed.opt,
        overrides=parsed.overrides,
        model_lib=parsed.model_lib,
    )
