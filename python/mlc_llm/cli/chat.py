"""Command line entrypoint of chat."""

from mlc_llm.interface.chat import ModelConfigOverride, chat
from mlc_llm.interface.help import HELP
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
        "--device",
        type=str,
        default="auto",
        help=HELP["device_deploy"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        default=None,
        help=HELP["model_lib"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--overrides",
        type=ModelConfigOverride.from_str,
        default="",
        help=HELP["modelconfig_overrides"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--random-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Generate N random tokens and run the model automatically before "
            "entering the interactive prompt. The token count compensates for "
            "chat-template overhead so the reported prompt tokens match N. "
            "Use /stats afterwards to see tok/s and exact token counts. "
            '(default: "%(default)s")'
        ),
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=None,
        metavar="M",
        help=(
            "Maximum number of tokens to generate in the decode step when "
            "--random-tokens is used. Has no effect on subsequent interactive "
            "prompts. "
            '(default: "%(default)s")'
        ),
    )
    parsed = parser.parse_args(argv)
    chat(
        model=parsed.model,
        device=parsed.device,
        model_lib=parsed.model_lib,
        overrides=parsed.overrides,
        random_tokens=parsed.random_tokens,
        max_decode_tokens=parsed.max_decode_tokens,
    )
