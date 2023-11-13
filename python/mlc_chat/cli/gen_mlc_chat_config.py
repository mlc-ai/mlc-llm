"""Command line entrypoint of configuration generation."""
import logging
from pathlib import Path
from typing import Union

from mlc_chat.compiler import CONV_TEMPLATES, HELP, MODELS, QUANTIZATION, gen_config

from ..support.argparse import ArgumentParser
from ..support.auto_config import detect_config, detect_model_type

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def main():
    """Parse command line argumennts and call `mlc_llm.compiler.gen_config`."""
    parser = ArgumentParser("MLC LLM Configuration Generator")

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    parser.add_argument(
        "--model",
        type=detect_config,
        required=True,
        dest="config",
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=True,
        choices=list(QUANTIZATION.keys()),
        help=HELP["quantization"] + " (required, choices: %(choices)s)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto"] + list(MODELS.keys()),
        help=HELP["model_type"] + ' (default: "%(default)s", choices: %(choices)s)',
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        required=True,
        choices=list(CONV_TEMPLATES),
        help=HELP["conv_template"] + " (required, choices: %(choices)s)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help=HELP["max_sequence_length"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help=HELP["output_gen_mlc_chat_config"] + " (required)",
    )
    parsed = parser.parse_args()
    model = detect_model_type(parsed.model_type, parsed.config)
    gen_config(
        config=parsed.config,
        model=model,
        quantization=QUANTIZATION[parsed.quantization],
        conv_template=parsed.conv_template,
        max_sequence_length=parsed.max_sequence_length,
        output=parsed.output,
    )


if __name__ == "__main__":
    main()
