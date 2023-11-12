"""Command line entrypoint of configuration generation."""
import argparse
import logging
from pathlib import Path
from typing import Union

from mlc_chat.compiler import CONV_TEMPLATES, MODELS, QUANTIZATION, gen_config

from ..support.auto_config import detect_config, detect_model_type

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def main():
    """Parse command line argumennts and call `mlc_llm.compiler.gen_config`."""
    parser = argparse.ArgumentParser("MLC LLM Configuration Generator")

    def _parse_config(path: Union[str, Path]) -> Path:
        try:
            return detect_config(path)
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"No valid config.json in: {path}. Error: {err}")

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    parser.add_argument(
        "--config",
        type=_parse_config,
        required=True,
        help="Path to config.json file or to the directory that contains config.json, which is "
        "a HuggingFace standard that defines model architecture, for example, "
        "https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json. "
        "This `config.json` file is expected to colocate with other configurations, such as "
        "tokenizer configuration and `generation_config.json`.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=True,
        choices=list(QUANTIZATION.keys()),
        help="Quantization format.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto"] + list(MODELS.keys()),
        help="Model architecture, for example, llama. If not set, it is inferred "
        "from the config.json file. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        required=True,
        choices=list(CONV_TEMPLATES),
        help='Conversation template. It depends on how the model is tuned. Use "LM" for vanilla '
        "base model",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Option to override the maximum sequence length supported by the model. "
        "An LLM is usually trained with a fixed maximum sequence length, which is usually "
        "explicitly specified in model spec. By default, if this option is not set explicitly, "
        "the maximum sequence length is determined by `max_sequence_length` or "
        "`max_position_embeddings` in config.json, which can be inaccuate for some models.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help="The output directory for generated configurations, including `mlc-chat-config.json`, "
        "and tokenizer configuration.",
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
