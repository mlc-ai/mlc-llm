"""Command line entrypoint of configuration generation."""

from pathlib import Path
from typing import Union

from mlc_llm.interface.gen_config import CONV_TEMPLATES, gen_config
from mlc_llm.interface.help import HELP
from mlc_llm.model import MODELS
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.auto_config import detect_config, detect_model_type


def main(argv):
    """Parse command line argumennts and call `mlc_llm.compiler.gen_config`."""
    parser = ArgumentParser("MLC LLM Configuration Generator")

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    parser.add_argument(
        "config",
        type=detect_config,
        help=HELP["config"] + " (required)",
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
        "--context-window-size",
        type=int,
        default=None,
        help=HELP["context_window_size"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=None,
        help=HELP["sliding_window_size"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=None,
        help=HELP["prefill_chunk_size"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--attention-sink-size",
        type=int,
        default=None,
        help=HELP["attention_sink_size"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--tensor-parallel-shards",
        type=int,
        default=None,
        help=HELP["tensor_parallel_shards"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--pipeline-parallel-stages",
        type=int,
        default=None,
        help=HELP["pipeline_parallel_stages"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--disaggregation",
        type=bool,
        default=None,
        help=HELP["disaggregation"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help=HELP["max_batch_size"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help=HELP["output_gen_mlc_chat_config"] + " (required)",
    )
    parsed = parser.parse_args(argv)
    model = detect_model_type(parsed.model_type, parsed.config)
    gen_config(
        config=parsed.config,
        model=model,
        quantization=QUANTIZATION[parsed.quantization],
        conv_template=parsed.conv_template,
        context_window_size=parsed.context_window_size,
        sliding_window_size=parsed.sliding_window_size,
        prefill_chunk_size=parsed.prefill_chunk_size,
        attention_sink_size=parsed.attention_sink_size,
        tensor_parallel_shards=parsed.tensor_parallel_shards,
        pipeline_parallel_stages=parsed.pipeline_parallel_stages,
        disaggregation=parsed.disaggregation,
        max_batch_size=parsed.max_batch_size,
        output=parsed.output,
    )
