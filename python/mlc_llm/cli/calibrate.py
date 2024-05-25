"""Command line entrypoint of calibration."""

from mlc_llm.help import HELP
from mlc_llm.interface.calibrate import calibrate
from mlc_llm.support.argparse import ArgumentParser


def main(argv):
    """Main entrypoint for calibration."""
    parser = ArgumentParser("MLC LLM Calibration CLI")
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
        "--output", "-o", type=str, required=True, help=HELP["output_calibration"] + " (required)"
    )
    # Download dataset from
    # https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    parser.add_argument(
        "--dataset", type=str, required=True, help=HELP["calibration_dataset"] + " (required)"
    )

    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=16,
        help=HELP["num_calibration_samples"] + ' (default: "%(default)s")',
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=HELP["seed_calibrate"] + ' (default: "%(default)s")',
    )

    # optional arguments for the engine
    parser.add_argument("--max-batch-size", type=int, help=HELP["max_batch_size"])
    parser.add_argument(
        "--max-total-seq-length", type=int, help=HELP["max_total_sequence_length_serve"]
    )
    parser.add_argument("--prefill-chunk-size", type=int, help=HELP["prefill_chunk_size_serve"])
    parser.add_argument("--max-history-size", type=int, help=HELP["max_history_size_serve"])
    parser.add_argument(
        "--gpu-memory-utilization", type=float, help=HELP["gpu_memory_utilization_serve"]
    )

    parsed = parser.parse_args(argv)
    calibrate(
        model=parsed.model,
        device=parsed.device,
        model_lib=parsed.model_lib,
        output=parsed.output,
        dataset=parsed.dataset,
        num_calibration_samples=parsed.num_calibration_samples,
        max_batch_size=parsed.max_batch_size,
        max_total_sequence_length=parsed.max_total_seq_length,
        prefill_chunk_size=parsed.prefill_chunk_size,
        max_history_size=parsed.max_history_size,
        gpu_memory_utilization=parsed.gpu_memory_utilization,
        seed=parsed.seed,
    )
