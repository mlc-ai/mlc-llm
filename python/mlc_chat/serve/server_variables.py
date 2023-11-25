"""Global server variables shared by multiple entrypoint files."""
# pylint: disable=global-statement
import argparse
import json
import os

from . import async_engine, config

engine: async_engine.AsyncEngine
hosted_model_id: str


def parse_args_and_initialize() -> argparse.Namespace:
    """Parse the server arguments and initialize the engine."""

    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, required=True)
    args.add_argument("--device", type=str, default="auto")
    args.add_argument("--model-lib-path", type=str, default="")
    args.add_argument("--batch-size", type=int, default=80)
    args.add_argument("--max-total-seq-length", type=int, default=16800)

    args.add_argument("--host", type=str, default="127.0.0.1", help="host name")
    args.add_argument("--port", type=int, default=8000, help="port")
    args.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    args.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    args.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    args.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")

    parsed = args.parse_args()
    assert parsed.batch_size % 16 == 0
    assert parsed.max_total_seq_length >= 2048

    global hosted_model_id
    _, hosted_model_id = os.path.split(parsed.model)
    if hosted_model_id == "":
        raise ValueError(
            f"Invalid model id {parsed.model}. "
            "Please make sure the model id does not end with slash."
        )

    # Initialize model loading info and KV cache config
    model_info = async_engine.ModelInfo(
        model=parsed.model,
        device=parsed.device,
        model_lib_path=parsed.model_lib_path if parsed.model_lib_path != "" else None,
    )
    kv_cache_config = config.KVCacheConfig(
        max_num_sequence=parsed.batch_size, max_total_sequence_length=parsed.max_total_seq_length
    )
    # Create engine and start the background loop
    global engine
    engine = async_engine.AsyncEngine(model_info, kv_cache_config)

    return parsed
