"""Entrypoint of RESTful HTTP request server in MLC LLM"""
import argparse
import json
import os

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from .. import async_engine, config
from .server_context import EngineClass, ServerContext


def parse_args_and_initialize() -> argparse.Namespace:
    """Parse the server arguments and initialize the engine."""

    args = argparse.ArgumentParser()  # pylint: disable=redefined-outer-name
    args.add_argument("--model", type=str, required=True)
    args.add_argument("--device", type=str, default="auto")
    args.add_argument("--model-lib-path", type=str, default="")
    args.add_argument("--max-batch-size", type=int, default=80)
    args.add_argument("--max-total-seq-length", type=int, default=16800)
    args.add_argument("--use-threaded-engine", action="store_true")
    args.add_argument("--enable-tracing", action="store_true")

    args.add_argument("--host", type=str, default="127.0.0.1", help="host name")
    args.add_argument("--port", type=int, default=8000, help="port")
    args.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    args.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    args.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    args.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")

    parsed = args.parse_args()
    assert parsed.max_total_seq_length >= 2048

    _, hosted_model = os.path.split(parsed.model)
    if hosted_model == "":
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
        max_num_sequence=parsed.max_batch_size,
        max_total_sequence_length=parsed.max_total_seq_length,
    )
    # Create engine and start the background loop
    engine: EngineClass
    if parsed.use_threaded_engine:
        engine = async_engine.AsyncThreadedEngine(
            model_info, kv_cache_config, enable_tracing=parsed.enable_tracing
        )
    else:
        engine = async_engine.AsyncEngine(
            model_info, kv_cache_config, enable_tracing=parsed.enable_tracing
        )

    ServerContext.add_model(hosted_model, engine)
    return parsed


if __name__ == "__main__":
    # Parse the arguments and initialize the asynchronous engine.
    args: argparse.Namespace = parse_args_and_initialize()
    app = fastapi.FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers from subdirectories.
    from ..entrypoints import debug_entrypoints, openai_entrypoints

    app.include_router(openai_entrypoints.app)
    app.include_router(debug_entrypoints.app)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
