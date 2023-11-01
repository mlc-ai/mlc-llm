import argparse
import logging
import logging.config
import os

import uvicorn

from mlc_llm import utils

from .api import create_app
from .engine import AsyncEngineConnector
from .engine.local import LocalProcessInferenceEngine
from .model.paged_cache_model import PagedCacheModelModule


def parse_args():
    # Example
    # python build.py --model vicuna-v1-7b --quantization q4f16_ft --use-cache=0 --max-seq-len 768 --batched
    # python tests/python/test_batched.py --local-id vicuna-v1-7b-q4f16_ft
    #
    # For Disco:
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --build-model-only --num-shards 2
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --convert-weight-only
    # /opt/bin/cuda-reserve.py  --num-gpus 2 python -m mlc_serve --local-id vicuna-v1-7b-q0f16 --num-shards 2
    #
    # Profile the gpu memory usage, and use the maximum number of cache blocks possible:
    # /opt/bin/cuda-reserve.py  --num-gpus 2 python -m mlc_serve --local-id vicuna-v1-7b-q0f16 --num-shards 2 --max-num-batched-tokens 2560 --max-input-len 256

    args = argparse.ArgumentParser()
    args.add_argument("--host", type=str, default="127.0.0.1")
    args.add_argument("--port", type=int, default=8000)
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--num-shards", type=int, default=1)
    args.add_argument("--max-num-batched-tokens", type=int, default=-1)
    args.add_argument("--max-input-len", type=int, default=-1)
    args.add_argument("--debug-logging", action="store_true")
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    return parsed


def setup_logging(args):
    level = "INFO"
    if args.debug_logging:
        level = "DEBUG"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "level": level,  # Set the handler's log level to DEBUG
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,  # Set the logger's log level to DEBUG
        },
        "mlc_serve.engine.local": {"level": level},
    }
    logging.config.dictConfig(logging_config)


def run_server():
    args = parse_args()
    setup_logging(args)
    model_module = PagedCacheModelModule(
        args.model,
        args.artifact_path,
        args.quantization.name,
        args.num_shards,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_input_len=args.max_input_len,
    )

    engine = LocalProcessInferenceEngine(
        model_module,
        max_batched_tokens=args.max_num_batched_tokens,
    )
    connector = AsyncEngineConnector(engine)
    app = create_app(connector)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    run_server()
