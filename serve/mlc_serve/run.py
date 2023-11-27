import argparse
import logging
import logging.config
import os
import uvicorn
from pathlib import Path

from .api import create_app
from .engine import AsyncEngineConnector, get_engine_config
from .engine.staging_engine import StagingInferenceEngine
from .engine.sync_engine import SynchronousInferenceEngine
from .model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from .logging_utils import configure_logging


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
    args.add_argument("--use-staging-engine", action="store_true")
    args.add_argument("--max-num-sequences", type=int, default=8)
    args.add_argument("--max-input-len", type=int, default=512)
    args.add_argument("--min-decode-steps", type=int, default=12)
    args.add_argument("--max-decode-steps", type=int, default=16)
    args.add_argument("--prompt-allocate-ratio", type=float, default=2.0)
    args.add_argument("--debug-logging", action="store_true")
    parsed = args.parse_args()
    return parsed


def create_engine(
    args: argparse.Namespace,
):
    """
      `model_artifact_path` has the following structure
      |- compiled artifact (.so)
      |- `build_config.json`: stores compile-time info, such as `num_shards` and `quantization`.
      |- params/ : stores weights in mlc format and `ndarray-cache.json`.
      |            `ndarray-cache.json` is especially important for Disco.
      |- model/ : stores info from hf model cards such as max context length and tokenizer
    """
    model_artifact_path = Path(os.path.join(args.artifact_path, args.local_id))
    if not os.path.exists(model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")

    # Set the engine config
    engine_config = get_engine_config({
        "use_staging_engine": args.use_staging_engine,
        "max_num_sequences": args.max_num_sequences,
        "max_input_len": args.max_input_len,
        "min_decode_steps": args.min_decode_steps,
        "max_decode_steps": args.max_decode_steps,
        "prompt_allocate_ratio": args.prompt_allocate_ratio
    })

    if args.use_staging_engine:
        return StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": model_artifact_path,
                "engine_config": engine_config,
            },
        )
    else:
        return SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path = model_artifact_path,
                engine_config = engine_config,
            )
        )


def run_server():
    args = parse_args()

    log_level = "DEBUG" if args.debug_logging else "INFO"
    configure_logging(enable_json_logs=True, log_level=log_level)

    engine = create_engine(args)
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
