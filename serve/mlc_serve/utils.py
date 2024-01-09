"""Utility functions for mlc-serve"""

from pathlib import Path
import os
import torch
import random
import argparse

from mlc_serve.engine import get_engine_config
from mlc_serve.logging_utils import configure_logging
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule


def get_default_mlc_serve_argparser(description="", allow_override=False):
    if allow_override:
        parser = argparse.ArgumentParser(
            description=description, conflict_handler="resolve"
        )
    else:
        parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--use-sync-engine", action="store_true")
    parser.add_argument("--max-num-sequences", type=int, default=8)
    parser.add_argument("--num-sequences-to-sample", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--min-decode-steps", type=int, default=32)
    parser.add_argument("--max-decode-steps", type=int, default=56)
    parser.add_argument("--debug-logging", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def postproc_mlc_serve_args(args):
    log_level = "DEBUG" if args.debug_logging else "INFO"
    configure_logging(enable_json_logs=False, log_level=log_level)
    args.model_artifact_path = Path(os.path.join(args.artifact_path, args.local_id))
    if not os.path.exists(args.model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")

    args.use_staging_engine = not args.use_sync_engine

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)


def create_mlc_engine(args: argparse.Namespace):
    engine_config = get_engine_config(
        {
            "use_staging_engine": args.use_staging_engine,
            "max_num_sequences": args.max_num_sequences,
            "max_input_len": args.max_input_len,
            "min_decode_steps": args.min_decode_steps,
            "max_decode_steps": args.max_decode_steps,
        }
    )

    # TODO(@team): There is a type mismatch in the definition. Let's fix this when have time.
    if args.use_staging_engine:
        engine = StagingInferenceEngine(  # type: ignore
            tokenizer_module=HfTokenizerModule(args.model_artifact_path),
            model_module_loader=PagedCacheModelModule,  # type: ignore
            model_module_loader_kwargs={
                "model_artifact_path": args.model_artifact_path,
                "engine_config": engine_config,
            },
        )
        engine.start()
    else:
        engine = SynchronousInferenceEngine(  # type: ignore
            PagedCacheModelModule(  # type: ignore
                model_artifact_path=args.model_artifact_path,
                engine_config=engine_config,
            )
        )
    return engine
