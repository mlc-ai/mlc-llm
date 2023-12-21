"""Utility functions for mlc-serve"""
from mlc_serve.logging_utils import configure_logging
from pathlib import Path
import os
import torch
import random
import argparse


def get_default_mlc_serve_argparser(description=""):
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
    return args
