"""Command line entrypoint of serve."""

import dataclasses
import json
from io import StringIO
from typing import Literal, Optional

from mlc_llm.interface.help import HELP
from mlc_llm.interface.serve import serve
from mlc_llm.support import argparse
from mlc_llm.support.argparse import ArgumentParser


@dataclasses.dataclass
class EngineConfigOverride:  # pylint: disable=too-many-instance-attributes
    """Arguments for overriding engine config."""

    # Overrides for EngineConfig (runtime)
    max_num_sequence: Optional[int] = None
    max_total_seq_length: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    max_history_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    spec_draft_length: Optional[int] = None
    spec_tree_width: Optional[int] = None
    prefix_cache_mode: Optional[Literal["disable", "radix"]] = None
    prefix_cache_max_num_recycling_seqs: Optional[int] = None
    prefill_mode: Optional[Literal["chunked", "hybrid"]] = None
    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None
    pipeline_parallel_stages: Optional[int] = None
    opt: Optional[str] = None

    def __repr__(self) -> str:
        out = StringIO()
        print(f"max_num_sequence={self.max_num_sequence}", file=out, end="")
        print(f";max_total_seq_length={self.max_total_seq_length}", file=out, end="")
        print(f";prefill_chunk_size={self.prefill_chunk_size}", file=out, end="")
        print(f";max_history_size={self.max_history_size}", file=out, end="")
        print(f";gpu_memory_utilization={self.gpu_memory_utilization}", file=out, end="")
        print(f";spec_draft_length={self.spec_draft_length}", file=out, end="")
        print(f";spec_tree_width={self.spec_tree_width}", file=out, end="")
        print(f";prefix_cache_mode={self.prefix_cache_mode}", file=out, end="")
        print(
            f";prefix_cache_max_num_recycling_seqs={self.prefix_cache_max_num_recycling_seqs}",
            file=out,
            end="",
        )
        print(f";prefill_mode={self.prefill_mode}", file=out, end="")
        print(f";context_window_size={self.context_window_size}", file=out, end="")
        print(f";sliding_window_size={self.sliding_window_size}", file=out, end="")
        print(f";attention_sink_size={self.attention_sink_size}", file=out, end="")
        print(f";tensor_parallel_shards={self.tensor_parallel_shards}", file=out, end="")
        print(
            f";pipeline_parallel_stages={self.pipeline_parallel_stages}",
            file=out,
            end="",
        )
        print(f";opt={self.opt}", file=out, end="")
        return out.getvalue().rstrip()

    @staticmethod
    def from_str(source: str) -> "EngineConfigOverride":
        """Parse engine config override values from a string."""
        parser = argparse.ArgumentParser(description="Engine config override values")

        parser.add_argument("--max_num_sequence", type=int, default=None)
        parser.add_argument("--max_total_seq_length", type=int, default=None)
        parser.add_argument("--prefill_chunk_size", type=int, default=None)
        parser.add_argument("--max_history_size", type=int, default=None)
        parser.add_argument("--gpu_memory_utilization", type=float, default=None)
        parser.add_argument("--spec_draft_length", type=int, default=None)
        parser.add_argument("--spec_tree_width", type=int, default=None)
        parser.add_argument("--prefix_cache_mode", type=str, default="radix")
        parser.add_argument("--prefix_cache_max_num_recycling_seqs", type=int, default=None)
        parser.add_argument("--prefill_mode", type=str, default="hybrid")
        parser.add_argument("--context_window_size", type=int, default=None)
        parser.add_argument("--sliding_window_size", type=int, default=None)
        parser.add_argument("--attention_sink_size", type=int, default=None)
        parser.add_argument("--tensor_parallel_shards", type=int, default=None)
        parser.add_argument("--pipeline_parallel_stages", type=int, default=None)
        parser.add_argument("--opt", type=str, default=None)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return EngineConfigOverride(
            max_num_sequence=results.max_num_sequence,
            max_total_seq_length=results.max_total_seq_length,
            prefill_chunk_size=results.prefill_chunk_size,
            max_history_size=results.max_history_size,
            gpu_memory_utilization=results.gpu_memory_utilization,
            spec_draft_length=results.spec_draft_length,
            spec_tree_width=results.spec_tree_width,
            prefix_cache_mode=results.prefix_cache_mode,
            prefix_cache_max_num_recycling_seqs=results.prefix_cache_max_num_recycling_seqs,
            prefill_mode=results.prefill_mode,
            context_window_size=results.context_window_size,
            sliding_window_size=results.sliding_window_size,
            attention_sink_size=results.attention_sink_size,
            tensor_parallel_shards=results.tensor_parallel_shards,
            pipeline_parallel_stages=results.pipeline_parallel_stages,
            opt=results.opt,
        )


def main(argv):
    """Parse command line arguments and call `mlc_llm.interface.serve`."""
    parser = ArgumentParser("MLC LLM Serve CLI")

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
        "--mode",
        type=str,
        choices=["local", "interactive", "server"],
        default="local",
        help=HELP["mode_serve"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--enable-debug",
        action="store_true",
        help="whether we enable debug end points and debug config when accepting requests",
    )
    parser.add_argument(
        "--additional-models", type=str, nargs="*", help=HELP["additional_models_serve"]
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Path to the embedding model weight directory (enables /v1/embeddings endpoint)",
    )
    parser.add_argument(
        "--embedding-model-lib",
        type=str,
        default=None,
        help="Path to the compiled embedding model library (.so/.dylib file)",
    )
    parser.add_argument(
        "--speculative-mode",
        type=str,
        choices=["disable", "small_draft", "eagle", "medusa"],
        default="disable",
        help=HELP["speculative_mode_serve"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--prefix-cache-mode",
        type=str,
        choices=["disable", "radix"],
        default="radix",
        help=HELP["prefix_cache_mode_serve"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--prefill-mode",
        type=str,
        choices=["hybrid", "chunked"],
        default="hybrid",
        help=HELP["prefill_mode"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--overrides",
        type=EngineConfigOverride.from_str,
        default="",
        help=HELP["overrides_serve"],
    )
    parser.add_argument("--enable-tracing", action="store_true", help=HELP["enable_tracing_serve"])
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="host name" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port" + ' (default: "%(default)s")',
    )
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument(
        "--allow-origins",
        type=json.loads,
        default=["*"],
        help="allowed origins" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--allow-methods",
        type=json.loads,
        default=["*"],
        help="allowed methods" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--allow-headers",
        type=json.loads,
        default=["*"],
        help="allowed headers" + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)

    additional_models = []
    if parsed.additional_models is not None:
        for additional_model in parsed.additional_models:
            splits = additional_model.split(",", maxsplit=1)
            if len(splits) == 2:
                additional_models.append((splits[0], splits[1]))
            else:
                additional_models.append(splits[0])

    serve(
        model=parsed.model,
        device=parsed.device,
        model_lib=parsed.model_lib,
        mode=parsed.mode,
        enable_debug=parsed.enable_debug,
        additional_models=additional_models,
        embedding_model=parsed.embedding_model,
        embedding_model_lib=parsed.embedding_model_lib,
        tensor_parallel_shards=parsed.overrides.tensor_parallel_shards,
        pipeline_parallel_stages=parsed.overrides.pipeline_parallel_stages,
        opt=parsed.overrides.opt,
        speculative_mode=parsed.speculative_mode,
        prefix_cache_mode=parsed.prefix_cache_mode,
        max_num_sequence=parsed.overrides.max_num_sequence,
        max_total_sequence_length=parsed.overrides.max_total_seq_length,
        max_single_sequence_length=parsed.overrides.context_window_size,
        prefill_chunk_size=parsed.overrides.prefill_chunk_size,
        sliding_window_size=parsed.overrides.sliding_window_size,
        attention_sink_size=parsed.overrides.attention_sink_size,
        max_history_size=parsed.overrides.max_history_size,
        gpu_memory_utilization=parsed.overrides.gpu_memory_utilization,
        spec_draft_length=parsed.overrides.spec_draft_length,
        spec_tree_width=parsed.overrides.spec_tree_width,
        prefix_cache_max_num_recycling_seqs=parsed.overrides.prefix_cache_max_num_recycling_seqs,
        prefill_mode=parsed.prefill_mode,
        enable_tracing=parsed.enable_tracing,
        host=parsed.host,
        port=parsed.port,
        allow_credentials=parsed.allow_credentials,
        allow_origins=parsed.allow_origins,
        allow_methods=parsed.allow_methods,
        allow_headers=parsed.allow_headers,
    )
