"""Python entrypoint of serve."""

import json
from typing import Any, List, Literal, Optional, Tuple, Union

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from mlc_llm.protocol import error_protocol
from mlc_llm.serve import engine
from mlc_llm.serve.embedding_engine import AsyncEmbeddingEngine
from mlc_llm.serve.entrypoints import (
    debug_entrypoints,
    metrics_entrypoints,
    microserving_entrypoints,
    openai_entrypoints,
)
from mlc_llm.serve.server import ServerContext
from mlc_llm.support import logging
from mlc_llm.support.auto_config import detect_mlc_chat_config

logger = logging.getLogger(__name__)


def _detect_model_task(model: str) -> str:
    """Detect the model_task field from the primary model's mlc-chat-config.json.

    Returns "chat" or "embedding". Defaults to "chat" if the field is absent.
    """
    config_path = detect_mlc_chat_config(model)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("model_task", "chat")


def serve(
    model: str,
    device: str,
    model_lib: Optional[str],
    mode: Literal["local", "interactive", "server"],
    enable_debug: bool,
    additional_models: List[Union[str, Tuple[str, str]]],
    tensor_parallel_shards: Optional[int],
    pipeline_parallel_stages: Optional[int],
    opt: Optional[str],
    max_num_sequence: Optional[int],
    max_total_sequence_length: Optional[int],
    max_single_sequence_length: Optional[int],
    prefill_chunk_size: Optional[int],
    sliding_window_size: Optional[int],
    attention_sink_size: Optional[int],
    max_history_size: Optional[int],
    gpu_memory_utilization: Optional[float],
    speculative_mode: Literal["disable", "small_draft", "eagle", "medusa"],
    spec_draft_length: Optional[int],
    spec_tree_width: Optional[int],
    prefix_cache_mode: Literal["disable", "radix"],
    prefix_cache_max_num_recycling_seqs: Optional[int],
    prefill_mode: Literal["hybrid", "chunked"],
    enable_tracing: bool,
    host: str,
    port: int,
    allow_credentials: bool,
    allow_origins: Any,
    allow_methods: Any,
    allow_headers: Any,
    api_key: Optional[str] = None,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Serve the model with the specified configuration."""
    # Detect primary model task to decide the serve path
    model_task = _detect_model_task(model)
    logger.info("Primary model task: %s", model_task)

    if model_task == "embedding":
        _serve_embedding(
            model=model,
            device=device,
            model_lib=model_lib,
            host=host,
            port=port,
            allow_credentials=allow_credentials,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            api_key=api_key,
        )
    else:
        _serve_chat(
            model=model,
            device=device,
            model_lib=model_lib,
            mode=mode,
            enable_debug=enable_debug,
            additional_models=additional_models,
            tensor_parallel_shards=tensor_parallel_shards,
            pipeline_parallel_stages=pipeline_parallel_stages,
            opt=opt,
            max_num_sequence=max_num_sequence,
            max_total_sequence_length=max_total_sequence_length,
            max_single_sequence_length=max_single_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
            sliding_window_size=sliding_window_size,
            attention_sink_size=attention_sink_size,
            max_history_size=max_history_size,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_mode=speculative_mode,
            spec_draft_length=spec_draft_length,
            spec_tree_width=spec_tree_width,
            prefix_cache_mode=prefix_cache_mode,
            prefix_cache_max_num_recycling_seqs=prefix_cache_max_num_recycling_seqs,
            prefill_mode=prefill_mode,
            enable_tracing=enable_tracing,
            host=host,
            port=port,
            allow_credentials=allow_credentials,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            api_key=api_key,
        )


def _serve_embedding(
    model: str,
    device: str,
    model_lib: Optional[str],
    host: str,
    port: int,
    allow_credentials: bool,
    allow_origins: Any,
    allow_methods: Any,
    allow_headers: Any,
    api_key: Optional[str],
):  # pylint: disable=too-many-arguments
    """Embedding-only serve path: only AsyncEmbeddingEngine, only /v1/models and /v1/embeddings."""
    if model_lib is None:
        raise ValueError("--model-lib is required when serving an embedding model.")

    emb_engine = AsyncEmbeddingEngine(
        model=model,
        model_lib=model_lib,
        device=device,
    )
    logger.info("Embedding model %s loaded successfully.", model)

    with ServerContext() as server_context:
        server_context.add_embedding_engine(model, emb_engine)
        server_context.api_key = api_key

        app = fastapi.FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=allow_credentials,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

        app.include_router(openai_entrypoints.embedding_app)

        app.exception_handler(error_protocol.BadRequestError)(
            error_protocol.bad_request_error_handler
        )

        logger.info("Embedding server started.")
        logger.info("  model_task  : embedding")
        logger.info("  base URL    : http://%s:%d", host, port)
        logger.info("  GET  /v1/models")
        logger.info("  POST /v1/embeddings")

        uvicorn.run(app, host=host, port=port, log_level="info")


def _serve_chat(
    model: str,
    device: str,
    model_lib: Optional[str],
    mode: Literal["local", "interactive", "server"],
    enable_debug: bool,
    additional_models: List[Union[str, Tuple[str, str]]],
    tensor_parallel_shards: Optional[int],
    pipeline_parallel_stages: Optional[int],
    opt: Optional[str],
    max_num_sequence: Optional[int],
    max_total_sequence_length: Optional[int],
    max_single_sequence_length: Optional[int],
    prefill_chunk_size: Optional[int],
    sliding_window_size: Optional[int],
    attention_sink_size: Optional[int],
    max_history_size: Optional[int],
    gpu_memory_utilization: Optional[float],
    speculative_mode: Literal["disable", "small_draft", "eagle", "medusa"],
    spec_draft_length: Optional[int],
    spec_tree_width: Optional[int],
    prefix_cache_mode: Literal["disable", "radix"],
    prefix_cache_max_num_recycling_seqs: Optional[int],
    prefill_mode: Literal["hybrid", "chunked"],
    enable_tracing: bool,
    host: str,
    port: int,
    allow_credentials: bool,
    allow_origins: Any,
    allow_methods: Any,
    allow_headers: Any,
    api_key: Optional[str],
):  # pylint: disable=too-many-arguments, too-many-locals
    """Chat serve path: existing AsyncMLCEngine behavior, unchanged."""
    # Create engine and start the background loop
    async_engine = engine.AsyncMLCEngine(
        model=model,
        device=device,
        model_lib=model_lib,
        mode=mode,
        engine_config=engine.EngineConfig(
            additional_models=additional_models,
            tensor_parallel_shards=tensor_parallel_shards,
            pipeline_parallel_stages=pipeline_parallel_stages,
            opt=opt,
            max_num_sequence=max_num_sequence,
            max_total_sequence_length=max_total_sequence_length,
            max_single_sequence_length=max_single_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
            sliding_window_size=sliding_window_size,
            attention_sink_size=attention_sink_size,
            max_history_size=max_history_size,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_mode=speculative_mode,
            spec_draft_length=spec_draft_length,
            spec_tree_width=spec_tree_width,
            prefix_cache_mode=prefix_cache_mode,
            prefix_cache_max_num_recycling_seqs=prefix_cache_max_num_recycling_seqs,
            prefill_mode=prefill_mode,
        ),
        enable_tracing=enable_tracing,
    )

    with ServerContext() as server_context:
        server_context.add_model(model, async_engine)
        server_context.api_key = api_key

        app = fastapi.FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=allow_credentials,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

        app.include_router(openai_entrypoints.chat_app)
        app.include_router(metrics_entrypoints.app)
        app.include_router(microserving_entrypoints.app)

        server_context.enable_debug = enable_debug

        if enable_debug:
            app.include_router(debug_entrypoints.app)
            logger.info("Enable debug endpoint and debug_config in requests...")

        app.exception_handler(error_protocol.BadRequestError)(
            error_protocol.bad_request_error_handler
        )

        logger.info("Chat server started.")
        logger.info("  model_task  : chat")
        logger.info("  base URL    : http://%s:%d", host, port)
        logger.info("  GET  /v1/models")
        logger.info("  POST /v1/completions")
        logger.info("  POST /v1/chat/completions")
        if enable_debug:
            logger.info("  POST /debug/dump_event_trace  (debug enabled)")

        uvicorn.run(app, host=host, port=port, log_level="info")
