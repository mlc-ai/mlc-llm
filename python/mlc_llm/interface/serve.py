"""Python entrypoint of serve."""

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

logger = logging.getLogger(__name__)


def serve(
    model: str,
    device: str,
    model_lib: Optional[str],
    mode: Literal["local", "interactive", "server"],
    enable_debug: bool,
    additional_models: List[Union[str, Tuple[str, str]]],
    embedding_model: Optional[str],
    embedding_model_lib: Optional[str],
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
):  # pylint: disable=too-many-arguments, too-many-locals
    """Serve the model with the specified configuration."""
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

    # Set up embedding model if specified
    emb_engine = None
    if embedding_model is not None:
        if embedding_model_lib is None:
            raise ValueError(
                "--embedding-model-lib is required when --embedding-model is specified."
            )
        emb_engine = AsyncEmbeddingEngine(
            model=embedding_model,
            model_lib=embedding_model_lib,
            device=device,
        )
        logger.info("Embedding model %s loaded successfully.", embedding_model)

    with ServerContext() as server_context:
        server_context.add_model(model, async_engine)
        if emb_engine is not None:
            server_context.add_embedding_engine(embedding_model, emb_engine)

        app = fastapi.FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=allow_credentials,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

        app.include_router(openai_entrypoints.app)
        app.include_router(metrics_entrypoints.app)
        app.include_router(microserving_entrypoints.app)

        server_context.enable_debug = enable_debug

        if enable_debug:
            app.include_router(debug_entrypoints.app)
            logger.info("Enable debug endpoint and debug_config in requests...")

        app.exception_handler(error_protocol.BadRequestError)(
            error_protocol.bad_request_error_handler
        )
        uvicorn.run(app, host=host, port=port, log_level="info")
