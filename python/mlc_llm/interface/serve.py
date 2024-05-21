"""Python entrypoint of serve."""

from typing import Any, List, Literal, Optional

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from mlc_llm.protocol import error_protocol
from mlc_llm.serve import engine
from mlc_llm.serve.entrypoints import debug_entrypoints, openai_entrypoints
from mlc_llm.serve.server import ServerContext


def serve(
    model: str,
    device: str,
    model_lib: Optional[str],
    mode: Literal["local", "interactive", "server"],
    additional_models: List[str],
    max_batch_size: Optional[int],
    max_total_sequence_length: Optional[int],
    prefill_chunk_size: Optional[int],
    max_history_size: Optional[int],
    prefix_cache_max_num_seqs: Optional[int],
    gpu_memory_utilization: Optional[float],
    speculative_mode: Literal["disable", "small_draft", "eagle", "medusa"],
    spec_draft_length: int,
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
        additional_models=additional_models,
        max_batch_size=max_batch_size,
        max_total_sequence_length=max_total_sequence_length,
        prefill_chunk_size=prefill_chunk_size,
        max_history_size=max_history_size,
        prefix_cache_max_num_seqs=prefix_cache_max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        speculative_mode=speculative_mode,
        spec_draft_length=spec_draft_length,
        enable_tracing=enable_tracing,
    )

    with ServerContext() as server_context:
        server_context.add_model(model, async_engine)

        app = fastapi.FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=allow_credentials,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

        app.include_router(openai_entrypoints.app)
        app.include_router(debug_entrypoints.app)
        app.exception_handler(error_protocol.BadRequestError)(
            error_protocol.bad_request_error_handler
        )
        uvicorn.run(app, host=host, port=port, log_level="info")
