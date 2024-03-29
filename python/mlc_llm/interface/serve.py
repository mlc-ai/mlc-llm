"""Python entrypoint of serve."""

from typing import Any, Optional

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from mlc_llm.serve import async_engine, config
from mlc_llm.serve.entrypoints import debug_entrypoints, openai_entrypoints
from mlc_llm.serve.server import ServerContext


def serve(
    model: str,
    device: str,
    model_lib_path: Optional[str],
    max_batch_size: int,
    max_total_sequence_length: Optional[int],
    prefill_chunk_size: Optional[int],
    enable_tracing: bool,
    host: str,
    port: int,
    allow_credentials: bool,
    allow_origins: Any,
    allow_methods: Any,
    allow_headers: Any,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Serve the model with the specified configuration."""
    # Initialize model loading info and KV cache config
    model_info = async_engine.ModelInfo(
        model=model,
        model_lib_path=model_lib_path,
        device=device,
    )
    kv_cache_config = config.KVCacheConfig(
        max_num_sequence=max_batch_size,
        max_total_sequence_length=max_total_sequence_length,
        prefill_chunk_size=prefill_chunk_size,
    )
    # Create engine and start the background loop
    engine = async_engine.AsyncThreadedEngine(
        model_info, kv_cache_config, enable_tracing=enable_tracing
    )

    with ServerContext() as server_context:
        server_context.add_model(model, engine)

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
        uvicorn.run(app, host=host, port=port, log_level="info")
