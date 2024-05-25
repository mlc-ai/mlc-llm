"""MLC LLM server metrics entrypoints"""

import fastapi
from fastapi.responses import PlainTextResponse

from mlc_llm.serve.server import ServerContext

app = fastapi.APIRouter()

################ /metrics ################


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics(_request: fastapi.Request):
    """Start the cuda profiler for the engine. Only for debug purpose."""
    server_context: ServerContext = ServerContext.current()
    # Use the metrics from first engine for now
    # TODO(mlc-team): consider refactor server context to
    # single engine since multiple AsyncMLCEngine do not work well with each other
    # We need to work within the internal engine instead.
    for model in server_context.get_model_list():
        async_engine = server_context.get_engine(model)
        return (await async_engine.metrics()).prometheus_text()
