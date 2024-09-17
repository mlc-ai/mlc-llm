"""MLC LLM server debug entrypoints"""

import json
from http import HTTPStatus

import fastapi

from mlc_llm.protocol import error_protocol
from mlc_llm.serve.server import ServerContext

app = fastapi.APIRouter()

################ /debug/dump_event_trace ################


@app.post("/debug/dump_event_trace")
async def debug_dump_event_trace(request: fastapi.Request):
    """Return the recorded events in Chrome Trace Event Format in JSON string.
    The input request payload should have only one field, specifying the
    model to query. For example: `{"model": "Llama-2-7b-chat-hf-q0f16"}`.
    """
    # Get the raw request body as bytes
    request_raw_data = await request.body()
    request_json_str = request_raw_data.decode("utf-8")
    try:
        # Parse the JSON string
        request_dict = json.loads(request_json_str)
    except json.JSONDecodeError:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f"Invalid request {request_json_str}"
        )
    if "model" not in request_dict:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f"Invalid request {request_json_str}"
        )

    # Check the requested model.
    model = request_dict["model"]

    server_context: ServerContext = ServerContext.current()
    async_engine = server_context.get_engine(model)

    if async_engine is None:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{model}" is not served.'
        )
    if async_engine.state.trace_recorder is None:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{model}" does not enable tracing'
        )

    return json.loads(async_engine.state.trace_recorder.dump_json())


################ /debug/cuda_profiler_start/end ################


@app.post("/debug/cuda_profiler_start")
async def debug_cuda_profiler_start(_request: fastapi.Request):
    """Start the cuda profiler for the engine. Only for debug purpose."""
    server_context: ServerContext = ServerContext.current()
    # Since the CUDA profiler is process-wise, call the function for one model is sufficient.
    for model in server_context.get_model_list():
        async_engine = server_context.get_engine(model)
        async_engine._debug_call_func_on_all_worker(  # pylint: disable=protected-access
            "mlc.debug_cuda_profiler_start"
        )
        break


@app.post("/debug/cuda_profiler_stop")
async def debug_cuda_profiler_stop(_request: fastapi.Request):
    """Stop the cuda profiler for the engine. Only for debug purpose."""
    server_context: ServerContext = ServerContext.current()
    # Since the CUDA profiler is process-wise, call the function for one model is sufficient.
    for model in server_context.get_model_list():
        async_engine = server_context.get_engine(model)
        async_engine._debug_call_func_on_all_worker(  # pylint: disable=protected-access
            "mlc.debug_cuda_profiler_stop"
        )
        break


@app.post("/debug/dump_engine_metrics")
async def debug_dump_engine_metrics(request: fastapi.Request):
    """Dump the engine metrics for the engine. Only for debug purpose."""
    # Get the raw request body as bytes
    request_raw_data = await request.body()
    request_json_str = request_raw_data.decode("utf-8")
    try:
        # Parse the JSON string
        request_dict = json.loads(request_json_str)
    except json.JSONDecodeError:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f"Invalid request {request_json_str}"
        )

    # Check the requested model.
    model = request_dict.get("model", None)

    server_context: ServerContext = ServerContext.current()
    async_engine = server_context.get_engine(model)
    res = await async_engine.metrics()
    return res


@app.post("/debug/reset_engine")
async def debug_reset_engine_stats(request: fastapi.Request):
    """Reset the engine, clean up all running data and metrics."""
    # Get the raw request body as bytes
    request_raw_data = await request.body()
    request_json_str = request_raw_data.decode("utf-8")
    try:
        # Parse the JSON string
        request_dict = json.loads(request_json_str)
    except json.JSONDecodeError:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f"Invalid request {request_json_str}"
        )
    if "model" not in request_dict:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f"Invalid request {request_json_str}"
        )

    # Check the requested model.
    model = request_dict["model"]

    server_context: ServerContext = ServerContext.current()
    async_engine = server_context.get_engine(model)
    async_engine.reset()
