"""MLC LLM server debug entrypoints"""
import json
from http import HTTPStatus

import fastapi

from ..server import ServerContext
from . import entrypoint_utils

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
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f"Invalid request {request_json_str}"
        )
    if "model" not in request_dict:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f"Invalid request {request_json_str}"
        )

    # - Check the requested model.
    model = request_dict["model"]
    async_engine = ServerContext.get_engine(model)
    if async_engine is None:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{model}" is not served.'
        )
    if async_engine.trace_recorder is None:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{model}" does not enable tracing'
        )

    return json.loads(async_engine.trace_recorder.dump_json())
