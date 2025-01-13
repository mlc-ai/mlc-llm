"""MicroServing server entrypoints in MLC LLM"""

import fastapi

from mlc_llm.protocol.debug_protocol import DisaggConfig
from mlc_llm.protocol.microserving_protocol import (
    PrepRecvRequest,
    PrepRecvResponse,
    RemoteSendRequest,
    StartGenerateRequest,
)
from mlc_llm.protocol.openai_api_protocol import StreamOptions

from .openai_entrypoints import request_completion

app = fastapi.APIRouter()


################ MicroServing Endpoints ################


@app.post("/microserving/prep_recv")
async def prep_recv(request: PrepRecvRequest, raw_request: fastapi.Request) -> PrepRecvResponse:
    """Handle the microserving request for receive preparation.
    Match the prompt in the prefix cache (when enabled),
    allocate entries in the KV cache to prepare receiving the KV data of the prompt.
    Return the matched prefix length and the allocated KV entry metadata.
    """
    request.debug_config.disagg_config = DisaggConfig(
        kind="prepare_receive",
        kv_window_begin=0,  # always zero for prepare_receive
        kv_window_end=request.end,
    )
    request.stream_options = StreamOptions(include_usage=True)
    request.stream = False

    response = await request_completion(request=request, raw_request=raw_request)
    assert response.usage is not None
    assert response.usage.extra is not None
    assert "prefix_matched_length" in response.usage.extra
    assert "kv_append_metadata" in response.usage.extra
    return PrepRecvResponse(
        prefix_matched_length=response.usage.extra["prefix_matched_length"],
        kv_append_metadata=response.usage.extra["kv_append_metadata"],
    )


@app.post("/microserving/remote_send")
async def remote_send(request: RemoteSendRequest, raw_request: fastapi.Request):
    """Compute and generate the KV data of the prompt in the specified KV window.
    Send the KV data to the destination server."""
    request.debug_config.disagg_config = DisaggConfig(
        kind="remote_send",
        kv_window_begin=request.begin,
        kv_window_end=request.end,
        kv_append_metadata=request.kv_addr_info,
        dst_group_offset=request.recv_rank,
    )
    request.stream_options = StreamOptions(include_usage=True)
    request.stream = False

    await request_completion(request=request, raw_request=raw_request)
    return {}


@app.post("/microserving/start_generate")
async def start_generate(request: StartGenerateRequest, raw_request: fastapi.Request):
    """Prefill the prompt in the specified KV window, and start decode."""
    request.debug_config.disagg_config = DisaggConfig(
        kind="start_generation",
        kv_window_begin=request.begin,
    )
    return await request_completion(request=request, raw_request=raw_request)
