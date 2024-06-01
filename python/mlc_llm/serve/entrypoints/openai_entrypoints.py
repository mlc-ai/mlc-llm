"""OpenAI API-compatible server entrypoints in MLC LLM"""

# pylint: disable=too-many-locals,too-many-return-statements,too-many-statements
import json
from datetime import datetime
from http import HTTPStatus
from typing import AsyncGenerator, List, Optional

import fastapi
import structlog

from mlc_llm.protocol import error_protocol
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ListResponse,
    LogProbsContent,
    ModelResponse,
)
from mlc_llm.serve import engine_base, engine_utils
from mlc_llm.serve.server import ServerContext

app = fastapi.APIRouter()
logger = structlog.stdlib.get_logger(__name__)
################ v1/models ################


@app.get("/v1/models")
async def request_models() -> ListResponse:
    """OpenAI-compatible served model query API.
    API reference: https://platform.openai.com/docs/api-reference/models
    """
    server_context: ServerContext = ServerContext.current()
    return ListResponse(data=[ModelResponse(id=model) for model in server_context.get_model_list()])


################ v1/completions ################


@app.post("/v1/completions")
async def request_completion(request: CompletionRequest, raw_request: fastapi.Request):
    """OpenAI-compatible completion API.
    API reference: https://platform.openai.com/docs/api-reference/completions/create
    """
    # - Check the requested model.
    server_context: ServerContext = ServerContext.current()
    request_final_usage_include_extra = server_context.enable_debug
    request_include_debug_config = server_context.enable_debug

    if not request_include_debug_config:
        request.debug_config = None

    async_engine = server_context.get_engine(request.model)
    if async_engine is None:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{request.model}" is not served.'
        )
    request_id = f"cmpl-{engine_utils.random_uuid()}"

    # Streaming response.
    if request.stream:
        # We manually get the first response from generator to
        # capture potential exceptions in this scope, rather then
        # the StreamingResponse scope.
        stream_generator = async_engine._handle_completion(  # pylint: disable=protected-access
            request, request_id, request_final_usage_include_extra=request_final_usage_include_extra
        )
        first_response = await anext(  # type: ignore  # pylint: disable=undefined-variable
            stream_generator
        )

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            if isinstance(first_response, StopAsyncIteration):
                yield "data: [DONE]\n\n"
                return
            yield f"data: {first_response.model_dump_json()}\n\n"
            async for response in stream_generator:
                yield f"data: {response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    # Normal response.
    request_final_usage = None
    output_texts = ["" for _ in range(request.n)]
    finish_reasons: List[Optional[str]] = [None for _ in range(request.n)]
    logprob_results: Optional[List[List[LogProbsContent]]] = (
        [[] for _ in range(request.n)] if request.logprobs else None
    )

    async for response in async_engine._handle_completion(  # pylint: disable=protected-access
        request, request_id, request_final_usage_include_extra=request_final_usage_include_extra
    ):
        if await raw_request.is_disconnected():
            # In non-streaming cases, the engine will not be notified
            # when the request is disconnected.
            # Therefore, we check if it is disconnected each time,
            # and abort the request from engine if so.
            await async_engine.abort(request_id)
            return error_protocol.create_error_response(
                HTTPStatus.BAD_REQUEST, message="The request has disconnected"
            )
        # this is the final chunk
        if response.usage is not None:
            request_final_usage = response.usage
            # remove extra information if debug is not enabled
            if not server_context.enable_debug:
                request_final_usage.extra = None
            continue
        for choice in response.choices:
            output_texts[choice.index] += choice.text
            if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                finish_reasons[choice.index] = choice.finish_reason
            if choice.logprobs is not None:
                assert logprob_results is not None
                logprob_results[choice.index] += choice.logprobs.content

    assert all(finish_reason is not None for finish_reason in finish_reasons)
    return engine_base.wrap_completion_response(
        request_id=request_id,
        model=request.model,
        output_texts=output_texts,
        finish_reasons=finish_reasons,
        logprob_results=logprob_results,
        usage=request_final_usage,
    )


################ v1/chat/completions ################


@app.post("/v1/chat/completions")
async def request_chat_completion(
    request: ChatCompletionRequest, raw_request: fastapi.Request
):  # pylint: disable=too-many-branches
    """OpenAI-compatible chat completion API.
    API reference: https://platform.openai.com/docs/api-reference/chat
    """
    # - Check the requested model.
    server_context: ServerContext = ServerContext.current()
    request_final_usage_include_extra = server_context.enable_debug
    request_include_debug_config = server_context.enable_debug

    if server_context.enable_debug:
        request_param = await raw_request.json()
        timestamp = {"timestamp": datetime.now().isoformat()}
        request_param = {**timestamp, **request_param}
        try:
            logger.info("Received chat completion request", request=json.dumps(request_param))
        except (  # pylint: disable=broad-exception-caught
            Exception,
            json.JSONDecodeError,
        ) as err:
            logger.error("Error in dumping request parameters: %s", err)

    if not request_include_debug_config:
        request.debug_config = None

    async_engine = server_context.get_engine(request.model)
    if async_engine is None:
        return error_protocol.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{request.model}" is not served.'
        )
    request_id = f"chatcmpl-{engine_utils.random_uuid()}"

    # Streaming response.
    if request.stream:
        # We manually get the first response from generator to
        # capture potential exceptions in this scope, rather then
        # the StreamingResponse scope.
        stream_generator = async_engine._handle_chat_completion(  # pylint: disable=protected-access
            request, request_id, request_final_usage_include_extra=request_final_usage_include_extra
        )
        first_response = await anext(  # type: ignore  # pylint: disable=undefined-variable
            stream_generator
        )

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            if isinstance(first_response, StopAsyncIteration):
                yield "data: [DONE]\n\n"
                return
            yield f"data: {first_response.model_dump_json()}\n\n"
            async for response in stream_generator:
                yield f"data: {response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    # Normal response.
    request_final_usage = None
    output_texts = ["" for _ in range(request.n)]
    finish_reasons: List[Optional[str]] = [None for _ in range(request.n)]
    logprob_results: Optional[List[List[LogProbsContent]]] = (
        [[] for _ in range(request.n)] if request.logprobs else None
    )

    async for response in async_engine._handle_chat_completion(  # pylint: disable=protected-access
        request, request_id, request_final_usage_include_extra=request_final_usage_include_extra
    ):
        if await raw_request.is_disconnected():
            # In non-streaming cases, the engine will not be notified
            # when the request is disconnected.
            # Therefore, we check if it is disconnected each time,
            # and abort the request from engine if so.
            await async_engine.abort(request_id)
            return error_protocol.create_error_response(
                HTTPStatus.BAD_REQUEST, message="The request has disconnected"
            )
        # usage is always the last chunk
        if response.usage is not None:
            request_final_usage = response.usage
            # remove extra information if debug is not enabled
            if not server_context.enable_debug:
                request_final_usage.extra = None

        for choice in response.choices:
            assert isinstance(choice.delta.content, str)
            output_texts[choice.index] += choice.delta.content
            if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                finish_reasons[choice.index] = choice.finish_reason
            if choice.logprobs is not None:
                assert logprob_results is not None
                logprob_results[choice.index] += choice.logprobs.content

    assert all(finish_reason is not None for finish_reason in finish_reasons)
    use_function_calling, tool_calls_list = engine_base.process_function_call_output(
        output_texts, finish_reasons
    )

    return engine_base.wrap_chat_completion_response(
        request_id=request_id,
        model=request.model,
        output_texts=output_texts,
        finish_reasons=finish_reasons,
        tool_calls_list=tool_calls_list,
        logprob_results=logprob_results,
        use_function_calling=use_function_calling,
        usage=request_final_usage,
    )
