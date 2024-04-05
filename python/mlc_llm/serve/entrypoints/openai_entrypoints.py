"""OpenAI API-compatible server entrypoints in MLC LLM"""

# pylint: disable=too-many-locals,too-many-return-statements,too-many-statements
import ast
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Union

import fastapi

from mlc_llm.protocol import error_protocol
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatFunctionCall,
    ChatToolCall,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ListResponse,
    LogProbs,
    LogProbsContent,
    ModelResponse,
    UsageInfo,
)
from mlc_llm.serve import engine_utils
from mlc_llm.serve.server import ServerContext

app = fastapi.APIRouter()

################ v1/models ################


@app.get("/v1/models")
async def request_models():
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
            request, request_id
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
    num_prompt_tokens = 0
    num_completion_tokens = 0
    output_texts = ["" for _ in range(request.n)]
    finish_reasons: List[Optional[str]] = [None for _ in range(request.n)]
    logprob_results: Optional[List[List[LogProbsContent]]] = (
        [[] for _ in range(request.n)] if request.logprobs else None
    )

    async for response in async_engine._handle_completion(  # pylint: disable=protected-access
        request, request_id
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
        num_prompt_tokens = response.usage.prompt_tokens
        num_completion_tokens = response.usage.completion_tokens
        for choice in response.choices:
            output_texts[choice.index] += choice.text
            if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                finish_reasons[choice.index] = choice.finish_reason
            if choice.logprobs is not None:
                assert logprob_results is not None
                logprob_results[choice.index] += choice.logprobs.content

    assert all(finish_reason is not None for finish_reason in finish_reasons)
    return CompletionResponse(
        id=request_id,
        choices=[
            CompletionResponseChoice(
                index=i,
                finish_reason=finish_reason,
                text=output_text,
                logprobs=(
                    LogProbs(content=logprob_results[i]) if logprob_results is not None else None
                ),
            )
            for i, (output_text, finish_reason) in enumerate(zip(output_texts, finish_reasons))
        ],
        model=request.model,
        usage=UsageInfo(prompt_tokens=num_prompt_tokens, completion_tokens=num_completion_tokens),
    )


################ v1/chat/completions ################


def convert_function_str_to_json(stringified_calls: str) -> List[Union[Dict, None]]:
    """Convert a (possibly list) of function call string to a list of json objects.
    Return None for invalid function call string."""

    def parse_function_call(call_str: str):
        node = ast.parse(call_str, mode="eval")
        call_node = node.body
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
            name = call_node.func.id
            arguments = {}
            for keyword in call_node.keywords:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            return {"name": name, "arguments": arguments}
        return None

    if (
        stringified_calls[0] == "[" and stringified_calls[-1] == "]"
    ):  # hacky way to check if string list
        calls = ast.literal_eval(stringified_calls)
    else:
        calls = [stringified_calls]
    function_calls_json = [parse_function_call(call_str) for call_str in calls]
    return function_calls_json


@app.post("/v1/chat/completions")
async def request_chat_completion(
    request: ChatCompletionRequest, raw_request: fastapi.Request
):  # pylint: disable=too-many-branches
    """OpenAI-compatible chat completion API.
    API reference: https://platform.openai.com/docs/api-reference/chat
    """
    # - Check the requested model.
    server_context: ServerContext = ServerContext.current()
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
            request, request_id
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
    num_prompt_tokens = 0
    num_completion_tokens = 0
    output_texts = ["" for _ in range(request.n)]
    finish_reasons: List[Optional[str]] = [None for _ in range(request.n)]
    logprob_results: Optional[List[List[LogProbsContent]]] = (
        [[] for _ in range(request.n)] if request.logprobs else None
    )

    async for response in async_engine._handle_chat_completion(  # pylint: disable=protected-access
        request, request_id
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
        num_prompt_tokens = response.usage.prompt_tokens
        num_completion_tokens = response.usage.completion_tokens
        for choice in response.choices:
            assert isinstance(choice.delta.content, str)
            output_texts[choice.index] += choice.delta.content
            if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                finish_reasons[choice.index] = choice.finish_reason
            if choice.logprobs is not None:
                assert logprob_results is not None
                logprob_results[choice.index] += choice.logprobs.content

    assert all(finish_reason is not None for finish_reason in finish_reasons)

    tool_calls_list: List[List[ChatToolCall]] = [[] for _ in range(request.n)]
    use_function_calling = any(finish_reason == "tool_calls" for finish_reason in finish_reasons)
    if use_function_calling:
        for i, output_text in enumerate(output_texts):
            try:
                fn_json_list = convert_function_str_to_json(output_text)
            except (SyntaxError, ValueError):
                output_text = "Got an invalid function call output from model"
                finish_reasons[i] = "error"
            else:
                tool_calls_list[i] = [
                    ChatToolCall(
                        type="function",
                        function=ChatFunctionCall(
                            name=fn_json_obj["name"], arguments=fn_json_obj["arguments"]
                        ),
                    )
                    for fn_json_obj in fn_json_list
                    if fn_json_obj is not None
                ]
                if len(tool_calls_list[i]) == 0:
                    output_texts[i] = "Got an invalid function call output from model"
                    finish_reasons[i] = "error"
                else:
                    finish_reasons[i] = "tool_calls"

    return ChatCompletionResponse(
        id=request_id,
        choices=[
            ChatCompletionResponseChoice(
                index=i,
                finish_reason=finish_reasons[i],
                message=(
                    ChatCompletionMessage(role="assistant", content=output_text)
                    if not use_function_calling or finish_reason == "error"
                    else ChatCompletionMessage(role="assistant", tool_calls=tool_calls)
                ),
                logprobs=(
                    LogProbs(content=logprob_results[i]) if logprob_results is not None else None
                ),
            )
            for i, (output_text, finish_reason, tool_calls) in enumerate(
                zip(output_texts, finish_reasons, tool_calls_list)
            )
        ],
        model=request.model,
        system_fingerprint="",
        usage=UsageInfo(prompt_tokens=num_prompt_tokens, completion_tokens=num_completion_tokens),
    )
