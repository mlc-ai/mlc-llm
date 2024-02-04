"""OpenAI API-compatible server entrypoints in MLC LLM"""
# pylint: disable=too-many-locals,too-many-return-statements,too-many-statements
from http import HTTPStatus
from typing import AsyncGenerator, List, Optional

import fastapi

from ...protocol import protocol_utils
from ...protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ListResponse,
    ModelResponse,
    UsageInfo,
)
from ..server import ServerContext
from . import entrypoint_utils

app = fastapi.APIRouter()

################ v1/models ################


@app.get("/v1/models")
async def request_models():
    """OpenAI-compatible served model query API.
    API reference: https://platform.openai.com/docs/api-reference/models
    """
    return ListResponse(data=[ModelResponse(id=model) for model in ServerContext.get_model_list()])


################ v1/completions ################


@app.post("/v1/completions")
async def request_completion(request: CompletionRequest, raw_request: fastapi.Request):
    """OpenAI-compatible completion API.
    API reference: https://platform.openai.com/docs/api-reference/completions/create
    """
    # - Check the requested model.
    async_engine = ServerContext.get_engine(request.model)
    if async_engine is None:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{request.model}" is not served.'
        )
    request_id = f"cmpl-{entrypoint_utils.random_uuid()}"
    async_engine.record_event(request_id, event="receive request")

    # - Check if unsupported arguments are specified.
    error = entrypoint_utils.check_unsupported_fields(request)
    if error is not None:
        return error

    # - Process prompt and check validity.
    async_engine.record_event(request_id, event="start tokenization")
    prompts = entrypoint_utils.process_prompts(request.prompt, async_engine.tokenizer.encode)
    async_engine.record_event(request_id, event="finish tokenization")
    if isinstance(prompts, fastapi.responses.JSONResponse):
        # Errored when processing the prompts
        return prompts
    if len(prompts) > 1:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST,
            message="Entrypoint /v1/completions only accept single prompt. "
            f"However, {len(prompts)} prompts {prompts} are received.",
        )
    error = entrypoint_utils.check_prompts_length(prompts, async_engine.max_single_sequence_length)
    if error is not None:
        return error
    prompt = prompts[0]

    # Process generation config. Create request id.
    generation_cfg = protocol_utils.get_generation_config(request)

    # Streaming response.
    if request.stream:

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            assert request.n == 1

            # - Echo back the prompt.
            if request.echo:
                text = async_engine.tokenizer.decode(prompt)
                response = CompletionResponse(
                    id=request_id,
                    choices=[CompletionResponseChoice(text=text)],
                    model=request.model,
                    usage=UsageInfo(
                        prompt_tokens=len(prompt),
                        completion_tokens=0,
                    ),
                )
                yield f"data: {response.model_dump_json()}\n\n"

            # - Generate new tokens.
            num_completion_tokens = 0
            finish_reason = None
            async_engine.record_event(request_id, event="invoke generate")
            async for delta_text, num_delta_tokens, finish_reason in async_engine.generate(
                prompt, generation_cfg, request_id
            ):
                num_completion_tokens += num_delta_tokens
                if delta_text == "":
                    # Ignore empty delta text -- do not yield.
                    continue

                response = CompletionResponse(
                    id=request_id,
                    choices=[
                        CompletionResponseChoice(
                            finish_reason=finish_reason,
                            text=delta_text,
                        )
                    ],
                    model=request.model,
                    usage=UsageInfo(
                        prompt_tokens=len(prompt),
                        completion_tokens=num_completion_tokens,
                    ),
                )
                yield f"data: {response.model_dump_json()}\n\n"
            async_engine.record_event(request_id, event="finish")

            # - Echo the suffix.
            if request.suffix is not None:
                assert finish_reason is not None
                response = CompletionResponse(
                    id=request_id,
                    choices=[
                        CompletionResponseChoice(
                            finish_reason=finish_reason,
                            text=request.suffix,
                        )
                    ],
                    model=request.model,
                    usage=UsageInfo(
                        prompt_tokens=len(prompt),
                        completion_tokens=num_completion_tokens,
                    ),
                )
                yield f"data: {response.model_dump_json()}\n\n"

            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    # Normal response.
    output_text = "" if not request.echo else async_engine.tokenizer.decode(prompt)
    num_completion_tokens = 0
    finish_reason: Optional[str] = None
    async_engine.record_event(request_id, event="invoke generate")
    async for delta_text, num_delta_tokens, finish_reason in async_engine.generate(
        prompt, generation_cfg, request_id
    ):
        if await raw_request.is_disconnected():
            # In non-streaming cases, the engine will not be notified
            # when the request is disconnected.
            # Therefore, we check if it is disconnected each time,
            # and abort the request from engine if so.
            await async_engine.abort(request_id)
            return entrypoint_utils.create_error_response(
                HTTPStatus.BAD_REQUEST, message="The request has disconnected"
            )
        output_text += delta_text
        num_completion_tokens += num_delta_tokens
    assert finish_reason is not None
    suffix = request.suffix if request.suffix is not None else ""
    async_engine.record_event(request_id, event="finish")
    response = CompletionResponse(
        id=request_id,
        choices=[
            CompletionResponseChoice(
                finish_reason=finish_reason,
                text=output_text + suffix,
            )
        ],
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=len(prompt),
            completion_tokens=num_completion_tokens,
        ),
    )
    return response


################ v1/chat/completions ################


def chat_completion_check_message_validity(
    messages: List[ChatCompletionMessage],
) -> Optional[str]:
    """Check if the given chat messages are valid. Return error message if invalid."""
    for i, message in enumerate(messages):
        if message.role == "system" and i != 0:
            return f"System prompt at position {i} in the message list is invalid."
        if message.role == "tool":
            return "Tool as the message author is not supported yet."
        if message.tool_call_id is not None:
            if message.role != "tool":
                return "Non-tool message having `tool_call_id` is invalid."
        if isinstance(message.content, list):
            if message.role != "user":
                return "Non-user message having a list of content is invalid."
            return "User message having a list of content is not supported yet."
        if message.tool_calls is not None:
            if message.role != "assistant":
                return "Non-assistant message having `tool_calls` is invalid."
            return "Assistant message having `tool_calls` is not supported yet."
    return None


@app.post("/v1/chat/completions")
async def request_chat_completion(request: ChatCompletionRequest, raw_request: fastapi.Request):
    """OpenAI-compatible chat completion API.
    API reference: https://platform.openai.com/docs/api-reference/chat
    """
    # - Check the requested model.
    async_engine = ServerContext.get_engine(request.model)
    if async_engine is None:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST, message=f'The requested model "{request.model}" is not served.'
        )
    request_id = f"chatcmpl-{entrypoint_utils.random_uuid()}"
    async_engine.record_event(request_id, event="receive request")
    # - Check if the model supports chat conversation.
    conv_template = ServerContext.get_conv_template(request.model)
    if conv_template is None:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST,
            message=f'The requested model "{request.model}" does not support chat.',
        )

    # - Check if unsupported arguments are specified.
    error = entrypoint_utils.check_unsupported_fields(request)
    if error is not None:
        return error

    # - Process messages and update the conversation template in three steps:
    #   i. Check the message validity.
    #  ii. Add the input messages to the conversation template.
    # iii. Add the additional message for the assistant.
    error_msg = chat_completion_check_message_validity(request.messages)
    if error_msg is not None:
        return entrypoint_utils.create_error_response(HTTPStatus.BAD_REQUEST, message=error_msg)
    for message in request.messages:
        role = message.role
        content = message.content
        assert isinstance(content, str), "Internal error: content is not a string."
        if role == "system":
            conv_template.system_message = content if content is not None else ""
            continue

        assert role != "tool", "Internal error: tool role."
        conv_template.messages.append((conv_template.roles[role == "assistant"], content))
    conv_template.messages.append((conv_template.roles[1], None))

    # - Get the prompt from template, and encode to token ids.
    # - Check prompt length
    async_engine.record_event(request_id, event="start tokenization")
    prompts = entrypoint_utils.process_prompts(
        conv_template.as_prompt(), async_engine.tokenizer.encode
    )
    async_engine.record_event(request_id, event="finish tokenization")
    assert isinstance(prompts, list) and len(prompts) == 1, "Internal error"
    if conv_template.system_prefix_token_ids is not None:
        prompts[0] = conv_template.system_prefix_token_ids + prompts[0]
    error = entrypoint_utils.check_prompts_length(prompts, async_engine.max_single_sequence_length)
    if error is not None:
        return error
    prompt = prompts[0]

    # Process generation config. Create request id.
    generation_cfg = protocol_utils.get_generation_config(
        request,
        extra_stop_token_ids=conv_template.stop_token_ids,
        extra_stop_str=conv_template.stop_str,
    )

    # Streaming response.
    if request.stream:

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            assert request.n == 1
            async_engine.record_event(request_id, event="invoke generate")
            async for delta_text, _, finish_reason in async_engine.generate(
                prompt, generation_cfg, request_id
            ):
                if delta_text == "":
                    async_engine.record_event(request_id, event="skip empty delta text")
                    # Ignore empty delta text -- do not yield.
                    continue

                response = ChatCompletionStreamResponse(
                    id=request_id,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            finish_reason=finish_reason,
                            delta=ChatCompletionMessage(content=delta_text, role="assistant"),
                        )
                    ],
                    model=request.model,
                    system_fingerprint="",
                )
                async_engine.record_event(request_id, event=f"yield delta text {delta_text}")
                yield f"data: {response.model_dump_json()}\n\n"
            async_engine.record_event(request_id, event="finish")
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    # Normal response.
    output_text = ""
    num_completion_tokens = 0
    finish_reason: Optional[str] = None
    async_engine.record_event(request_id, event="invoke generate")
    async for delta_text, num_delta_tokens, finish_reason in async_engine.generate(
        prompt, generation_cfg, request_id
    ):
        if await raw_request.is_disconnected():
            # In non-streaming cases, the engine will not be notified
            # when the request is disconnected.
            # Therefore, we check if it is disconnected each time,
            # and abort the request from engine if so.
            await async_engine.abort(request_id)
            return entrypoint_utils.create_error_response(
                HTTPStatus.BAD_REQUEST, message="The request has disconnected"
            )
        output_text += delta_text
        num_completion_tokens += num_delta_tokens
    assert finish_reason is not None

    async_engine.record_event(request_id, event="finish")
    return ChatCompletionResponse(
        id=request_id,
        choices=[
            ChatCompletionResponseChoice(
                finish_reason=finish_reason,
                message=ChatCompletionMessage(role="assistant", content=output_text),
            )
        ],
        model=request.model,
        system_fingerprint="",
        usage=UsageInfo(prompt_tokens=len(prompt), completion_tokens=num_completion_tokens),
    )
