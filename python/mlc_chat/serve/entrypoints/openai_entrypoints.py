"""OpenAI API-compatible server entrypoints in MLC LLM"""

# pylint: disable=too-many-locals,too-many-return-statements,too-many-statements
import ast
import json
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Union

import fastapi

from ...protocol import protocol_utils
from ...protocol.conversation_protocol import Conversation
from ...protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
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
            # - Echo back the prompt.
            if request.echo:
                text = async_engine.tokenizer.decode(prompt)
                response = CompletionResponse(
                    id=request_id,
                    choices=[
                        CompletionResponseChoice(index=i, text=text)
                        for i in range(generation_cfg.n)
                    ],
                    model=request.model,
                    usage=UsageInfo(
                        prompt_tokens=len(prompt),
                        completion_tokens=0,
                    ),
                )
                yield f"data: {response.model_dump_json()}\n\n"

            # - Generate new tokens.
            num_completion_tokens = 0
            finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
            async_engine.record_event(request_id, event="invoke generate")
            async for delta_outputs in async_engine.generate(prompt, generation_cfg, request_id):
                assert len(delta_outputs) == generation_cfg.n
                choices = []
                for i, delta_output in enumerate(delta_outputs):
                    finish_reason_updated = False
                    if delta_output.finish_reason is not None and finish_reasons[i] is None:
                        finish_reasons[i] = delta_output.finish_reason
                        finish_reason_updated = True
                    num_completion_tokens += delta_output.num_delta_tokens
                    if not finish_reason_updated and delta_output.delta_text == "":
                        # Ignore empty delta text when finish reason is not updated.
                        continue

                    choices.append(
                        CompletionResponseChoice(
                            index=i,
                            finish_reason=finish_reasons[i],
                            text=delta_output.delta_text,
                            logprobs=(
                                LogProbs(
                                    content=[
                                        LogProbsContent.model_validate_json(logprob_json_str)
                                        for logprob_json_str in delta_output.delta_logprob_json_strs
                                    ]
                                )
                                if delta_output.delta_logprob_json_strs is not None
                                else None
                            ),
                        )
                    )

                if len(choices) == 0:
                    # Skip yield when there is no delta output.
                    continue
                response = CompletionResponse(
                    id=request_id,
                    choices=choices,
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
                assert all(finish_reason is not None for finish_reason in finish_reasons)
                response = CompletionResponse(
                    id=request_id,
                    choices=[
                        CompletionResponseChoice(
                            index=i,
                            finish_reason=finish_reason,
                            text=request.suffix,
                        )
                        for i, finish_reason in enumerate(finish_reasons)
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
    init_output_text = "" if not request.echo else async_engine.tokenizer.decode(prompt)
    output_texts = [init_output_text for _ in range(generation_cfg.n)]
    num_completion_tokens = 0
    finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
    logprob_json_strs_list: Optional[List[List[str]]] = (
        [[] for _ in range(generation_cfg.n)] if generation_cfg.logprobs else None
    )
    async_engine.record_event(request_id, event="invoke generate")
    async for delta_outputs in async_engine.generate(prompt, generation_cfg, request_id):
        if await raw_request.is_disconnected():
            # In non-streaming cases, the engine will not be notified
            # when the request is disconnected.
            # Therefore, we check if it is disconnected each time,
            # and abort the request from engine if so.
            await async_engine.abort(request_id)
            return entrypoint_utils.create_error_response(
                HTTPStatus.BAD_REQUEST, message="The request has disconnected"
            )

        assert len(delta_outputs) == generation_cfg.n
        for i, delta_output in enumerate(delta_outputs):
            if delta_output.finish_reason is not None and finish_reasons[i] is None:
                finish_reasons[i] = delta_output.finish_reason
            output_texts[i] += delta_output.delta_text
            num_completion_tokens += delta_output.num_delta_tokens
            if logprob_json_strs_list is not None:
                assert delta_output.delta_logprob_json_strs is not None
                logprob_json_strs_list[i] += delta_output.delta_logprob_json_strs
    assert all(finish_reason is not None for finish_reason in finish_reasons)
    suffix = request.suffix if request.suffix is not None else ""
    async_engine.record_event(request_id, event="finish")
    response = CompletionResponse(
        id=request_id,
        choices=[
            CompletionResponseChoice(
                index=i,
                finish_reason=finish_reason,
                text=output_text + suffix,
                logprobs=(
                    LogProbs(
                        content=[
                            LogProbsContent.model_validate_json(logprob_json_str)
                            for logprob_json_str in logprob_json_strs_list[  # pylint: disable=unsubscriptable-object
                                i
                            ]
                        ]
                    )
                    if logprob_json_strs_list is not None
                    else None
                ),
            )
            for i, (output_text, finish_reason) in enumerate(zip(output_texts, finish_reasons))
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


def check_function_call_usage(
    request: ChatCompletionRequest, conv_template: Conversation
) -> Optional[str]:
    """Check if function calling is used and update the conversation template.
    Return error message if invalid request format for function calling.
    """

    # return if no tools are provided or tool_choice is set to none
    if request.tools is None or (
        isinstance(request.tool_choice, str) and request.tool_choice == "none"
    ):
        conv_template.use_function_calling = False
        return None

    # select the tool based on the tool_choice if specified
    if isinstance(request.tool_choice, dict):
        if request.tool_choice["type"] != "function":
            return "Only 'function' tool choice is supported"

        if len(request.tool_choice["function"]) > 1:
            return "Only one tool is supported when tool_choice is specified"

        for tool in request.tools:
            if tool.function.name == request.tool_choice["function"]["name"]:
                conv_template.use_function_calling = True
                conv_template.function_string = tool.function.model_dump_json()
                return None

        return (
            f"The tool_choice function {request.tool_choice['function']['name']}"
            " is not found in the tools list"
        )

    if isinstance(request.tool_choice, str) and request.tool_choice != "auto":
        return f"Invalid tool_choice value: {request.tool_choice}"

    function_list = []
    for tool in request.tools:
        if tool.type != "function":
            return "Only 'function' tool type is supported"
        function_list.append(tool.function.model_dump())

    conv_template.use_function_calling = True
    conv_template.function_string = json.dumps(function_list)
    return None


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

    # Check for function calling usage and update the conversation template
    error_msg = check_function_call_usage(request, conv_template)
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
        conv_template.messages.append((role, content))
    conv_template.messages.append(("assistant", None))

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
            async_engine.record_event(request_id, event="invoke generate")
            finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
            async for delta_outputs in async_engine.generate(prompt, generation_cfg, request_id):
                assert len(delta_outputs) == generation_cfg.n
                choices = []
                for i, delta_output in enumerate(delta_outputs):
                    finish_reason_updated = False
                    if delta_output.finish_reason is not None and finish_reasons[i] is None:
                        finish_reasons[i] = (
                            delta_output.finish_reason
                            if not conv_template.use_function_calling
                            else "tool_calls"
                        )
                        finish_reason_updated = True
                    if not finish_reason_updated and delta_output.delta_text == "":
                        # Ignore empty delta text when finish reason is not updated.
                        async_engine.record_event(request_id, event="skip empty delta text")
                        continue

                    choices.append(
                        ChatCompletionStreamResponseChoice(
                            index=i,
                            finish_reason=finish_reasons[i],
                            delta=ChatCompletionMessage(
                                content=delta_output.delta_text, role="assistant"
                            ),
                            logprobs=(
                                LogProbs(
                                    content=[
                                        LogProbsContent.model_validate_json(logprob_json_str)
                                        for logprob_json_str in delta_output.delta_logprob_json_strs
                                    ]
                                )
                                if delta_output.delta_logprob_json_strs is not None
                                else None
                            ),
                        )
                    )

                if len(choices) == 0:
                    # Skip yield when there is no delta output.
                    continue
                response = ChatCompletionStreamResponse(
                    id=request_id,
                    choices=choices,
                    model=request.model,
                    system_fingerprint="",
                )
                async_engine.record_event(request_id, event="yield delta output")
                yield f"data: {response.model_dump_json()}\n\n"
            async_engine.record_event(request_id, event="finish")
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    # Normal response.
    output_texts = ["" for _ in range(generation_cfg.n)]
    num_completion_tokens = 0
    finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
    logprob_json_strs_list: Optional[List[List[str]]] = (
        [[] for _ in range(generation_cfg.n)] if generation_cfg.logprobs else None
    )
    async_engine.record_event(request_id, event="invoke generate")
    async for delta_outputs in async_engine.generate(prompt, generation_cfg, request_id):
        if await raw_request.is_disconnected():
            # In non-streaming cases, the engine will not be notified
            # when the request is disconnected.
            # Therefore, we check if it is disconnected each time,
            # and abort the request from engine if so.
            await async_engine.abort(request_id)
            return entrypoint_utils.create_error_response(
                HTTPStatus.BAD_REQUEST, message="The request has disconnected"
            )

        assert len(delta_outputs) == generation_cfg.n
        for i, delta_output in enumerate(delta_outputs):
            if delta_output.finish_reason is not None and finish_reasons[i] is None:
                finish_reasons[i] = delta_output.finish_reason
            output_texts[i] += delta_output.delta_text
            num_completion_tokens += delta_output.num_delta_tokens
            if logprob_json_strs_list is not None:
                assert delta_output.delta_logprob_json_strs is not None
                logprob_json_strs_list[i] += delta_output.delta_logprob_json_strs
    assert all(finish_reason is not None for finish_reason in finish_reasons)

    async_engine.record_event(request_id, event="finish")

    tool_calls_list: List[List[ChatToolCall]] = [[] for _ in range(generation_cfg.n)]
    if conv_template.use_function_calling:
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
                    if (not conv_template.use_function_calling or finish_reason == "error")
                    else ChatCompletionMessage(role="assistant", tool_calls=tool_calls)
                ),
                logprobs=(
                    LogProbs(
                        content=[
                            LogProbsContent.model_validate_json(logprob_json_str)
                            for logprob_json_str in logprob_json_strs_list[  # pylint: disable=unsubscriptable-object
                                i
                            ]
                        ]
                    )
                    if logprob_json_strs_list is not None
                    else None
                ),
            )
            for i, (output_text, finish_reason, tool_calls) in enumerate(
                zip(output_texts, finish_reasons, tool_calls_list)
            )
        ],
        model=request.model,
        system_fingerprint="",
        usage=UsageInfo(prompt_tokens=len(prompt), completion_tokens=num_completion_tokens),
    )
