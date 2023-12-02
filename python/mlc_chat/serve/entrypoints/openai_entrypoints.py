"""OpenAI API-compatible server entrypoints in MLC LLM"""
# pylint: disable=too-many-locals,too-many-return-statements
from http import HTTPStatus
from typing import AsyncGenerator, Optional

import fastapi

from ...protocol import protocol_utils
from ...protocol.openai_api_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ListResponse,
    ModelResponse,
    UsageInfo,
)
from ..server_context import ServerContext
from . import entrypoint_utils

app = fastapi.APIRouter()


@app.get("/v1/models")
async def request_models():
    """OpenAI-compatible served model query API.
    API reference: https://platform.openai.com/docs/api-reference/models
    """
    return ListResponse(data=[ModelResponse(id=model) for model in ServerContext.get_model_list()])


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

    # - Check if unsupported arguments are specified.
    error = entrypoint_utils.check_unsupported_fields(request)
    if error is not None:
        return error

    # - Process prompt and check validity.
    prompts = entrypoint_utils.process_prompts(request.prompt, async_engine.tokenizer.encode)
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
    request_id = f"cmpl-{entrypoint_utils.random_uuid()}"

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
            async for delta_text, num_delta_tokens, finish_reason in async_engine.generate(
                prompt, generation_cfg, request_id
            ):
                num_completion_tokens += num_delta_tokens
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

            yield "data: [Done]\n\n"

        return fastapi.responses.StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    # Normal response.
    output_text = "" if not request.echo else async_engine.tokenizer.decode(prompt)
    num_completion_tokens = 0
    finish_reason: Optional[str] = None
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
