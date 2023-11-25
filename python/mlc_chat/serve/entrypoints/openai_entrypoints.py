"""OpenAI API-compatible server entrypoints in MLC LLM"""
# pylint: disable=too-many-locals,too-many-return-statements
from http import HTTPStatus
from typing import AsyncGenerator, Callable, List, Optional, Tuple

import fastapi

from ...protocol import protocol_utils
from ...protocol.openai_api_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    UsageInfo,
)
from ..server_variables import engine, hosted_model_id
from . import entrypoint_utils

app = fastapi.APIRouter()


@app.post("/v1/completions")
async def request_completion(
    request: CompletionRequest, raw_request: fastapi.Request
) -> fastapi.responses.JSONResponse:
    """OpenAI-compatible completion API.
    API reference: https://platform.openai.com/docs/api-reference/completions/create
    """
    # - Check the model id.
    # - Check if unsupported arguments are specified.
    error_checks: List[Tuple[Callable, Tuple]] = [
        (entrypoint_utils.check_model_id, (request.model, hosted_model_id)),
        (entrypoint_utils.check_unsupported_fields, (request,)),
    ]
    for ferror_check, args in error_checks:
        error = ferror_check(*args)
        if error is not None:
            return error

    # - Process prompt and check validity.
    prompts = entrypoint_utils.process_prompts(request.prompt, engine.engine.tokenize)
    if isinstance(prompts, fastapi.responses.JSONResponse):
        # Errored when processing the prompts
        return prompts
    if len(prompts) > 1:
        return entrypoint_utils.create_error_response(
            HTTPStatus.BAD_REQUEST,
            message="Entrypoint /v1/completions only accept single prompt. "
            f"However, {len(prompts)} prompts {prompts} are received.",
        )
    error = entrypoint_utils.check_prompts_length(prompts, engine.engine.max_single_sequence_length)
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
            prev_text_len = 0

            # - Echo back the prompt.
            if request.echo:
                text = engine.engine.detokenize(prompt)
                prev_text_len = len(text)
                response = CompletionResponse(
                    id=request_id,
                    choices=[CompletionResponseChoice(text=text)],
                    model=hosted_model_id,
                    usage=UsageInfo(
                        prompt_tokens=len(prompt),
                        completion_tokens=0,
                    ),
                )
                yield f"data: {response.model_dump_json()}\n\n"

            # - Generate new tokens.
            tokens = list(prompt)
            finish_reason = None
            async for output_token, finish_reason in engine.generate(
                prompt, generation_cfg, request_id
            ):
                tokens.append(output_token)
                text = engine.engine.detokenize(tokens)
                delta_text = text[prev_text_len:]
                prev_text_len = len(text)

                response = CompletionResponse(
                    id=request_id,
                    choices=[
                        CompletionResponseChoice(
                            finish_reason=finish_reason,
                            text=delta_text,
                        )
                    ],
                    model=hosted_model_id,
                    usage=UsageInfo(
                        prompt_tokens=len(prompt),
                        completion_tokens=len(tokens) - len(prompt),
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
                    model=hosted_model_id,
                    usage=UsageInfo(
                        prompt_tokens=len(prompt),
                        completion_tokens=len(tokens) - len(prompt),
                    ),
                )
                yield f"data: {response.model_dump_json()}\n\n"

            yield "data: [Done]\n\n"

        return fastapi.responses.StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    # Normal response.
    output_tokens = []
    finish_reason: Optional[str] = None
    async for output_token, finish_reason in engine.generate(prompt, generation_cfg, request_id):
        if await raw_request.is_disconnected():
            # In non-streaming cases, the engine will not be notified
            # when the request is disconnected.
            # Therefore, we check if it is disconnected each time,
            # and abort the request from engine if so.
            await engine.abort(request_id)
            return entrypoint_utils.create_error_response(
                HTTPStatus.BAD_REQUEST, message="The request has disconnected"
            )
        output_tokens.append(output_token)
    assert finish_reason is not None
    text = engine.engine.detokenize(prompt + output_tokens if request.echo else output_tokens)
    suffix = request.suffix if request.suffix is not None else ""
    response = CompletionResponse(
        id=request_id,
        choices=[
            CompletionResponseChoice(
                finish_reason=finish_reason,
                text=text + suffix,
            )
        ],
        model=hosted_model_id,
        usage=UsageInfo(
            prompt_tokens=len(prompt),
            completion_tokens=len(output_tokens),
        ),
    )
    return response
