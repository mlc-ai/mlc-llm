import time
import uuid
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

# TODO(amalyshe): hadnle random_seed
# from .base import set_global_random_seed
from ..api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)
from ..engine.async_connector import AsyncEngineConnector
from ..engine.types import Request, SamplingParams, StoppingCriteria
from .dependencies import get_async_engine_connector


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )


router = APIRouter()


def _get_sampling_params(request: ChatCompletionRequest) -> SamplingParams:
    sampling_params = SamplingParams(
        n=request.n,
        # These params came from vllm
        # TODO(amnalyshe): should they be put into mlc-llm batch serving ChatCompletionRequest?
        # best_of=request.best_of,
        # top_k=request.top_k,
        # ignore_eos=request.ignore_eos,
        # use_beam_search=request.use_beam_search,
    )
    if request.presence_penalty is not None:
        sampling_params.presence_penalty = request.presence_penalty
    if request.frequency_penalty is not None:
        sampling_params.frequency_penalty = request.frequency_penalty
    if request.temperature is not None:
        sampling_params.temperature = request.temperature
    if request.top_p is not None:
        sampling_params.top_p = request.top_p
    return sampling_params


@router.post("/v1/chat/completions")
async def request_completion(
    request: ChatCompletionRequest,
    async_engine_connector: Annotated[
        AsyncEngineConnector, Depends(get_async_engine_connector)
    ],
) -> ChatCompletionResponse:
    """
    Creates model response for the given chat conversation.
    """

    def random_uuid() -> str:
        return str(uuid.uuid4().hex)

    # TODO(amalyshe) remove this verification and handle a case properly
    if len(request.messages) > 1 and not isinstance(request.messages, str):
        raise ValueError(
            """
                The /v1/chat/completions endpoint currently only supports single message prompts.
                Please ensure your request contains only one message
                """
        )

    if request.stream:
        # TODO(amalyshe): handle streamed requests
        raise ValueError(
            """
                Streamsed requests are not supported yet
                """
        )
    else:
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())
        model_name = request.model
        try:
            sampling_params = _get_sampling_params(request)
        except ValueError as e:
            raise ValueError(
                """
                issues with sampling parameters
                """
            )

        result_generator = async_engine_connector.generate(
            Request(
                request_id=request_id,
                prompt=request.messages,
                sampling_params=sampling_params,
                stopping_criteria=StoppingCriteria(max_tokens=request.max_tokens),
            )
        )

        # Non-streaming response
        outputs = []
        async for res in result_generator:
            # TODO: verify that the request cancellation happens after this returns
            outputs.append(res.delta)
        if not outputs:
            return create_error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR, "No text generated"
            )

        choices = []
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="".join(outputs)),
            finish_reason=res.finish_reason,
        )
        choices.append(choice_data)

        # TODO(amalyshe): prompt tokens is also required for openapi output
        # num_prompt_tokens = len(final_res.prompt_token_ids)
        # num_generated_tokens = sum(
        #     len(output.token_ids) for output in final_res.outputs)
        num_prompt_tokens = 1
        num_generated_tokens = 5
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response
