import argparse
import time
import uuid
from http import HTTPStatus
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import Request
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from dataclasses import dataclass, field, fields

# TODO(amalyshe): hadnle random_seed
# from .base import set_global_random_seed
from ..api.protocol import ChatCompletionRequest, ChatCompletionResponseChoice, ChatMessage, UsageInfo, ChatCompletionResponse, ErrorResponse

import numpy as np

from .serving_layer import ServingLayer
from .sampling_params import SamplingParams
from .types import RequestOutput
from .arg_utils import EngineArgs

def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)

@dataclass
class RestAPIArgs:
    """RestAPIArgs is the dataclass that organizes the arguments used for starting a REST API server."""
    host: str = field(
        default="127.0.0.1",
        metadata={
            "help": (
                """
                The host at which the server should be started, defaults to ``127.0.0.1``.
                """
            )
        }
    )
    port: int = field(
        default=8000,
        metadata={
            "help": (
                """
                The port on which the server should be started, defaults to ``8000``.
                """
            )
        }
    )

def convert_args_to_argparser() -> argparse.ArgumentParser:
    """Convert from RestAPIArgs to an equivalent ArgumentParser."""
    args = argparse.ArgumentParser("MLC Chat REST API")
    for field in fields(RestAPIArgs):
        name = field.name.replace("_", "-")
        field_name = f"--{name}"
        # `kwargs` contains `help`, `choices`, and `action`
        kwargs = field.metadata.copy()
        if field.type == bool:
            # boolean arguments do not need to specify `type`
            args.add_argument(field_name, default=field.default, **kwargs)
        else:
            args.add_argument(field_name, type=field.type, default=field.default, **kwargs)
    return args


engine = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # that was in original mlc-llm rest api, but since we migrated from mlc-llm/python, need to figure out what to do with this param
    # TODO(amalyshe): handle random_seed
    # if ARGS.random_seed is not None:
    #     set_global_random_seed(ARGS.random_seed)
    engine_args = EngineArgs.from_cli_args(ARGS)
    engine = ServingLayer.from_engine_args(engine_args)

    yield

    engine = None


app = FastAPI(lifespan=lifespan)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def request_completion(request: ChatCompletionRequest,
                             raw_request: Request):
    """
    Creates model response for the given chat conversation.
    """

    global engine
    def random_uuid() -> str:
        return str(uuid.uuid4().hex)

    # TODO(amalyshe) remove this verification and handle a case properly
    if len(request.messages) > 1 and not isinstance(request.messages, str):
            raise ValueError(
                """
                The /v1/chat/completions endpoint currently only supports single message prompts.
                Please ensure your request contains only one message
                """)

    if request.stream:
        # TODO(amalyshe): handle streamed requests
        raise ValueError(
                """
                Streamsed requests are not supported yet
                """)
    else:
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())
        model_name = request.model
        try:
            sampling_params = SamplingParams(
                n=request.n,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                max_tokens=request.max_tokens,
                # These params came from vllm
                # TODO(amnalyshe): should they be put into mlc-llm batch serving ChatCompletionRequest?
                # best_of=request.best_of,
                # top_k=request.top_k,
                # ignore_eos=request.ignore_eos,
                # use_beam_search=request.use_beam_search,
            )
        except ValueError as e:
            raise ValueError( """
                issues with sampling parameters
                """)

        result_generator = engine.generate(request.messages, request.model, sampling_params, request_id)

        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine._engine_abort({request_id,})
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                            "Client disconnected")
            final_res = res
        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role="assistant", content=output.text),
                finish_reason=output.finish_reason,
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

ARGS = convert_args_to_argparser()
ARGS = EngineArgs.add_cli_args(ARGS)
ARGS = ARGS.parse_args()
if __name__ == "__main__":
    uvicorn.run("mlc_serve.engine.rest:app", host=ARGS.host, port=ARGS.port, reload=False, access_log=False)
