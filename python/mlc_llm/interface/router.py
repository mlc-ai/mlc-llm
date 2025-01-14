"""Python entrypoint of router."""

# pylint: disable=fixme
from http import HTTPStatus
from typing import AsyncGenerator, List, Literal, Optional, Type

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from mlc_llm.protocol import error_protocol
from mlc_llm.protocol.openai_api_protocol import CompletionLogProbs, CompletionRequest
from mlc_llm.router import Router
from mlc_llm.serve import engine_base, engine_utils


def serve(
    model: str,
    model_lib: Optional[str],
    router_host: str,
    router_port: int,
    endpoint_hosts: List[str],
    endpoint_ports: List[int],
    endpoint_num_gpus: List[int],
    enable_prefix_cache: bool,
    router_mode: Literal["disagg", "round-robin"] = "round-robin",
    pd_balance_factor: float = 0.0,
    router_type: Type[Router] = Router,
):  # pylint: disable=too-many-arguments
    """Start the router with the specified configuration."""
    # 1. Instantiate router
    router = router_type(
        model=model,
        model_lib=model_lib,
        hosts=endpoint_hosts,
        ports=endpoint_ports,
        num_gpus=endpoint_num_gpus,
        enable_prefix_cache=enable_prefix_cache,
        router_mode=router_mode,
        pd_balance_factor=pd_balance_factor,
    )

    router_app = fastapi.APIRouter()

    @router_app.post("/v1/completions")
    async def request_completion(request: CompletionRequest, raw_request: fastapi.Request):
        """OpenAI-compatible completion API.
        API reference: https://platform.openai.com/docs/api-reference/completions/create
        """
        if router is None:
            return error_protocol.create_error_response(
                HTTPStatus.BAD_REQUEST, message="Router is not initialized."
            )
        request_id = f"cmpl-{engine_utils.random_uuid()}"

        # Streaming response.
        if request.stream:
            # We manually get the first response from generator to
            # capture potential exceptions in this scope, rather then
            # the StreamingResponse scope.
            stream_generator = router.handle_completion(  # pylint: disable=protected-access
                request, request_id
            )
            first_response = await anext(  # type: ignore  # pylint: disable=undefined-variable
                stream_generator
            )

            async def completion_stream_generator() -> AsyncGenerator[str, None]:
                if isinstance(first_response, StopAsyncIteration):
                    yield "data: [DONE]\n\n"
                    return
                yield f"data: {first_response.model_dump_json(by_alias=True)}\n\n"
                async for response in stream_generator:
                    yield f"data: {response.model_dump_json(by_alias=True)}\n\n"
                yield "data: [DONE]\n\n"

            return fastapi.responses.StreamingResponse(
                completion_stream_generator(), media_type="text/event-stream"
            )

        # FIXME: Non-streaming response not fully implemented
        request_final_usage = None
        output_texts = [""] * request.n
        finish_reasons: List[Optional[str]] = [None] * request.n
        logprob_results: List[Optional[CompletionLogProbs]] = [None] * request.n

        async for response in router.handle_completion(  # pylint: disable=protected-access
            request, request_id
        ):
            if await raw_request.is_disconnected():
                # In non-streaming cases, the engine will not be notified
                # when the request is disconnected.
                # Therefore, we check if it is disconnected each time,
                # and explicitly return.
                # Note that requesta abort is triggered when the async for and funciton scope ends.
                return error_protocol.create_error_response(
                    HTTPStatus.BAD_REQUEST, message="The request has disconnected"
                )
            # TODO(Charlie): This is copied from engine.py --
            # why is it here? Non-streaming only has a single chunk right?
            # this is the final chunk
            # if response.usage is not None:
            #     request_final_usage = response.usage
            #     continue
            for choice in response.choices:
                output_texts[choice.index] += choice.text
                if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                    finish_reasons[choice.index] = choice.finish_reason
                if choice.logprobs is not None:
                    logprob_results[choice.index] = choice.logprobs

        assert all(finish_reason is not None for finish_reason in finish_reasons)
        return engine_base.wrap_completion_response(
            request_id=request_id,
            model=request.model,
            output_texts=output_texts,
            finish_reasons=finish_reasons,
            logprob_results=logprob_results,
            usage=request_final_usage,
        )

    # 2. Set up app
    app = fastapi.FastAPI()
    app.add_middleware(CORSMiddleware)
    app.include_router(router_app)
    app.exception_handler(error_protocol.BadRequestError)(error_protocol.bad_request_error_handler)

    # 3. Run
    uvicorn.run(app, host=router_host, port=router_port, log_level="info")
