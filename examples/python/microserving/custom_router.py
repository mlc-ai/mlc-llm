"""Microserving customized router example."""

from typing import Any, AsyncGenerator

import aiohttp  # pylint: disable=import-error

from mlc_llm.interface.router import serve
from mlc_llm.protocol import openai_api_protocol
from mlc_llm.router import Router
from mlc_llm.serve.entrypoints import microserving_entrypoints


class CustomRouter(Router):
    """A customized router class in Microserving."""

    async def translate_request(
        self, request: openai_api_protocol.CompletionRequest, request_id: str
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        # we will pass request_id as an argument in microserving API calls
        request.user = request_id

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3 * 3600), trust_env=True
        ) as session:
            decode_start = len(request.prompt) - 1
            # 1. Ask decode engine to prepare KV entries to receive from prefill engine
            prep_recv_request = microserving_entrypoints.PrepRecvRequest(
                **request.model_dump(), end=decode_start
            )
            (
                kv_addr_info,
                _,
            ) = await self.send_prepare_receive(
                session=session,
                request=prep_recv_request,
                server_url=self.server_urls[
                    1
                ],  # engine 0 is prefill, engine 1 is decode. Here is decode engine
            )
            # 2. Ask prefill engine to send KV to decode engine
            remote_send_request = microserving_entrypoints.RemoteSendRequest(
                **request.model_dump(),
                begin=0,
                end=decode_start,
                kv_addr_info=kv_addr_info,
                recv_rank=self.device_id_starts[1],  # the rank of decode engine
            )
            await self.send_remote_send(
                session=session,
                request=remote_send_request,
                server_url=self.server_urls[0],  # prefill engine
            )
            # 3. Start decoding
            start_generate_request = microserving_entrypoints.StartGenerateRequest(
                **request.model_dump(),
                begin=decode_start,
            )
            async for response in self.send_start_generate(
                session=session,
                request=start_generate_request,
                server_url=self.server_urls[1],
            ):
                if len(response.choices) > 0:
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == "preempt":
                        yield None
                yield response


serve(
    model="/path/to/model",  # replace this with actual path
    model_lib="/path/to/model_lib.so",  # replace this with actual path
    router_host="127.0.0.1",
    router_port=9123,
    endpoint_hosts=["127.0.0.1", "127.0.0.1"],
    endpoint_ports=[9124, 9125],
    endpoint_num_gpus=[2, 2],
    enable_prefix_cache=False,
    router_type=CustomRouter,
)
