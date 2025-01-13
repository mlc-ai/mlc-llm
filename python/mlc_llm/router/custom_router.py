from ..router import Router
from mlc_llm.protocol import openai_api_protocol
from typing import Any, AsyncGenerator, Iterable, List, Literal, Optional, Tuple
from mlc_llm.serve.entrypoints import microserving_entrypoints

import aiohttp
class CustomRouter(Router):
    async def pick_strategy(self, request: openai_api_protocol.CompletionRequest, request_id: str) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        #shall we keep this hack?
        request.user = request_id
        if request.debug_config is None:
            request.debug_config = openai_api_protocol.DebugConfig()
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3 * 3600), trust_env=True
        ) as session:
            # 1. Ask D to prepare metadata
            prep_recv_request = microserving_entrypoints.PrepRecvRequest(
                **request.model_dump(), end=-1
            )
            (
                kv_append_metadata_base64,
                _,
            ) = await self.send_prepare_receive(
                session=session,
                request=prep_recv_request,
                server_url=self.server_urls[1], # server 0 is P, server 1 is D
            )
            kv_window_end = (
                len(request.prompt) + kv_window_end if kv_window_end < 0 else kv_window_end
            )
            remote_send_request = microserving_entrypoints.RemoteSendRequest(
                **request.model_dump(),
                begin=0,
                end=kv_window_end,
                kv_addr_info=kv_append_metadata_base64,
                recv_rank=self.device_id_starts[1],
            )
            await self.send_remote_send(
                session=session,
                request=remote_send_request,
                server_url=self.server_urls[0],
            )
            # 3. Start decoding, receive and yield back response as a normal request
            # The kv window passed through denotes the range to prefill on the
            # decode server, which should be [-1:] here.
            start_generate_request = microserving_entrypoints.StartGenerateRequest(
                **request.model_dump(),
                begin=kv_window_end,
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