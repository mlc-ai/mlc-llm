"""Programmable router for dispatching OpenAI API to Microserving API"""

import json
import math
import threading
from typing import Any, AsyncGenerator, Iterable, List, Literal, Optional, Tuple

import aiohttp  # pylint: disable=import-error
import tvm

from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve import EngineConfig, PopenServer
from mlc_llm.serve.entrypoints import microserving_entrypoints
from mlc_llm.tokenizers import Tokenizer


class Router:  # pylint: disable=too-many-instance-attributes
    """Programmable Router Implementation"""

    def __init__(
        self,
        model: str,
        model_lib: Optional[str] = None,
        hosts: Optional[List[str]] = None,
        ports: Optional[List[int]] = None,
        num_gpus: Optional[List[int]] = None,
        enable_prefix_cache: bool = False,
        router_mode: Literal["disagg", "round-robin"] = "disagg",
        pd_balance_factor: float = 0.0,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """
        Spawn len(host_list) server endpoints with Popen.
        """
        if hosts is None:
            hosts = ["127.0.0.1"]
        if ports is None:
            ports = [8080]
        if num_gpus is None:
            num_gpus = [1]

        self.router_mode = router_mode
        self.pd_balance_factor = pd_balance_factor
        # Get endpoint urls
        self.num_servers = len(hosts)
        assert self.num_servers == len(ports) == len(num_gpus)
        self.hosts = hosts
        self.ports = ports
        self.server_urls = []
        for i in range(self.num_servers):
            self.server_urls.append(f"http://{hosts[i]}:{ports[i]}")

        # Misc
        self.headers = {"Content-Type": "application/json"}
        self.num_running_requests = [0] * self.num_servers

        # Call nvshmem_init here to get uid, then pass to env variables to server.start() below
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = list(f_init_nvshmem_uid())

        # Start underlying servers concurrently. Otherwise 1 server cannot start on its own
        # since initializing nvhsmem world requires all GPUs.
        self.servers: List[PopenServer] = []

        self.device_id_starts = [0]
        for num_gpus_val in num_gpus:
            self.device_id_starts.append(self.device_id_starts[-1] + num_gpus_val)
        # device_id_starts[-1] is the total number of GPUs.

        def start_server(i: int):
            nvshmem_config = {
                "uid": uid,
                "npes": self.device_id_starts[-1],  # total number of workers in the nvshmem world
                "pe_start": self.device_id_starts[i],  # start of PE for this endpoint's workers
            }

            server = PopenServer(
                model=model,
                model_lib=model_lib,
                host=hosts[i],
                port=ports[i],
                enable_debug=True,
                device=f"cuda:{self.device_id_starts[i]}",
                mode="server",
                engine_config=EngineConfig(
                    prefix_cache_mode="radix" if enable_prefix_cache else "disable",
                    gpu_memory_utilization=0.8,
                ),
            )
            self.servers.append(server)
            server.start(extra_env={"MLC_NVSHMEM_INIT_CONFIG_JSON_STR": json.dumps(nvshmem_config)})

        threads = []
        num_used_gpus = 0
        for i in range(self.num_servers):
            thread = threading.Thread(
                target=start_server,
                args=[i],
            )
            num_used_gpus += num_gpus[i]
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.tokenizer = Tokenizer(model)

    def terminate(self):
        """Terminate the underlying servers"""
        for server in self.servers:
            server.terminate()

    async def handle_completion(
        self,
        request: openai_api_protocol.CompletionRequest,
        request_id: str,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Handle a completion request from API with a schedule.
        """
        if isinstance(request.prompt, str):
            request.prompt = self.tokenizer.encode(request.prompt)
        # Add a debugConfig if not present
        if request.debug_config is None:
            request.debug_config = openai_api_protocol.DebugConfig()
        completed = False
        while not completed:
            completed = True
            async for response in self.translate_request(request, request_id):
                if response is None:
                    completed = False
                    break
                yield response

    async def translate_request(
        self, request: openai_api_protocol.CompletionRequest, request_id: str
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Translate OpenAI API request to microserving API calls.
        """
        if self.router_mode == "disagg":
            async for response in self._handle_completion_disagg(
                request, request_id, pd_balance_factor=self.pd_balance_factor
            ):
                yield response
        elif self.router_mode == "round-robin":
            async for response in self._handle_completion_round_robin(request):
                yield response
        else:
            raise ValueError("Cannot reach here")

    def _pick_endpoint(self, endpoint_ids: Iterable[int]) -> int:
        # Pick the least congested endpoint.
        endpoint_id = -1
        min_running_req = int(1e9)
        for candidate_id in endpoint_ids:
            if self.num_running_requests[candidate_id] < min_running_req:
                min_running_req = self.num_running_requests[candidate_id]
                endpoint_id = candidate_id
        assert endpoint_id != -1
        return endpoint_id

    async def _handle_completion_round_robin(
        self,
        request: openai_api_protocol.CompletionRequest,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Handle a completion request from API. Given a streaming request, yields multiple response
        chunks. Given a non-streaming request, yield a single response. Dispatch request to
        endpoints with round-robin scheduling at a request level.
        """
        # Round robin
        cur_endpoint = self._pick_endpoint(range(self.num_servers))
        self.num_running_requests[cur_endpoint] += 1
        payload = request.model_dump()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3 * 3600), trust_env=True
        ) as session:
            # pylint: disable=fixme
            # todo: replace this with start_generate
            # pylint: enable=fixme
            async with session.post(
                self.server_urls[cur_endpoint] + "/v1/completions",
                json=payload,
                headers=self.headers,
            ) as response:
                assert response.status == 200, await response.text()
                if payload["stream"]:
                    async for chunk in response.content:
                        # Convert raw bytes to CompletionResponse
                        chunk = chunk.strip()
                        if not chunk or chunk == b"\n":
                            continue
                        # Get rid of the prefix "data: " and suffix "\n"
                        raw_data = chunk[6:].strip()
                        if raw_data == b"[DONE]":
                            continue
                        data = json.loads(raw_data)
                        # Commented because we still want usage chunk to be passed back
                        # if not data["choices"]:
                        #     continue
                        response = openai_api_protocol.CompletionResponse.model_validate(data)
                        if response.choices:
                            reason = response.choices[0].finish_reason
                            if reason == "preempt":
                                yield None
                        yield response
                else:
                    data = await response.json()
                    response = openai_api_protocol.CompletionResponse.model_validate(data)
                    if response.choices:
                        reason = response.choices[0].finish_reason
                        if reason == "preempt":
                            yield None
                    yield response
            self.num_running_requests[cur_endpoint] -= 1

    #
    # Below methods are for disaggregated serving
    # Note that only _handle_completion_disagg() has scheduling logics. The other three
    # helper methods only reflect our flow.
    #
    async def _handle_completion_disagg(  # pylint: disable=too-many-locals
        self,
        original_request: openai_api_protocol.CompletionRequest,
        request_id: str,
        pd_balance_factor=0,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Handle a completion request from API with disaggregated scheduling. Given two servers
        P (prefill) and D (decode), the router does the following:
            1. Ask D to prepare metadata, receive D's metadata
            (prefix cache, KV append positions, etc.)
            2. Send P the prefill request and D's metadata, receive ack
            3. Ask D to start decoding, receive response as a normal streaming
        """
        original_request.user = request_id
        # Arbitrarily determine server 0 is P, other servers are D
        prefill_server_id = 0
        decode_server_id = self._pick_endpoint(range(1, self.num_servers))

        # Tell D to prepare metadata for prompt[0:kv_window_end].
        # P does not need to sample. Ask D to treat the last
        # token like the first sampled token.
        kv_window_end = (
            -1
            if math.fabs(pd_balance_factor) < 1e-5
            else int((1 - pd_balance_factor) * len(original_request.prompt))
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3 * 3600), trust_env=True
        ) as session:
            self.num_running_requests[decode_server_id] += 1
            try:
                # 1. Ask D to prepare metadata
                prep_recv_request = microserving_entrypoints.PrepRecvRequest(
                    **original_request.model_dump(), end=kv_window_end
                )
                (
                    kv_append_metadata_base64,
                    prefix_matched_length,
                ) = await self.send_prepare_receive(
                    session=session,
                    request=prep_recv_request,
                    server_url=self.server_urls[decode_server_id],
                )

                kv_window_end = (
                    len(original_request.prompt) + kv_window_end
                    if kv_window_end < 0
                    else kv_window_end
                )
                assert prefix_matched_length <= kv_window_end

                # 2. Send P the prefill request and D's metadata. When it returns, it means that
                # KV transfer has finished prefilling and transferring the KV of
                # prompt[prefix_matched_length:kv_window_end]. So D is ready to decode.
                if prefix_matched_length < kv_window_end:
                    remote_send_request = microserving_entrypoints.RemoteSendRequest(
                        **original_request.model_dump(),
                        begin=prefix_matched_length,
                        end=kv_window_end,
                        kv_addr_info=kv_append_metadata_base64,
                        recv_rank=self.device_id_starts[decode_server_id],
                    )
                    await self.send_remote_send(
                        session=session,
                        request=remote_send_request,
                        server_url=self.server_urls[prefill_server_id],
                    )

                # 3. Start decoding, receive and yield back response as a normal request
                # The kv window passed through denotes the range to prefill on the
                # decode server, which should be [-1:] here.
                start_generate_request = microserving_entrypoints.StartGenerateRequest(
                    **original_request.model_dump(),
                    begin=kv_window_end,
                )
                async for response in self.send_start_generate(
                    session=session,
                    request=start_generate_request,
                    server_url=self.server_urls[decode_server_id],
                ):
                    if len(response.choices) > 0:
                        finish_reason = response.choices[0].finish_reason
                        if finish_reason == "preempt":
                            yield None
                    yield response
            except Exception as e:
                self.num_running_requests[decode_server_id] -= 1
                raise e
            self.num_running_requests[decode_server_id] -= 1

    async def send_prepare_receive(
        self,
        session: aiohttp.ClientSession,
        request: openai_api_protocol.CompletionRequest,
        server_url: str,
    ) -> Tuple[str, int]:
        """
        Performs step 1 of disaggregated serving: ask D to prepare metadata.
        Returns:
            The metadata received from D, which is a tuple of 2 elements:
                - kv_append_metadata_base64: str, info about KV append encoded in base64 string
                - prefix_matched_length: int, length of the matched prefix.
                    i.e. prompt[0:prefix_matched_length] is the matched prefix
        """
        # Send request to the decode server for receive preparation.
        # Get the prompt length, matched prefix length and the KV metadata.
        async with session.post(
            server_url + "/microserving/prep_recv",
            json=request.model_dump(),
            headers=self.headers,
        ) as response:
            assert response.status == 200, await response.text()
            data = await response.json()

            return (
                data["kv_append_metadata"],
                data["prefix_matched_length"],
            )

    async def send_remote_send(
        self,
        session: aiohttp.ClientSession,
        request: openai_api_protocol.CompletionRequest,
        server_url: str,
    ) -> None:
        """
        Performs step 2 of disaggregated serving: ask P to prefill and transfer KV to D.
        P returns an empty chunk to acknowledge completion.
        """
        # Send request to P and get ack
        async with session.post(
            server_url + "/microserving/remote_send",
            json=request.model_dump(),
            headers=self.headers,
        ) as response:
            assert response.status == 200, await response.text()
            await response.json()

    async def send_start_generate(
        self,
        session: aiohttp.ClientSession,
        request: openai_api_protocol.CompletionRequest,
        server_url: str,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Performs step 3 of disaggregated serving: ask D to decode and return normal response.
        """
        # pylint: disable=fixme
        # Todo: return string directly to reduce str->json->str roundtrip overhead
        # pylint: enable=fixme
        async with session.post(
            server_url + "/microserving/start_generate",
            json=request.model_dump(),
            headers=self.headers,
        ) as response:
            assert response.status == 200, await response.text()
            if request.stream:
                async for chunk in response.content:
                    # Convert raw bytes to CompletionResponse
                    chunk = chunk.strip()
                    if not chunk or chunk == b"\n":
                        continue
                    # Get rid of the prefix "data: " and suffix "\n"
                    raw_data = chunk[6:].strip()
                    if raw_data == b"[DONE]":
                        continue
                    data = json.loads(raw_data)
                    # Commented because we still want usage chunk to be passed back
                    # if not data["choices"]:
                    #     continue
                    yield openai_api_protocol.CompletionResponse.model_validate(data)
            else:
                data = await response.json()
                yield openai_api_protocol.CompletionResponse.model_validate(data)
