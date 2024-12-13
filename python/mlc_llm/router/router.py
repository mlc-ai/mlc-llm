import json
import math
import threading
from typing import Any, AsyncGenerator, Iterable, List, Literal, Optional, Tuple

import aiohttp
import tvm

from mlc_llm.protocol import debug_protocol, openai_api_protocol
from mlc_llm.serve import EngineConfig, PopenServer
from mlc_llm.tokenizers import Tokenizer


class Router:

    def __init__(
        self,
        model: str,
        model_lib: Optional[str] = None,
        hosts: List[str] = ["127.0.0.1"],
        ports: List[int] = [8080],
        num_gpus: List[int] = [1],
        enable_prefix_cache: bool = False,
        router_mode: Literal["disagg", "round-robin"] = "disagg",
        pd_balance_factor: float = 0.0,
    ):
        """
        Spawn len(host_list) server endpoints with Popen.
        """
        self.router_mode = router_mode
        self.pd_balance_factor = pd_balance_factor
        # Get endpoint urls
        self.num_endpoints = len(hosts)
        assert self.num_endpoints == len(ports) == len(num_gpus)
        self.hosts = hosts
        self.ports = ports
        self.endpoints = []
        for i in range(self.num_endpoints):
            self.endpoints.append(f"http://{hosts[i]}:{ports[i]}/v1/completions")

        # Misc
        self.headers = {"Content-Type": "application/json"}
        self.num_running_requests = [0] * self.num_endpoints

        # Call nvshmem_init here to get uid, then pass to env variables to server.start() below
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = list(f_init_nvshmem_uid())

        # Start underlying endpoints concurrently. Otherwise 1 server cannot start on its own
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
        for i in range(self.num_endpoints):
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
        """Terminate the underlying endpoints"""
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
        if self.router_mode == "disagg":
            async for response in self._handle_completion_disagg(
                request, request_id, pd_balance_factor=self.pd_balance_factor
            ):
                yield response
        elif self.router_mode == "round-robin":
            async for response in self._handle_completion_round_robin(request, request_id):
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
        request_id: str,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Handle a completion request from API. Given a streaming request, yields multiple response
        chunks. Given a non-streaming request, yield a single response. Dispatch request to
        endpoints with round-robin scheduling at a request level.
        """
        # Round robin
        cur_endpoint = self._pick_endpoint(range(self.num_endpoints))
        self.num_running_requests[cur_endpoint] += 1
        payload = request.model_dump()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3 * 3600), trust_env=True
        ) as session:
            async with session.post(
                self.endpoints[cur_endpoint], json=payload, headers=self.headers
            ) as response:
                assert response.status == 200, await response.text()
                completed = False
                while not completed:
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
                                    break
                                elif reason != None:
                                    completed = True
                            yield response
                    else:
                        data = await response.json()
                        response = openai_api_protocol.CompletionResponse.model_validate(data)
                        if response.choices:
                            reason = response.choices[0].finish_reason
                            if reason == "preempt":
                                break
                            elif reason != None:
                                completed = True
                        yield response
            self.num_running_requests[cur_endpoint] -= 1

    #
    # Below methods are for disaggregated serving
    # Note that only _handle_completion_disagg() has scheduling logics. The other three
    # helper methods only reflect our flow.
    #
    async def _handle_completion_disagg(
        self,
        original_request: openai_api_protocol.CompletionRequest,
        request_id: str,
        pd_balance_factor=0,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Handle a completion request from API with disaggregated scheduling. Given two servers
        P (prefill) and D (decode), the router does the following:
            1. Ask D to prepare metadata, receive D's metadata (prefix cache, KV append positions, etc.)
            2. Send P the prefill request and D's metadata, receive ack
            3. Ask D to start decoding, receive response as a normal streaming
        """
        original_request.user = request_id
        # Arbitrarily determine server 0 is P, other servers are D
        prefill_server_id = 0
        decode_server_id = self._pick_endpoint(range(1, self.num_endpoints))

        # Add a debugConfig if not present
        if original_request.debug_config is None:
            original_request.debug_config = openai_api_protocol.DebugConfig()

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
                completed = False
                while not completed:
                    # 1. Ask D to prepare metadata
                    prepare_request = original_request.model_copy()
                    prepare_request.debug_config.disagg_config = debug_protocol.DisaggConfig(
                        kind="prepare_prefill",
                        kv_window_begin=0,  # always zero for prepare_prefill
                        kv_window_end=kv_window_end,
                    )
                    prepare_request.stream_options = openai_api_protocol.StreamOptions(
                        include_usage=True
                    )
                    prompt_length, prefix_matched_length, kv_append_metadata_base64 = (
                        await self.send_decode_prepare(
                            session=session,
                            prepare_request=prepare_request,
                            decode_endpoint=self.endpoints[decode_server_id],
                        )
                    )
                    kv_window_end = (
                        prompt_length + kv_window_end if kv_window_end < 0 else kv_window_end
                    )
                    assert prefix_matched_length <= kv_window_end

                    # 2. Send P the prefill request and D's metadata. When it returns, it means that
                    # KV transfer has finished prefilling and transferring the KV of
                    # prompt[prefix_matched_length:kv_window_end]. So D is ready to decode.
                    if prefix_matched_length < kv_window_end:
                        prefill_request = original_request.model_copy()
                        prefill_request.stream_options = openai_api_protocol.StreamOptions(
                            include_usage=True
                        )
                        prefill_request.debug_config.disagg_config = debug_protocol.DisaggConfig(
                            kind="remote_prefill",
                            kv_window_begin=prefix_matched_length,
                            kv_window_end=kv_window_end,
                            kv_append_metadata=kv_append_metadata_base64,
                            dst_group_offset=self.device_id_starts[decode_server_id],
                        )
                        await self.send_prefill(
                            session=session,
                            prefill_request=prefill_request,
                            prefill_endpoint=self.endpoints[prefill_server_id],
                        )

                    # 3. Start decoding, receive and yield back response as a normal request
                    # The kv window passed through denotes the range to prefill on the
                    # decode server, which should be [-1:] here.
                    decode_request = original_request.model_copy()
                    decode_request.debug_config.disagg_config = debug_protocol.DisaggConfig(
                        kind="start_decode",
                        kv_window_begin=kv_window_end,
                    )
                    async for response in self.send_decode(
                        session=session,
                        decode_request=decode_request,
                        decode_endpoint=self.endpoints[decode_server_id],
                    ):
                        response_json = response.dict()
                        if response_json["choices"]:
                            reason = response_json["choices"][0]["finish_reason"]
                            if reason == "preempt":
                                break
                            elif reason != None:
                                completed = True
                        yield response
            except Exception as e:
                self.num_running_requests[decode_server_id] -= 1
                raise e
            self.num_running_requests[decode_server_id] -= 1

    async def send_decode_prepare(
        self,
        session: aiohttp.ClientSession,
        prepare_request: openai_api_protocol.CompletionRequest,
        decode_endpoint: str,
    ) -> Tuple[int, int, str]:
        """
        Performs step 1 of disaggregated serving: ask D to prepare metadata.
        Returns:
            The metadata received from D, which is a tuple of 2 elements:
                - prompt_length, which is the raw prompt length of the request.
                - prefix_matched_length: int, length of the matched prefix.
                    i.e. prompt[0:prefix_matched_length] is the matched prefix
                - kv_append_metadata_base64: str, info about KV append encoded in base64 string
        """
        # Send request to D and get metadata
        async with session.post(decode_endpoint, json=prepare_request.model_dump()) as response:
            assert response.status == 200, await response.text()
            # Expect decode to only return a single usage chunk
            data = None
            async for chunk in response.content:
                if prepare_request.stream:
                    chunk = chunk.strip()
                    if not chunk or chunk == b"\n":
                        continue
                    # Get rid of the prefix "data: " and suffix "\n"
                    raw_data = chunk[6:].strip()
                    if raw_data == b"[DONE]":
                        continue
                    assert (
                        data is None
                    ), f"Expecting only one effective chunk response. data: {data}, current={json.loads(raw_data)}"
                    data = json.loads(raw_data)
                else:
                    data = await response.json()

            assert "extra" in data["usage"]
            assert "prefix_matched_length" in data["usage"]["extra"]
            assert "kv_append_metadata" in data["usage"]["extra"]

            return (
                data["usage"]["extra"]["prompt_length"],
                data["usage"]["extra"]["prefix_matched_length"],
                data["usage"]["extra"]["kv_append_metadata"],
            )

    async def send_prefill(
        self,
        session: aiohttp.ClientSession,
        prefill_request: openai_api_protocol.CompletionRequest,
        prefill_endpoint: str,
    ) -> None:
        """
        Performs step 2 of disaggregated serving: ask P to prefill and transfer KV to D.
        P returns an empty chunk to acknowledge completion.
        """
        # Send request to P and get ack
        async with session.post(prefill_endpoint, json=prefill_request.model_dump()) as response:
            assert response.status == 200, await response.text()
            # Expect decode to only return an empty chunk
            data = None
            async for chunk in response.content:
                if prefill_request.stream:
                    chunk = chunk.strip()
                    if not chunk or chunk == b"\n":
                        continue
                    # Get rid of the prefix "data: " and suffix "\n"
                    raw_data = chunk[6:].strip()
                    if raw_data == b"[DONE]":
                        continue
                    assert data is None, "Expecting only one effective chunk response."
                    data = json.loads(raw_data)
                else:
                    data = await response.json()

            assert "extra" in data["usage"]
            return

    async def send_decode(
        self,
        session: aiohttp.ClientSession,
        decode_request: openai_api_protocol.CompletionRequest,
        decode_endpoint: str,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """
        Performs step 3 of disaggregated serving: ask D to decode and return normal response.
        """
        # Todo: return string directly to reduce str->json->str roundtrip overhead
        async with session.post(
            decode_endpoint, json=decode_request.model_dump(), headers=self.headers
        ) as response:
            assert response.status == 200, await response.text()
            if decode_request.stream:
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
