Implement LLM Cross-engine Orchestration Patterns
======================================================================

In this tutorial, we will introduce how to implement LLM cross-engine
orchestration patterns, like prefill-decode disaggregation, in MLC-LLM
via microserving API. Aiming to make disaggregated serving programmable,
MicroServing provides a new RISC-style approach to design LLM serving
API at sub-request level. It enables programmable cross-engine serving
patterns in a few lines of python code. For more information of
microserving API, check out
https://blog.mlc.ai/2025/01/07/microserving-llm-engines.

Below is an example of prefill-decode disaggregation implementation. An
LLM cross-engine orchestration pattern is implemented in a router, which
dispatches original OpenAI-style completion requests to a chain of
microserving API calls. In this code example, we create a subclass of
Router (which includes wrappers for calling microserving APIs), and
override ``translate_request`` function. The ``translate_request``
function takes in a request and a unique identifier of the request
(``request_id``), and returns an AsyncGenerator of response. We launch
the CustomRouter and 2 engines, each of which has tensor parallel degree
2. Engine 0 is prefill engine and engine 1 is decode engine.

.. code:: python

   from mlc_llm.router import Router
   from mlc_llm.protocol import openai_api_protocol
   from typing import Any, AsyncGenerator
   from mlc_llm.serve.entrypoints import microserving_entrypoints
   from mlc_llm.interface.router import serve

   import aiohttp

   class CustomRouter(Router):
       async def translate_request(self, request: openai_api_protocol.CompletionRequest, request_id: str) -&gt; AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
           pass


   serve(
       model="/path/to/model", # replace this with actual path
       model_lib="/path/to/model_lib", # replace this with actual path
       router_host="127.0.0.1",
       router_port=9123,
       endpoint_hosts=["127.0.0.1", "127.0.0.1"],
       endpoint_ports=[9124,9125],
       endpoint_num_gpus=[2,2],
       enable_prefix_cache=False,
       router_type=CustomRouter,
   )

In the ``translate_request`` function, we first assign ``request_id`` to
request.user, and later the request id will be passed as an argument to
the microserving API.

.. code:: python

   # we will pass request_id as an argument in microserving API calls
   request.user = request_id


Next, call ``prep_recv`` on the decode engine to prepare KV entries for
receiving from remote. ``end=-1`` means that we will let the prefill
engine prefill all except the last token, which makes sure that the
prefill engine does not need sampling logic. ``prep_recv`` returns
address to receive KV from remote and matched prefix length. For
simplicity, we do not enable prefix cache in the tutorial, so we only
need the kv address here.

.. code:: python

   async with aiohttp.ClientSession(
       timeout=aiohttp.ClientTimeout(total=3 * 3600), trust_env=True
   ) as session:
       decode_start = len(request.prompt) -1
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
           server_url=self.server_urls[1], # engine 0 is prefill, engine 1 is decode. Here is decode engine
       )

Then, call ``remote_send`` on the prefill engine to compute and send KV
to decode engine. ``recv_rank=self.device_id_starts[1]`` means that we
are sending KV to engine 1 (decode engine).

.. code:: python


   # 2. Ask prefill engine to send KV to decode engine
   remote_send_request = microserving_entrypoints.RemoteSendRequest(
       **request.model_dump(),
       begin=0,
       end=decode_start,
       kv_addr_info=kv_addr_info,
       recv_rank=self.device_id_starts[1], # the rank of decode engine
   )
   await self.send_remote_send(
       session=session,
       request=remote_send_request,
       server_url=self.server_urls[0], # prefill engine
   )

Finally, call ``start_generate`` on the decode engine to start
generating tokens. ``begin=decode_start`` means we will prefill the last
token in the prompt and start decoding. Notably, the decode process of
the request may be preempted. In such case, we yield None, so that the
router will rerun the ``translate_request`` function.

.. code:: python

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
       if len(response.choices) &gt; 0:
           finish_reason = response.choices[0].finish_reason
           if finish_reason == "preempt":
               yield None
       yield response

Bringing everything together, the complete code is as below:

.. code:: python

   from mlc_llm.router import Router
   from mlc_llm.protocol import openai_api_protocol
   from typing import Any, AsyncGenerator
   from mlc_llm.serve.entrypoints import microserving_entrypoints
   from mlc_llm.interface.router import serve

   import aiohttp
   class CustomRouter(Router):
       async def translate_request(self, request: openai_api_protocol.CompletionRequest, request_id: str) -&gt; AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
           # we will pass request_id as an argument in microserving API calls
           request.user = request_id

           async with aiohttp.ClientSession(
               timeout=aiohttp.ClientTimeout(total=3 * 3600), trust_env=True
           ) as session:
               decode_start = len(request.prompt) -1
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
                   server_url=self.server_urls[1], # engine 0 is prefill, engine 1 is decode. Here is decode engine
               )
               # 2. Ask prefill engine to send KV to decode engine
               remote_send_request = microserving_entrypoints.RemoteSendRequest(
                   **request.model_dump(),
                   begin=0,
                   end=decode_start,
                   kv_addr_info=kv_addr_info,
                   recv_rank=self.device_id_starts[1], # the rank of decode engine
               )
               await self.send_remote_send(
                   session=session,
                   request=remote_send_request,
                   server_url=self.server_urls[0], # prefill engine
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
                   if len(response.choices) &gt; 0:
                       finish_reason = response.choices[0].finish_reason
                       if finish_reason == "preempt":
                           yield None
                   yield response


   serve(
       model="/path/to/model", # replace this with actual path
       model_lib="/path/to/model_lib", # replace this with actual path
       router_host="127.0.0.1",
       router_port=9123,
       endpoint_hosts=["127.0.0.1", "127.0.0.1"],
       endpoint_ports=[9124,9125],
       endpoint_num_gpus=[2,2],
       enable_prefix_cache=False,
       router_type=CustomRouter,
   )
