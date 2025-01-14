.. _deploy-rest-api:

REST API
========

.. contents:: Table of Contents
   :local:
   :depth: 2

We provide `REST API <https://www.ibm.com/topics/rest-apis#:~:text=the%20next%20step-,What%20is%20a%20REST%20API%3F,representational%20state%20transfer%20architectural%20style.>`_
for a user to interact with MLC-LLM in their own programs.

Install MLC-LLM Package
------------------------

SERVE is a part of the MLC-LLM package, installation instruction for which can be found :ref:`here <install-mlc-packages>`. Once you have install the MLC-LLM package, you can run the following command to check if the installation was successful:

.. code:: bash

   mlc_llm serve --help

You should see serve help message if the installation was successful.

Quick Start
------------

This section provides a quick start guide to work with MLC-LLM REST API. To launch a server, run the following command:

.. code:: bash

   mlc_llm serve MODEL [--model-lib PATH-TO-MODEL-LIB]

where ``MODEL`` is the model folder after compiling with :ref:`MLC-LLM build process <compile-model-libraries>`. Information about other arguments can be found under :ref:`Launch the server <rest_launch_server>` section.

Once you have launched the Server, you can use the API in your own program to send requests. Below is an example of using the API to interact with MLC-LLM in Python without Streaming (suppose the server is running on ``http://127.0.0.1:8080/``):

.. code:: bash

   import requests

   # Get a response using a prompt without streaming
   payload = {
      "model": "./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/",
      "messages": [
         {"role": "user", "content": "Write a haiku about apples."},
      ],
      "stream": False,
      # "n": 1,
      "max_tokens": 300,
   }
   r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload)
   choices = r.json()["choices"]
   for choice in choices:
      print(f"{choice['message']['content']}\n")

Run CLI with Multi-GPU
----------------------

If you want to enable tensor parallelism to run LLMs on multiple GPUs, please specify argument ``--overrides "tensor_parallel_shards=$NGPU"``. For example,

.. code:: shell

   mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC --overrides "tensor_parallel_shards=2"

------------------------------------------------


.. _rest_launch_server:


Launch the Server
-----------------

To launch the MLC Server for MLC-LLM, run the following command in your terminal.

.. code:: bash

   mlc_llm serve MODEL [--model-lib PATH-TO-MODEL-LIB] [--device DEVICE] [--mode MODE] \
       [--additional-models ADDITIONAL-MODELS] \
       [--speculative-mode SPECULATIVE-MODE] \
       [--overrides OVERRIDES] \
       [--enable-tracing] \
       [--host HOST] \
       [--port PORT] \
       [--allow-credentials] \
       [--allowed-origins ALLOWED_ORIGINS] \
       [--allowed-methods ALLOWED_METHODS] \
       [--allowed-headers ALLOWED_HEADERS]


MODEL                  The model folder after compiling with MLC-LLM build process. The parameter
                       can either be the model name with its quantization scheme
                       (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
                       folder. In the former case, we will use the provided name to search
                       for the model folder over possible paths.

--model-lib            A field to specify the full path to the model library file to use (e.g. a ``.so`` file).
--device               The description of the device to run on. User should provide a string in the
                       form of ``device_name:device_id`` or ``device_name``, where ``device_name`` is one of
                       ``cuda``, ``metal``, ``vulkan``, ``rocm``, ``opencl``, ``auto`` (automatically detect the
                       local device), and ``device_id`` is the device id to run on. The default value is ``auto``,
                       with the device id set to 0 for default.
--mode                 The engine mode in MLC LLM.
                       We provide three preset modes: ``local``, ``interactive`` and ``server``.
                       The default mode is ``local``.

                       The choice of mode decides the values of "max_num_sequence", "max_total_sequence_length"
                       and "prefill_chunk_size" when they are not explicitly specified.

                       1. Mode "local" refers to the local server deployment which has low
                       request concurrency. So the max batch size will be set to 4, and max
                       total sequence length and prefill chunk size are set to the context
                       window size (or sliding window size) of the model.

                       2. Mode "interactive" refers to the interactive use of server, which
                       has at most 1 concurrent request. So the max batch size will be set to 1,
                       and max total sequence length and prefill chunk size are set to the context
                       window size (or sliding window size) of the model.

                       3. Mode "server" refers to the large server use case which may handle
                       many concurrent request and want to use GPU memory as much as possible.
                       In this mode, we will automatically infer the largest possible max batch
                       size and max total sequence length.

                       You can manually specify arguments "max_num_sequence", "max_total_seq_length" and
                       "prefill_chunk_size" via ``--overrides`` to override the automatic inferred values.
                       For example: ``--overrides "max_num_sequence=32;max_total_seq_length=4096"``.
--additional-models    The model paths and (optional) model library paths of additional models (other
                       than the main model).

                       When engine is enabled with speculative decoding, additional models are needed.
                       **We only support one additional model for speculative decoding now.**
                       The way of specifying the additional model is:
                       ``--additional-models model_path_1`` or
                       ``--additional-models model_path_1,model_lib_1``.

                       When the model lib of a model is not given, JIT model compilation will be activated
                       to compile the model automatically.
--speculative-mode     The speculative decoding mode. Right now four options are supported:

                       - ``disable``, where speculative decoding is not enabled,

                       - ``small_draft``, denoting the normal speculative decoding (small draft) style,

                       - ``eagle``, denoting the eagle-style speculative decoding.

                       - ``medusa``, denoting the medusa-style speculative decoding.
--overrides            Overriding extra configurable fields of EngineConfig.

                       Supporting fields that can be be overridden: ``tensor_parallel_shards``, ``max_num_sequence``,
                       ``max_total_seq_length``, ``prefill_chunk_size``, ``max_history_size``, ``gpu_memory_utilization``,
                       ``spec_draft_length``, ``prefix_cache_max_num_recycling_seqs``, ``context_window_size``,
                       ``sliding_window_size``, ``attention_sink_size``.

                       Please check out the documentation of EngineConfig in ``mlc_llm/serve/config.py``
                       for detailed docstring of each field.
                       Example: ``--overrides "max_num_sequence=32;max_total_seq_length=4096;tensor_parallel_shards=2"``
--enable-tracing       A boolean indicating if to enable event logging for requests.
--host                 The host at which the server should be started, defaults to ``127.0.0.1``.
--port                 The port on which the server should be started, defaults to ``8000``.
--allow-credentials    A flag to indicate whether the server should allow credentials. If set, the server will
                       include the ``CORS`` header in the response
--allowed-origins      Specifies the allowed origins. It expects a JSON list of strings, with the default value being ``["*"]``, allowing all origins.
--allowed-methods      Specifies the allowed methods. It expects a JSON list of strings, with the default value being ``["*"]``, allowing all methods.
--allowed-headers      Specifies the allowed headers. It expects a JSON list of strings, with the default value being ``["*"]``, allowing all headers.

You can access ``http://127.0.0.1:PORT/docs`` (replace ``PORT`` with the port number you specified) to see the list of
supported endpoints.

API Endpoints
-------------

The REST API provides the following endpoints:

.. http:get:: /v1/models

------------------------------------------------

   Get a list of models available for MLC-LLM.

**Example**

.. code:: bash

   import requests

   url = "http://127.0.0.1:8000/v1/models"
   headers = {"accept": "application/json"}

   response = requests.get(url, headers=headers)

   if response.status_code == 200:
      print("Response:")
      print(response.json())
   else:
      print("Error:", response.status_code)


.. http:post:: /v1/chat/completions

------------------------------------------------

   Get a response from MLC-LLM using a prompt, either with or without streaming.

**Chat Completion Request Object**

- **messages** (*List[ChatCompletionMessage]*, required): A sequence of messages that have been exchanged in the conversation so far. Each message in the conversation is represented by a `ChatCompletionMessage` object, which includes the following fields:
    - **content** (*Optional[Union[str, List[Dict[str, str]]]]*): The text content of the message or structured data in case of tool-generated messages.
    - **role** (*Literal["system", "user", "assistant", "tool"]*): The role of the message sender, indicating whether the message is from the system, user, assistant, or a tool.
    - **name** (*Optional[str]*): An optional name for the sender of the message.
    - **tool_calls** (*Optional[List[ChatToolCall]]*): A list of calls to external tools or functions made within this message, applicable when the role is `tool`.
    - **tool_call_id** (*Optional[str]*): A unique identifier for the tool call, relevant when integrating external tools or services.

- **model** (*str*, required): The model to be used for generating responses.

- **frequency_penalty** (*float*, optional, default=0.0): Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model’s likelihood to repeat tokens.

- **presence_penalty** (*float*, optional, default=0.0): Positive values penalize new tokens if they are already present in the text so far, decreasing the model’s likelihood to repeat tokens.

- **logprobs** (*bool*, optional, default=False): Indicates whether to include log probabilities for each token in the response.

- **top_logprobs** (*int*, optional, default=0): An integer ranging from 0 to 20. It determines the number of tokens, most likely to appear at each position, to be returned. Each token is accompanied by a log probability. If this parameter is used, 'logprobs' must be set to true.

- **logit_bias** (*Optional[Dict[int, float]]*): Allows specifying biases for or against specific tokens during generation.

- **max_tokens** (*Optional[int]*): The maximum number of tokens to generate in the response(s).

- **n** (*int*, optional, default=1): Number of responses to generate for the given prompt.

- **seed** (*Optional[int]*): A seed for deterministic generation. Using the same seed and inputs will produce the same output.

- **stop** (*Optional[Union[str, List[str]]]*): One or more strings that, if encountered, will cause generation to stop.

- **stream** (*bool*, optional, default=False): If `True`, responses are streamed back as they are generated.

- **temperature** (*float*, optional, default=1.0): Controls the randomness of the generation. Lower values lead to less random completions.

- **top_p** (*float*, optional, default=1.0): Nucleus sampling parameter that controls the diversity of the generated responses.

- **tools** (*Optional[List[ChatTool]]*): Specifies external tools or functions that can be called as part of the chat.

- **tool_choice** (*Optional[Union[Literal["none", "auto"], Dict]]*): Controls how tools are selected for use in responses.

- **user** (*Optional[str]*): An optional identifier for the user initiating the request.

- **response_format** (*RequestResponseFormat*, optional): Specifies the format of the response. Can be either "text" or "json_object", with optional schema definition for JSON responses.

**Returns**

- If `stream` is `False`, a `ChatCompletionResponse` object containing the generated response(s).
- If `stream` is `True`, a stream of `ChatCompletionStreamResponse` objects, providing a real-time feed of generated responses.


**ChatCompletionResponseChoice**

- **finish_reason** (*Optional[Literal["stop", "length", "tool_calls", "error"]]*, optional): The reason the completion process was terminated. It can be due to reaching a stop condition, the maximum length, output of tool calls, or an error.

- **index** (*int*, required, default=0): Indicates the position of this choice within the list of choices.

- **message** (*ChatCompletionMessage*, required): The message part of the chat completion, containing the content of the chat response.

- **logprobs** (*Optional[LogProbs]*, optional): Optionally includes log probabilities for each output token

**ChatCompletionStreamResponseChoice**

- **finish_reason** (*Optional[Literal["stop", "length", "tool_calls"]]*, optional): Specifies why the streaming completion process ended. Valid reasons are "stop", "length", and "tool_calls".

- **index** (*int*, required, default=0): Indicates the position of this choice within the list of choices.

- **delta** (*ChatCompletionMessage*, required): Represents the incremental update or addition to the chat completion message in the stream.

- **logprobs** (*Optional[LogProbs]*, optional): Optionally includes log probabilities for each output token

**ChatCompletionResponse**

- **id** (*str*, required): A unique identifier for the chat completion session.

- **choices** (*List[ChatCompletionResponseChoice]*, required): A collection of `ChatCompletionResponseChoice` objects, representing the potential responses generated by the model.

- **created** (*int*, required, default=current time): The UNIX timestamp representing when the response was generated.

- **model** (*str*, required): The name of the model used to generate the chat completions.

- **system_fingerprint** (*str*, required): A system-generated fingerprint that uniquely identifies the computational environment.

- **object** (*Literal["chat.completion"]*, required, default="chat.completion"): A string literal indicating the type of object, here always "chat.completion".

- **usage** (*UsageInfo*, required, default=empty `UsageInfo` object): Contains information about the API usage for this specific request.

**ChatCompletionStreamResponse**

- **id** (*str*, required): A unique identifier for the streaming chat completion session.

- **choices** (*List[ChatCompletionStreamResponseChoice]*, required): A list of `ChatCompletionStreamResponseChoice` objects, each representing a part of the streaming chat response.

- **created** (*int*, required, default=current time): The creation time of the streaming response, represented as a UNIX timestamp.

- **model** (*str*, required): Specifies the model that was used for generating the streaming chat completions.

- **system_fingerprint** (*str*, required): A unique identifier for the system generating the streaming completions.

- **object** (*Literal["chat.completion.chunk"]*, required, default="chat.completion.chunk"): A literal indicating that this object represents a chunk of a streaming chat completion.

------------------------------------------------


**Example**

Below is an example of using the API to interact with MLC-LLM in Python with Streaming.

.. code:: bash

   import requests
   import json

   # Get a response using a prompt with streaming
   payload = {
    "model": "./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": True,
   }
   with requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload, stream=True) as r:
      for chunk in r.iter_content(chunk_size=None):
         chunk = chunk.decode("utf-8")
         if "[DONE]" in chunk[6:]:
            break
         response = json.loads(chunk[6:])
         content = response["choices"][0]["delta"].get("content", "")
         print(content, end="", flush=True)
   print("\n")

------------------------------------------------

There is also support for function calling similar to OpenAI (https://platform.openai.com/docs/guides/function-calling). Below is an example on how to use function calling in Python.

.. code:: bash

   import requests
   import json

   tools = [
      {
         "type": "function",
         "function": {
               "name": "get_current_weather",
               "description": "Get the current weather in a given location",
               "parameters": {
                  "type": "object",
                  "properties": {
                     "location": {
                           "type": "string",
                           "description": "The city and state, e.g. San Francisco, CA",
                     },
                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                  },
                  "required": ["location"],
               },
         },
      }
   ]

   payload = {
      "model": "./dist/gorilla-openfunctions-v1-q4f16_1-MLC/",
      "messages": [
         {
               "role": "user",
               "content": "What is the current weather in Pittsburgh, PA in fahrenheit?",
         }
      ],
      "stream": False,
      "tools": tools,
   }

   r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload)
   print(f"{r.json()['choices'][0]['message']['tool_calls'][0]['function']}\n")

   # Output: {'name': 'get_current_weather', 'arguments': {'location': 'Pittsburgh, PA', 'unit': 'fahrenheit'}}

------------------------------------------------

Function Calling with streaming is also supported. Below is an example on how to use function calling with streaming in Python.

.. code:: bash

   import requests
   import json

   tools = [
      {
         "type": "function",
         "function": {
               "name": "get_current_weather",
               "description": "Get the current weather in a given location",
               "parameters": {
                  "type": "object",
                  "properties": {
                     "location": {
                           "type": "string",
                           "description": "The city and state, e.g. San Francisco, CA",
                     },
                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                  },
                  "required": ["location"],
               },
         },
      }
   ]

   payload = {
      "model": "./dist/gorilla-openfunctions-v1-q4f16_1-MLC/",
      "messages": [
         {
               "role": "user",
               "content": "What is the current weather in Pittsburgh, PA and Tokyo, JP in fahrenheit?",
         }
      ],
      "stream": True,
      "tools": tools,
   }

   with requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload, stream=True) as r:
    for chunk in r.iter_content(chunk_size=None):
        chunk = chunk.decode("utf-8")
        if "[DONE]" in chunk[6:]:
            break
        response = json.loads(chunk[6:])
        content = response["choices"][0]["delta"].get("content", "")
        print(f"{content}", end="", flush=True)
   print("\n")

   # Output: ["get_current_weather(location='Pittsburgh,PA',unit='fahrenheit')", "get_current_weather(location='Tokyo,JP',unit='fahrenheit')"]


.. note::
   The API is a uniform interface that supports multiple languages. You can also utilize these functionalities in languages other than Python.
