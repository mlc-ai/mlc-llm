Rest API
========

.. contents:: Table of Contents
   :local:
   :depth: 2

We provide `REST API <https://www.ibm.com/topics/rest-apis#:~:text=the%20next%20step-,What%20is%20a%20REST%20API%3F,representational%20state%20transfer%20architectural%20style.>`_
for a user to interact with MLC-Chat in their own programs.

Install MLC-Chat Package
------------------------

SERVE is a part of the MLC-Chat package, installation instruction for which we be found here :doc:`<../install/mlc_llm>`.

Verify Installation
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   python -m mlc_llm.serve.server --help

You are expected to see the help information of the MLC SERVE.

.. _mlcchat_package_build_from_source:


Launch the Server
-----------------

To launch the MLC Server for MLC-Chat, run the following command in your terminal.

.. code:: bash

   python -m mlc_llm.serve.server --model MODEL --model-lib-path MODEL_LIB_PATH [--device DEVICE] [--max-batch-size MAX_BATCH_SIZE] [--max-total-seq-length MAX_TOTAL_SEQ_LENGTH] [--prefill-chunk-size PREFILL_CHUNK_SIZE] [--enable-tracing] [--host HOST] [--port PORT] [--allow-credentials] [--allowed-origins ALLOWED_ORIGINS] [--allowed-methods ALLOWED_METHODS] [--allowed-headers ALLOWED_HEADERS]

--model                The model folder after compiling with MLC-LLM build process. The parameter
                       can either be the model name with its quantization scheme
                       (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
                       folder. In the former case, we will use the provided name to search
                       for the model folder over possible paths.
--model-lib-path       A field to specify the full path to the model library file to use (e.g. a ``.so`` file).
--device               The description of the device to run on. User should provide a string in the
                       form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
                       'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
                       local device), and 'device_id' is the device id to run on. The default value is ``auto``,
                       with the device id set to 0 for default.
--host                 The host at which the server should be started, defaults to ``127.0.0.1``.
--port                 The port on which the server should be started, defaults to ``8000``.
--allow-credentials    A flag to indicate whether the server should allow credentials. If set, the server will
                       include the ``CORS`` header in the response
--allowed-origins      Specifies the allowed origins. It expects a JSON list of strings, with the default value being ``["*"]``, allowing all origins.
--allowed-methods      Specifies the allowed methods. It expects a JSON list of strings, with the default value being ``["*"]``, allowing all methods.
--allowed-headers      Specifies the allowed headers. It expects a JSON list of strings, with the default value being ``["*"]``, allowing all headers.
--max-batch-size       The maximum batch size for processing.
--max-total-seq-length   The maximum total number of tokens whose KV data are allowed to exist in the KV cache at any time. Set it to None to enable automatic computation of the max total sequence length.
--prefill-chunk-size   The maximum total sequence length in a prefill. If not specified, it will be automatically inferred from model config.
--enable-tracing       A boolean indicating if to enable event logging for requests.

You can access ``http://127.0.0.1:PORT/docs`` (replace ``PORT`` with the port number you specified) to see the list of
supported endpoints.

API Endpoints
-------------

The REST API provides the following endpoints:

.. http:get:: /v1/models

------------------------------------------------

   Get a list of models available for MLC-Chat.

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


.. http:get:: /v1/completions

------------------------------------------------

   Get a completion from MLC-Chat using a prompt.

**Request body**

**model**: *str* (required)
   The model folder after compiling with MLC-LLM build process. The parameter
   can either be the model name with its quantization scheme
   (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
   folder. In the former case, we will use the provided name to search
   for the model folder over possible paths.
**prompt**: *str* (required)
   A list of chat messages. The last message should be from the user.
**stream**: *bool* (optional)
   Whether to stream the response. If ``True``, the response will be streamed
   as the model generates the response. If ``False``, the response will be
   returned after the model finishes generating the response.
**temperature**: *float* (optional)
   The temperature applied to logits before sampling. The default value is
   ``0.7``. A higher temperature encourages more diverse outputs, while a
   lower temperature produces more deterministic outputs.
**top_p**: *float* (optional)
   This parameter determines the set of tokens from which we sample during
   decoding. The default value is set to ``0.95``. At each step, we select
   tokens from the minimal set that has a cumulative probability exceeding
   the ``top_p`` parameter.

   For additional information on top-p sampling, please refer to this blog
   post: https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling.
**repetition_penalty**: *float* (optional)
   The repetition penalty controls the likelihood of the model generating
   repeated texts. The default value is set to ``1.0``, indicating that no
   repetition penalty is applied. Increasing the value reduces the
   likelihood of repeat text generation. However, setting a high
   ``repetition_penalty`` may result in the model generating meaningless
   texts. The ideal choice of repetition penalty may vary among models.

   For more details on how repetition penalty controls text generation, please
   check out the CTRL paper (https://arxiv.org/pdf/1909.05858.pdf).
**presence_penalty**: *float* (optional)
   Positive values penalize new tokens if they are already present in the text so far,
   decreasing the model's likelihood to repeat tokens.
**frequency_penalty**: *float* (optional)
   Positive values penalize new tokens based on their existing frequency in the text so far,
   decreasing the model's likelihood to repeat tokens.
**mean_gen_len**: *int* (optional)
   The approximated average number of generated tokens in each round. Used
   to determine whether the maximum window size would be exceeded.
**max_gen_len**: *int* (optional)
   This parameter determines the maximum length of the generated text. If it is
   not set, the model will generate text until it encounters a stop token.

------------------------------------------------

**Returns**
   If ``stream`` is set to ``False``, the response will be a ``CompletionResponse`` object.
   If ``stream`` is set to ``True``, the response will be a stream of ``CompletionStreamResponse`` objects.


.. http:get:: /v1/chat/completions

------------------------------------------------

   Get a response from MLC-Chat using a prompt, either with or without streaming.

**Request body**

**model**: *str* (required)
   The model folder after compiling with MLC-LLM build process. The parameter
   can either be the model name with its quantization scheme
   (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
   folder. In the former case, we will use the provided name to search
   for the model folder over possible paths.
**messages**: *list[ChatMessage]* (required)
   A list of chat messages. The last message should be from the user.
**stream**: *bool* (optional)
   Whether to stream the response. If ``True``, the response will be streamed
   as the model generates the response. If ``False``, the response will be
   returned after the model finishes generating the response.
**temperature**: *float* (optional)
   The temperature applied to logits before sampling. The default value is
   ``0.7``. A higher temperature encourages more diverse outputs, while a
   lower temperature produces more deterministic outputs.
**top_p**: *float* (optional)
   This parameter determines the set of tokens from which we sample during
   decoding. The default value is set to ``0.95``. At each step, we select
   tokens from the minimal set that has a cumulative probability exceeding
   the ``top_p`` parameter.

   For additional information on top-p sampling, please refer to this blog
   post: https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling.
**repetition_penalty**: *float* (optional)
   The repetition penalty controls the likelihood of the model generating
   repeated texts. The default value is set to ``1.0``, indicating that no
   repetition penalty is applied. Increasing the value reduces the
   likelihood of repeat text generation. However, setting a high
   ``repetition_penalty`` may result in the model generating meaningless
   texts. The ideal choice of repetition penalty may vary among models.

   For more details on how repetition penalty controls text generation, please
   check out the CTRL paper (https://arxiv.org/pdf/1909.05858.pdf).
**presence_penalty**: *float* (optional)
   Positive values penalize new tokens if they are already present in the text so far,
   decreasing the model's likelihood to repeat tokens.
**frequency_penalty**: *float* (optional)
   Positive values penalize new tokens based on their existing frequency in the text so far,
   decreasing the model's likelihood to repeat tokens.
**mean_gen_len**: *int* (optional)
   The approximated average number of generated tokens in each round. Used
   to determine whether the maximum window size would be exceeded.
**max_gen_len**: *int* (optional)
   This parameter determines the maximum length of the generated text. If it is
   not set, the model will generate text until it encounters a stop token.
**n**: *int* (optional)
   This parameter determines the number of text samples to generate. The default
   value is ``1``. Note that this parameter is only used when ``stream`` is set to
   ``False``.
**stop**: *str* or *list[str]* (optional)
   When ``stop`` is encountered, the model will stop generating output.
   It can be a string or a list of strings. If it is a list of strings, the model
   will stop generating output when any of the strings in the list is encountered.
   Note that this parameter does not override the default stop string of the model.

------------------------------------------------

**Returns**
   If ``stream`` is set to ``False``, the response will be a ``ChatCompletionResponse`` object.
   If ``stream`` is set to ``True``, the response will be a stream of ``ChatCompletionStreamResponse`` objects.

**Example**

Once you have launched the Server, you can use the API in your own program. Below is an example of using the API to interact with MLC-Chat in Python without Streaming (suppose the server is running on ``http://127.0.0.1:8080/``):

.. code:: bash

   import requests

   # Get a response using a prompt without streaming
   payload = {
      "model": "./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/",
      "messages": [
         {"role": "user", "content": "Hello! Our project is MLC LLM."},
         {
               "role": "assistant",
               "content": "Hello! It's great to hear about your project, MLC LLM.",
         },
         {"role": "user", "content": "What is the name of our project?"},
      ],
      "stream": False,
      # "n": 1,
      "max_tokens": 300,
   }
   r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload)
   choices = r.json()["choices"]
   for choice in choices:
      print(f"{choice['message']['content']}\n")


Below is an example of using the API to interact with MLC-Chat in Python with Streaming.

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


.. http:get:: /chat/reset

   Reset the chat.

.. http:get:: /stats

   Get the latest runtime stats (encode/decode speed).

.. http:get:: /verbose_stats

   Get the verbose runtime stats (encode/decode speed, total runtime).


Request Objects
---------------

**ChatMessage**

**role**: *str* (required)
   The role(author) of the message. It can be either ``user`` or ``assistant``.
**content**: *str* (required)
   The content of the message.
**name**: *str* (optional)
   The name of the author of the message.

Response Objects
----------------

**CompletionResponse**

**id**: *str*
   The id of the completion.
**object**: *str*
   The object name ``text.completion``.
**created**: *int*
   The time when the completion is created.
**choices**: *list[CompletionResponseChoice]*
   A list of choices generated by the model.
**usage**: *UsageInfo* or *None*
   The usage information of the model.

------------------------------------------------

**CompletionResponseChoice**

**index**: *int*
   The index of the choice.
**text**: *str*
   The message generated by the model.
**finish_reason**: *str*
   The reason why the model finishes generating the message. It can be either
   ``stop`` or ``length``.


------------------------------------------------

**CompletionStreamResponse**

**id**: *str*
   The id of the completion.
**object**: *str*
   The object name ``text.completion.chunk``.
**created**: *int*
   The time when the completion is created.
**choices**: *list[ChatCompletionResponseStreamhoice]*
   A list of choices generated by the model.

------------------------------------------------

**ChatCompletionResponseStreamChoice**

**index**: *int*
   The index of the choice.
**text**: *str*
   The message generated by the model.
**finish_reason**: *str*
   The reason why the model finishes generating the message. It can be either
   ``stop`` or ``length``.

------------------------------------------------

**ChatCompletionResponse**

**id**: *str*
   The id of the completion.
**object**: *str*
   The object name ``chat.completion``.
**created**: *int*
   The time when the completion is created.
**choices**: *list[ChatCompletionResponseChoice]*
   A list of choices generated by the model.
**usage**: *UsageInfo* or *None*
   The usage information of the model.

------------------------------------------------

**ChatCompletionResponseChoice**

**index**: *int*
   The index of the choice.
**message**: *ChatMessage*
   The message generated by the model.
**finish_reason**: *str*
   The reason why the model finishes generating the message. It can be either
   ``stop`` or ``length``.

------------------------------------------------

**ChatCompletionStreamResponse**

**id**: *str*
   The id of the completion.
**object**: *str*
   The object name ``chat.completion.chunk``.
**created**: *int*
   The time when the completion is created.
**choices**: *list[ChatCompletionResponseStreamhoice]*
   A list of choices generated by the model.

------------------------------------------------

**ChatCompletionResponseStreamChoice**

**index**: *int*
   The index of the choice.
**delta**: *DeltaMessage*
   The delta message generated by the model.
**finish_reason**: *str*
   The reason why the model finishes generating the message. It can be either
   ``stop`` or ``length``.

------------------------------------------------


**DeltaMessage**

**role**: *str*
   The role(author) of the message. It can be either ``user`` or ``assistant``.
**content**: *str*
   The content of the message.

