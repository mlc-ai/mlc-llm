Rest API
========

.. contents:: Table of Contents
   :local:
   :depth: 2

We provide `REST API <https://www.ibm.com/topics/rest-apis#:~:text=the%20next%20step-,What%20is%20a%20REST%20API%3F,representational%20state%20transfer%20architectural%20style.>`_
for a user to interact with MLC-Chat in their own programs.

Install MLC-Chat Package
------------------------

The REST API is a part of the MLC-Chat package, which we have prepared pre-built :doc:`pip wheels <../install/mlc_llm>`.

Verify Installation
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   python -m mlc_chat.rest --help

You are expected to see the help information of the REST API.

.. _mlcchat_package_build_from_source:

Optional: Build from Source
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the prebuilt is unavailable on your platform, or you would like to build a runtime
that supports other GPU runtime than the prebuilt version. We can build a customized version
of mlc chat runtime. You only need to do this if you choose not to use the prebuilt.

First, make sure you install TVM unity (following the instruction in :ref:`install-tvm-unity`).
You can choose to only pip install `mlc-ai-nightly` that comes with the tvm unity but skip `mlc-chat-nightly`.
Then please follow the instructions in :ref:`mlcchat_build_from_source` to build the necessary libraries.

You can now use ``mlc_chat`` package by including the `python` directory to ``PYTHONPATH`` environment variable.

.. code:: bash

   PYTHONPATH=python python -m mlc_chat.rest --help

Launch the Server
-----------------

To launch the REST server for MLC-Chat, run the following command in your terminal.

.. code:: bash

   python -m mlc_chat.rest --model MODEL [--lib-path LIB_PATH] [--device DEVICE] [--host HOST] [--port PORT]

--model                The model folder after compiling with MLC-LLM build process. The parameter
                       can either be the model name with its quantization scheme
                       (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
                       folder. In the former case, we will use the provided name to search
                       for the model folder over possible paths.
--lib-path             An optional field to specify the full path to the model library file to use (e.g. a ``.so`` file).
--device               The description of the device to run on. User should provide a string in the
                       form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
                       'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
                       local device), and 'device_id' is the device id to run on. The default value is ``auto``,
                       with the device id set to 0 for default.
--host                 The host at which the server should be started, defaults to ``127.0.0.1``.
--port                 The port on which the server should be started, defaults to ``8000``.

You can access ``http://127.0.0.1:PORT/docs`` (replace ``PORT`` with the port number you specified) to see the list of
supported endpoints.

API Endpoints
-------------

The REST API provides the following endpoints:

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
      
------------------------------------------------


Use REST API in your own program
--------------------------------

Once you have launched the REST server, you can use the REST API in your own program. Below is an example of using REST API to interact with MLC-Chat in Python (suppose the server is running on ``http://127.0.0.1:8000/``):

.. code:: bash

   import requests
   import json

   # Get a response using a prompt without streaming
   payload = {
      "model": "vicuna-v1-7b",
      "messages": [{"role": "user", "content": "Write a haiku"}],
      "stream": False
   }
   r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
   print(f"Without streaming:\n{r.json()['choices'][0]['message']['content']}\n")

   # Reset the chat
   r = requests.post("http://127.0.0.1:8000/chat/reset", json=payload)
   print(f"Reset chat: {str(r)}\n")

   # Get a response using a prompt with streaming
   payload = {
      "model": "vicuna-v1-7b",
      "messages": [{"role": "user", "content": "Write a haiku"}],
      "stream": True
   }
   with requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, stream=True) as r:
      print(f"With streaming:")
      for chunk in r:
         content = json.loads(chunk[6:-2])["choices"][0]["delta"].get("content", "")
         print(f"{content}", end="", flush=True)
      print("\n")

   # Get the latest runtime stats
   r = requests.get("http://127.0.0.1:8000/stats")
   print(f"Runtime stats: {r.json()}\n")

Please check `example folder <https://github.com/mlc-ai/mlc-llm/tree/main/examples/rest>`__ for more examples using REST API.

.. note::
   The REST API is a uniform interface that supports multiple languages. You can also utilize the REST API in languages other than Python.
