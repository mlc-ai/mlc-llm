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

   Get a completion from MLC-Chat using a prompt.

.. http:get:: /v1/chat/completions

   Get a response from MLC-Chat using a prompt, either with or without streaming.

.. http:get:: /chat/reset

   Reset the chat.

.. http:get:: /stats

   Get the latest runtime stats (encode/decode speed).

.. http:get:: /verbose_stats

   Get the verbose runtime stats (encode/decode speed, total runtime).

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
