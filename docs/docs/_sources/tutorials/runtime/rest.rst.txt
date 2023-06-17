Use MLC-Chat with REST APIs
===========================

.. contents:: Table of Contents
   :local:
   :depth: 2

We provide `REST API <https://www.ibm.com/topics/rest-apis#:~:text=the%20next%20step-,What%20is%20a%20REST%20API%3F,representational%20state%20transfer%20architectural%20style.>`_
for user to interact with MLC-Chat in their own programs.

Install MLC-Chat Package
------------------------

The REST API is a part of the MLC-Chat package, which we have prepared pre-built pip wheels and you can install it by
following the instructions in `<https://mlc.ai/package/>`_.

Verify Installation
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   python -m mlc_chat.rest --help

You are expected to see the help information of the REST API.

Launch the Server
-----------------

To launch the REST server for MLC-Chat, run the following command in your terminal.

.. code:: bash

   python -m mlc_chat.rest --device-name DEVICE [--quantization QUANTIZATION_MODE] [--device-id DEVICE_ID] [--port PORT] [--artifact-path ARTIFACT_PATH]


--device-name          The device name to run the model. Available options are:
                       ``metal``, ``cuda``, ``vulkan``, ``cpu``.
--device-id            The device id to run the model. The default value is ``0``.
--quantization-mode    The code indicating the quantization mode to run the model. See :ref:`quantization_mode` for more details.
--artifact-path        The path to the artifact folder where models are stored. The default value is ``dist``.
--port                 The port to run the server. The default value is ``8000``.

You can access ``http://127.0.0.1:PORT/docs`` (replace ``PORT`` with the port number you specified) to see the list of
supported endpoints.

API Endpoints
-------------

The REST API provides the following endpoints:

.. http:get:: /v1/chat/completions

   Get a response from MLC-Chat using a prompt.

.. http:get:: /chat/reset

   Reset the chat.

.. http:get:: /stats

   Get the latest runtime stats (encode/decode speed).


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

Please check `example folder <https://github.com/mlc-ai/mlc-llm/examples/rest>`__ for more examples using REST API.

.. note:: 
   The REST API is a uniform interface that supports multiple languages. You can also utilize the REST API in languages other than Python.
