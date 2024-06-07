.. _deploy-python-engine:

Python API
==========

.. note::
  This page introduces the Python API with MLCEngine in MLC LLM.

.. contents:: Table of Contents
  :local:
  :depth: 2


MLC LLM provides Python API through classes :class:`mlc_llm.MLCEngine` and :class:`mlc_llm.AsyncMLCEngine`
which **support full OpenAI API completeness** for easy integration into other Python projects.

This page introduces how to use the engines in MLC LLM.
The Python API is a part of the MLC-LLM package, which we have prepared pre-built pip wheels via
the :ref:`installation page <install-mlc-packages>`.


Verify Installation
-------------------

.. code:: bash

  python -c "from mlc_llm import MLCEngine; print(MLCEngine)"

You are expected to see the output of ``<class 'mlc_llm.serve.engine.MLCEngine'>``.

If the command above results in error, follow :ref:`install-mlc-packages` to install prebuilt pip
packages or build MLC LLM from source.


Run MLCEngine
-------------

:class:`mlc_llm.MLCEngine` provides the interface of OpenAI chat completion synchronously.
:class:`mlc_llm.MLCEngine` does not batch concurrent request due to the synchronous design,
and please use :ref:`AsyncMLCEngine <python-engine-async-llm-engine>` for request batching process.

**Stream Response.** In :ref:`quick-start` and :ref:`introduction-to-mlc-llm`,
we introduced the basic use of :class:`mlc_llm.MLCEngine`.

.. code:: python

  from mlc_llm import MLCEngine

  # Create engine
  model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
  engine = MLCEngine(model)

  # Run chat completion in OpenAI API.
  for response in engine.chat.completions.create(
      messages=[{"role": "user", "content": "What is the meaning of life?"}],
      model=model,
      stream=True,
  ):
      for choice in response.choices:
          print(choice.delta.content, end="", flush=True)
  print("\n")

  engine.terminate()

This code example first creates an :class:`mlc_llm.MLCEngine` instance with the 8B Llama-3 model.
**We design the Python API** :class:`mlc_llm.MLCEngine` **to align with OpenAI API**,
which means you can use :class:`mlc_llm.MLCEngine` in the same way of using
`OpenAI's Python package <https://github.com/openai/openai-python?tab=readme-ov-file#usage>`_
for both synchronous and asynchronous generation.

**Non-stream Response.** The code example above uses the synchronous chat completion
interface and iterate over all the stream responses.
If you want to run without streaming, you can run

.. code:: python

  response = engine.chat.completions.create(
      messages=[{"role": "user", "content": "What is the meaning of life?"}],
      model=model,
      stream=False,
  )
  print(response)

Please refer to `OpenAI's Python package <https://github.com/openai/openai-python?tab=readme-ov-file#usage>`_
and `OpenAI chat completion API <https://platform.openai.com/docs/api-reference/chat/create>`_
for the complete chat completion interface.

.. note::

  If you want to enable tensor parallelism to run LLMs on multiple GPUs,
  please specify argument ``model_config_overrides`` in MLCEngine constructor.
  For example,

  .. code:: python

    from mlc_llm import MLCEngine
    from mlc_llm.serve.config import EngineConfig

    model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
    engine = MLCEngine(
        model,
        engine_config=EngineConfig(tensor_parallel_shards=2),
    )


.. _python-engine-async-llm-engine:

Run AsyncMLCEngine
------------------

:class:`mlc_llm.AsyncMLCEngine` provides the interface of OpenAI chat completion with
asynchronous features.
**We recommend using** :class:`mlc_llm.AsyncMLCEngine` **to batch concurrent request for better throughput.**

**Stream Response.** The core use of :class:`mlc_llm.AsyncMLCEngine` for stream responses is as follows.

.. code:: python

  async for response in await engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    model=model,
    stream=True,
  ):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)

.. collapse:: The collapsed is a complete runnable example of AsyncMLCEngine in Python.

  .. code:: python

    import asyncio
    from typing import Dict

    from mlc_llm.serve import AsyncMLCEngine

    model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
    prompts = [
        "Write a three-day travel plan to Pittsburgh.",
        "What is the meaning of life?",
    ]


    async def test_completion():
        # Create engine
        async_engine = AsyncMLCEngine(model=model)

        num_requests = len(prompts)
        output_texts: Dict[str, str] = {}

        async def generate_task(prompt: str):
            async for response in await async_engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                stream=True,
            ):
                if response.id not in output_texts:
                    output_texts[response.id] = ""
                output_texts[response.id] += response.choices[0].delta.content

        tasks = [asyncio.create_task(generate_task(prompts[i])) for i in range(num_requests)]
        await asyncio.gather(*tasks)

        # Print output.
        for request_id, output in output_texts.items():
            print(f"Output of request {request_id}:\n{output}\n")

        async_engine.terminate()


    asyncio.run(test_completion())

|

**Non-stream Response.** Similarly, :class:`mlc_llm.AsyncEngine` provides the non-stream response
interface.

.. code:: python

  response = await engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    model=model,
    stream=False,
  )
  print(response)

Please refer to `OpenAI's Python package <https://github.com/openai/openai-python?tab=readme-ov-file#usage>`_
and `OpenAI chat completion API <https://platform.openai.com/docs/api-reference/chat/create>`_
for the complete chat completion interface.

.. note::

  If you want to enable tensor parallelism to run LLMs on multiple GPUs,
  please specify argument ``model_config_overrides`` in AsyncMLCEngine constructor.
  For example,

  .. code:: python

    from mlc_llm import AsyncMLCEngine
    from mlc_llm.serve.config import EngineConfig

    model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
    engine = AsyncMLCEngine(
        model,
        engine_config=EngineConfig(tensor_parallel_shards=2),
    )


Engine Mode
-----------

To ease the engine configuration, the constructors of :class:`mlc_llm.MLCEngine` and
:class:`mlc_llm.AsyncMLCEngine` have an optional argument ``mode``,
which falls into one of the three options ``"local"``, ``"interactive"`` or ``"server"``.
The default mode is ``"local"``.

Each mode denotes a pre-defined configuration of the engine to satisfy different use cases.
The choice of the mode controls the request concurrency of the engine,
as well as engine's KV cache token capacity (or in other words, the maximum
number of tokens that the engine's KV cache can hold),
and further affects the GPU memory usage of the engine.

In short,

- mode ``"local"`` uses low request concurrency and low KV cache capacity, which is suitable for cases where **concurrent requests are not too many, and the user wants to save GPU memory usage**.
- mode ``"interactive"`` uses 1 as the request concurrency and low KV cache capacity, which is designed for **interactive use cases** such as chats and conversations.
- mode ``"server"`` uses as much request concurrency and KV cache capacity as possible. This mode aims to **fully utilize the GPU memory for large server scenarios** where concurrent requests may be many.

**For system benchmark, please select mode** ``"server"``.
Please refer to :ref:`python-engine-api-reference` for detailed documentation of the engine mode.


Deploy Your Own Model with Python API
-------------------------------------

The :ref:`introduction page <introduction-deploy-your-own-model>` introduces how we can deploy our
own models with MLC LLM.
This section introduces how you can use the model weights you convert and the model library you build
in :class:`mlc_llm.MLCEngine` and :class:`mlc_llm.AsyncMLCEngine`.

We use the `Phi-2 <https://huggingface.co/microsoft/phi-2>`_ as the example model.

**Specify Model Weight Path.** Assume you have converted the model weights for your own model,
you can construct a :class:`mlc_llm.MLCEngine` as follows:

.. code:: python

  from mlc_llm import MLCEngine

  model = "models/phi-2"  # Assuming the converted phi-2 model weights are under "models/phi-2"
  engine = MLCEngine(model)


**Specify Model Library Path.** Further, if you build the model library on your own,
you can use it in :class:`mlc_llm.MLCEngine` by passing the library path through argument ``model_lib``.

.. code:: python

  from mlc_llm import MLCEngine

  model = "models/phi-2"
  model_lib = "models/phi-2/lib.so"  # Assuming the phi-2 model library is built at "models/phi-2/lib.so"
  engine = MLCEngine(model, model_lib=model_lib)


The same applies to :class:`mlc_llm.AsyncMLCEngine`.


.. _python-engine-api-reference:

API Reference
-------------

The :class:`mlc_llm.MLCEngine` and :class:`mlc_llm.AsyncMLCEngine` classes provide the following constructors.

The MLCEngine and AsyncMLCEngine have full OpenAI API completeness.
Please refer to `OpenAI's Python package <https://github.com/openai/openai-python?tab=readme-ov-file#usage>`_
and `OpenAI chat completion API <https://platform.openai.com/docs/api-reference/chat/create>`_
for the complete chat completion interface.

.. currentmodule:: mlc_llm

.. autoclass:: MLCEngine
  :members:
  :exclude-members: evaluate
  :undoc-members:
  :show-inheritance:

  .. automethod:: __init__

.. autoclass:: AsyncMLCEngine
  :members:
  :exclude-members: evaluate
  :undoc-members:
  :show-inheritance:

  .. automethod:: __init__
