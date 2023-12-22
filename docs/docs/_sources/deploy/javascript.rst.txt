.. _webllm-runtime:

WebLLM and Javascript API
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2

`WebLLM <https://www.npmjs.com/package/@mlc-ai/web-llm>`_ is an MLC chat web runtime
that allows you to build chat applications directly in the browser, leveraging
`WebGPU <https://www.w3.org/TR/webgpu/>`_ and providing users a natural layer of abstraction.

Try out the Prebuilt Webpage
----------------------------

To get started, you can try out `WebLLM prebuilt webpage <https://webllm.mlc.ai/#chat-demo>`__.

A WebGPU-compatible browser and a local GPU are needed to run WebLLM.
You can download the latest Google Chrome and use `WebGPU Report <https://webgpureport.org/>`__
to verify the functionality of WebGPU on your browser.


Use WebLLM NPM Package
----------------------

WebLLM is available as an `npm package <https://www.npmjs.com/package/@mlc-ai/web-llm>`_.
The source code is available in `the WebLLM repo <https://github.com/mlc-ai/web-llm>`_,
where you can make your own modifications and build from source.

Note that the `WebLLM prebuilt webpage <https://webllm.mlc.ai/#chat-demo>`__ above
is powered by the WebLLM npm package, specifically with the code in
the `simple-chat <https://github.com/mlc-ai/web-llm/tree/main/examples/simple-chat>`__ example.

Each of the model in the  `WebLLM prebuilt webpage <https://webllm.mlc.ai/#chat-demo>`__
is registered as an instance of ``ModelRecord``. Looking at the most straightforward example 
`get-started <https://github.com/mlc-ai/web-llm/blob/main/examples/get-started/src/get_started.ts>`__,
we see the code snippet:

.. code:: typescript

  const myAppConfig: AppConfig = {
    model_list: [
      {
        "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-hf-q4f32_1-MLC/resolve/main/",
        "local_id": "Llama-2-7b-chat-hf-q4f32_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-MLC-webgpu.wasm",
      },
      {
        "model_url": "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/resolve/main/",
        "local_id": "Mistral-7B-Instruct-v0.2-q4f16_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-MLC-webgpu.wasm",
        "required_features": ["shader-f16"],
      },
      // Add your own models here...
    ]
  }
  const selectedModel = "Llama-2-7b-chat-hf-q4f32_1"
  // const selectedModel = "Mistral-7B-Instruct-v0.1-q4f16_1"
  await chat.reload(selectedModel, undefined, myAppConfig);

Just like any other platforms, to run a model with on WebLLM, you need:

1. Model weights converted to MLC format (e.g. `Llama-2-7b-hf-q4f32_1-MLC 
   <https://huggingface.co/mlc-ai/Llama-2-7b-hf-q4f32_1-MLC/tree/main>`_.): downloaded through ``model_url``
2. Model library that comprises the inference logic (see repo `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__): downloaded through ``model_lib_url``.

.. _webllm-add-model-variant:

Bring Your Own Model Variant
----------------------------

In cases where the model you are adding is simply a variant of an existing
model, we only need to convert weights and reuse existing model library. For instance:

- Adding ``OpenMistral`` when MLC supports ``Mistral``
- Adding ``Llama2-uncensored`` when MLC supports ``Llama2``


In this section, we walk you through adding ``WizardMath-7B-V1.1-q4f16_1`` to the 
`get-started <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started>`__ example.
According to the model's ``config.json`` on `its Huggingface repo <https://huggingface.co/WizardLM/WizardMath-7B-V1.1/blob/main/config.json>`_,
it reuses the Mistral model architecture.

.. note:: 

  Before proceeding, make sure you have followed the instruction
  in `the WebLLM README <https://github.com/mlc-ai/web-llm?tab=readme-ov-file#web-llm>`_
  so that your environment is ready. You do not need to build from source unless
  you added your own conversation template in step 2.


**Step 1 Convert weights**

Please follow :ref:`convert-weights-via-MLC` for the instruction. 
Note that the weights are shared across all platforms in MLC.

After successfully following all steps in the page (including :ref:`distribute-compiled-models`),
you should end up with a Huggingface repo similar to 
`WizardMath-7B-V1.1-q4f16_1-MLC <https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC>`__,
which includes the converted/quantized weights, the ``mlc-chat-config.json``, and tokenizer files.

**Step 2 Conversation Template**

Depending on whether the model variant you add needs a conversation template different
from the ones MLC provides, you may need to add it to ``/path/to/web-llm/src/conversation.ts``.
We look up the template to use with the ``conv_template`` field in ``mlc-chat-config.json``.

For more details, please see :ref:`configure-mlc-chat-json`.

.. note:: 

  If you added your conversation template in ``src/conversation.ts``, you need to build from source
  following the instruction in `the WebLLM repo's README <https://github.com/mlc-ai/web-llm?tab=readme-ov-file#build-webllm-package-from-source>`_. 

  Alternatively, you could use the ``"custom"`` conversation template so that you can pass in
  your own ``ConvTemplateConfig`` in runtime without having to build the package from source.

**Step 3 Adding as ModelRecord**

Finally, we modify the code snippet for
`get-started <https://github.com/mlc-ai/web-llm/blob/main/examples/get-started/src/get_started.ts>`__
pasted above.

We simply specify the Huggingface link as ``model_url``, while reusing the ``model_lib_url`` for 
``Mistral-7B``. Note that we need the suffix to be ``/resolve/main/``.

.. code:: typescript

  const myAppConfig: AppConfig = {
    model_list: [
      // Other records here omitted...
      {
        "model_url": "https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC/resolve/main/",
        "local_id": "WizardMath-7B-V1.1-q4f16_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-MLC-webgpu.wasm",
        "required_features": ["shader-f16"],
      },
    ]
  }

  const selectedModel = "WizardMath-7B-V1.1-q4f16_1"
  await chat.reload(selectedModel, undefined, myAppConfig);

Now, running the ``get-started`` example will use the ``WizardMath`` model you just added.


Bring Your Own Model Library
----------------------------

A model library is specified by:

 - The model architecture (e.g. ``llama-2``, ``gpt-neox``)
 - Quantization (e.g. ``q4f16_1``, ``q0f32``)
 - Metadata (e.g. ``context_window_size``, ``sliding_window_size``, ``prefill-chunk-size``), which affects memory planning
 - Platform (e.g. ``cuda``, ``webgpu``, ``iOS``)

In cases where the model you want to run is not compatible with the provided MLC
prebuilt model libraries (e.g. having a different quantization, a different
metadata spec, or even a different model architecture), you need to build your
own model library.

Follow the page :ref:`compile-model-libraries`, specifically the ``WebGPU`` option. The page
guides you through compiling ``RedPajama-INCITE-Instruct-3B-v1`` with ``q4f16_1``.
After successfully following the page, you should end up with:

.. code:: shell

    ~/mlc-llm > ls dist/rp_q4f16_1
      rp_q4f16_1.wasm                                  # ===> the model library
      params                                           # ===> containing the model weights, tokenizer and chat config

Upload the ``rp_q4f16_1.wasm`` to a github repository (for us,
it is in `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__). Then
upload the ``rp_q4f16_1/params`` to a Huggingface repo just like we do in :ref:`distribute-compiled-models`.

Finally, we are able to run the model we added in WebLLM's `get-started <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started>`__:

.. code:: typescript

  const myAppConfig: AppConfig = {
    model_list: [
      // Other records here omitted...
      {
        "model_url": "https://huggingface.co/my-hf-account/my-rp_q4f16_1-weight-hf-repo/resolve/main/",
        "local_id": "RedPajama-INCITE-Instruct-3B-v1",
        "model_lib_url": "https://raw.githubusercontent.com/my-github-account/my-github-repo/main/rp_q4f16_1.wasm",
        "required_features": ["shader-f16"],
      },
    ]
  }

  const selectedModel = "RedPajama-INCITE-Instruct-3B-v1"
  await chat.reload(selectedModel, undefined, myAppConfig);
