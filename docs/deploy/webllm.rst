.. _webllm-runtime:

WebLLM Javascript SDK
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

`WebLLM <https://www.npmjs.com/package/@mlc-ai/web-llm>`_ is a high-performance in-browser LLM
inference engine, aiming to be the backend of AI-powered web applications and agents.

It provides a specialized runtime for the web backend of MLCEngine, leverages
`WebGPU <https://www.w3.org/TR/webgpu/>`_ for local acceleration, offers OpenAI-compatible API,
and provides built-in support for web workers to separate heavy computation from the UI flow.

Please checkout the `WebLLM repo <https://github.com/mlc-ai/web-llm>`__ on how to use WebLLM to build
web application in Javascript/Typescript. Here we only provide a high-level idea and discuss how to
use MLC-LLM to compile your own model to run with WebLLM.

Getting Started
---------------

To get started, try out `WebLLM Chat <https://chat.webllm.ai/>`__, which provides a great example
of integrating WebLLM into a full web application.

A WebGPU-compatible browser is needed to run WebLLM-powered web applications.
You can download the latest Google Chrome and use `WebGPU Report <https://webgpureport.org/>`__
to verify the functionality of WebGPU on your browser.

WebLLM is available as an `npm package <https://www.npmjs.com/package/@mlc-ai/web-llm>`_ and is
also CDN-delivered. Try a simple chatbot example in
`this JSFiddle example <https://jsfiddle.net/neetnestor/4nmgvsa2/>`__ without setup.

You can also checkout `existing examples <https://github.com/mlc-ai/web-llm/tree/main/examples>`__
on more advanced usage of WebLLM such as JSON mode, streaming, and more.

Model Records in WebLLM
-----------------------

Each of the model in `WebLLM Chat <https://chat.webllm.ai>`__ is registered as an instance of
``ModelRecord`` and can be accessed at
`webllm.prebuiltAppConfig.model_list <https://github.com/mlc-ai/web-llm/blob/main/src/config.ts#L293>`__.

Looking at the most straightforward example `get-started <https://github.com/mlc-ai/web-llm/blob/main/examples/get-started/src/get_started.ts>`__,
there are two ways to run a model.

One can either use the prebuilt model by simply calling ``reload()`` with the ``model_id``:

.. code:: typescript

  const selectedModel = "Llama-3-8B-Instruct-q4f32_1-MLC";
  const engine = await webllm.CreateMLCEngine(selectedModel);

Or one can specify their own model to run by creating a model record:

.. code:: typescript

  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        model: "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f32_1-MLC",
        model_id: "Llama-3-8B-Instruct-q4f32_1-MLC",
        model_lib:
          webllm.modelLibURLPrefix +
          webllm.modelVersion +
          "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      },
      // Add your own models here...
    ],
  };
  const selectedModel = "Llama-3-8B-Instruct-q4f32_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { appConfig: appConfig },
  );

Looking at the code above, we find that, just like any other platforms supported by MLC-LLM, to
run a model on WebLLM, you need:

1. **Model weights** converted to MLC format (e.g. `Llama-3-8B-Instruct-q4f32_1-MLC
   <https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f32_1-MLC/tree/main>`_.): downloaded through the url ``ModelRecord.model``
2. **Model library** that comprises the inference logic (see repo `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs/tree/main/web-llm-models>`__): downloaded through the url ``ModelRecord.model_lib``.

In sections below, we walk you through two examples on how to add your own model besides the ones in
`webllm.prebuiltAppConfig.model_list <https://github.com/mlc-ai/web-llm/blob/main/src/config.ts#L293>`__.
Before proceeding, please verify installation of ``mlc_llm`` and ``tvm``.

Verify Installation for Adding Models
-------------------------------------

**Step 1. Verify mlc_llm**

We use the python package ``mlc_llm`` to compile models. This can be installed by
following :ref:`install-mlc-packages`, either by building from source, or by
installing the prebuilt package. Verify ``mlc_llm`` installation in command line via:

.. code:: bash

    $ mlc_llm --help
    # You should see help information with this line
    usage: MLC LLM Command Line Interface. [-h] {compile,convert_weight,gen_config}

.. note::
    If it runs into error ``command not found: mlc_llm``, try ``python -m mlc_llm --help``.

**Step 2. Verify TVM**

To compile models, you also need to follow :ref:`install-tvm-unity`.
Here we verify ``tvm`` quickly with command line (for full verification, see :ref:`tvm-unity-validate`):

.. code:: bash

    $ python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.11/site-packages/tvm/__init__.py


.. _webllm-add-model-variant:

Bring Your Own Model Variant
----------------------------

In cases where the model you are adding is simply a variant of an existing
model, we only need to convert weights and reuse existing model library. For instance:

- Adding ``OpenMistral`` when MLC supports ``Mistral``
- Adding a ``Llama3`` fine-tuned on a domain-specific task when MLC supports ``Llama3``


In this section, we walk you through adding ``WizardMath-7B-V1.1-q4f16_1`` to the
`get-started <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started>`__ example.
According to the model's ``config.json`` on `its Huggingface repo <https://huggingface.co/WizardLM/WizardMath-7B-V1.1/blob/main/config.json>`_,
it reuses the Mistral model architecture.

.. note::

  This section largely replicates :ref:`convert-weights-via-MLC`.
  See that page for more details. Note that the weights are shared across
  all platforms in MLC.

**Step 1 Clone from HF and convert_weight**

You can be under the mlc-llm repo, or your own working directory. Note that all platforms
can share the same compiled/quantized weights. See :ref:`compile-command-specification`
for specification of ``convert_weight``.

.. code:: shell

    # Create directory
    mkdir -p dist/models && cd dist/models
    # Clone HF weights
    git lfs install
    git clone https://huggingface.co/WizardLM/WizardMath-7B-V1.1
    cd ../..
    # Convert weight
    mlc_llm convert_weight ./dist/models/WizardMath-7B-V1.1/ \
        --quantization q4f16_1 \
        -o dist/WizardMath-7B-V1.1-q4f16_1-MLC

**Step 2 Generate MLC Chat Config**

Use ``mlc_llm gen_config`` to generate ``mlc-chat-config.json`` and process tokenizers.
See :ref:`compile-command-specification` for specification of ``gen_config``.

.. code:: shell

    mlc_llm gen_config ./dist/models/WizardMath-7B-V1.1/ \
        --quantization q4f16_1 --conv-template wizard_coder_or_math \
        -o dist/WizardMath-7B-V1.1-q4f16_1-MLC/

For the ``conv-template``, `conversation_template.py <https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_llm/conversation_template>`__
contains a full list of conversation templates that MLC provides. You can also manually modify the ``mlc-chat-config.json`` to
add your customized conversation template.

**Step 3 Upload weights to HF**

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-wizardMath-weight-huggingface-repo
    cd my-wizardMath-weight-huggingface-repo
    cp path/to/mlc-llm/dist/WizardMath-7B-V1.1-q4f16_1-MLC/* .
    git add . && git commit -m "Add wizardMath model weights"
    git push origin main

After successfully following all steps, you should end up with a Huggingface repo similar to
`WizardMath-7B-V1.1-q4f16_1-MLC <https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC>`__,
which includes the converted/quantized weights, the ``mlc-chat-config.json``, and tokenizer files.


**Step 4 Register as a ModelRecord**

Finally, we modify the code snippet for
`get-started <https://github.com/mlc-ai/web-llm/blob/main/examples/get-started/src/get_started.ts>`__
pasted above.

We simply specify the Huggingface link as ``model``, while reusing the ``model_lib`` for
``Mistral-7B``.

.. code:: typescript

  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        model: "https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC",
        model_id: "WizardMath-7B-V1.1-q4f16_1-MLC",
        model_lib:
          webllm.modelLibURLPrefix +
          webllm.modelVersion +
          "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      },
      // Add your own models here...
    ],
  };

  const selectedModel = "WizardMath-7B-V1.1-q4f16_1"
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { appConfig: appConfig },
  );

Now, running the ``get-started`` example will use the ``WizardMath`` model you just added.
See `get-started's README <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started#webllm-get-started-app>`__
on how to run it.


Bring Your Own Model Library
----------------------------

A model library is specified by:

 - The model architecture (e.g. ``llama-3``, ``gpt-neox``, ``phi-3``)
 - Quantization (e.g. ``q4f16_1``, ``q0f32``)
 - Metadata (e.g. ``context_window_size``, ``sliding_window_size``, ``prefill-chunk-size``), which affects memory planning (currently only ``prefill-chunk-size`` affects the compiled model)
 - Platform (e.g. ``cuda``, ``webgpu``, ``iOS``)

In cases where the model you want to run is not compatible with the provided MLC
prebuilt model libraries (e.g. having a different quantization, a different
metadata spec, or even a different model architecture), you need to build your
own model library.

In this section, we walk you through adding ``RedPajama-INCITE-Chat-3B-v1`` to the
`get-started <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started>`__ example.

This section largely replicates :ref:`compile-model-libraries`. See that page for
more details, specifically the ``WebGPU`` option.

**Step 0. Install dependencies**

To compile model libraries for webgpu, you need to :ref:`build mlc_llm from source <mlcchat_build_from_source>`.
Besides, you also need to follow :ref:`install-web-build`. Otherwise, it would run into error:

.. code:: text

    RuntimeError: Cannot find libraries: wasm_runtime.bc

**Step 1. Clone from HF and convert_weight**

You can be under the mlc-llm repo, or your own working directory. Note that all platforms
can share the same compiled/quantized weights.

.. code:: shell

    # Create directory
    mkdir -p dist/models && cd dist/models
    # Clone HF weights
    git lfs install
    git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
    cd ../..
    # Convert weight
    mlc_llm convert_weight ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
        --quantization q4f16_1 \
        -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC

**Step 2. Generate mlc-chat-config and compile**

A model library is specified by:

 - The model architecture (e.g. ``llama-2``, ``gpt-neox``)
 - Quantization (e.g. ``q4f16_1``, ``q0f32``)
 - Metadata (e.g. ``context_window_size``, ``sliding_window_size``, ``prefill-chunk-size``), which affects memory planning
 - Platform (e.g. ``cuda``, ``webgpu``, ``iOS``)

All these knobs are specified in ``mlc-chat-config.json`` generated by ``gen_config``.

.. code:: shell

    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
    mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
        --quantization q4f16_1 --conv-template redpajama_chat \
        -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
    # 2. compile: compile model library with specification in mlc-chat-config.json
    mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
        --device webgpu -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm

.. note::
    When compiling larger models like ``Llama-3-8B``, you may want to add ``--prefill_chunk_size 1024``
    to decrease memory usage. Otherwise, during runtime, you may run into issues like:

    .. code:: text

        TypeError: Failed to execute 'createBuffer' on 'GPUDevice': Failed to read the 'size' property from
        'GPUBufferDescriptor': Value is outside the 'unsigned long long' value range.


**Step 3. Distribute model library and model weights**

After following the steps above, you should end up with:

.. code:: shell

    ~/mlc-llm > ls dist/libs
      RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm  # ===> the model library

    ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
      mlc-chat-config.json                             # ===> the chat config
      ndarray-cache.json                               # ===> the model weight info
      params_shard_0.bin                               # ===> the model weights
      params_shard_1.bin
      ...
      tokenizer.json                                   # ===> the tokenizer files
      tokenizer_config.json

Upload the ``RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm`` to a github repository (for us,
it is in `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__). Then
upload the ``RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC`` to a Huggingface repo:

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-redpajama3b-weight-huggingface-repo
    cd my-redpajama3b-weight-huggingface-repo
    cp path/to/mlc-llm/dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/* .
    git add . && git commit -m "Add redpajama-3b instruct model weights"
    git push origin main

This would result in something like `RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
<https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/tree/main>`_.

**Step 4. Register as a ModelRecord**

Finally, we are able to run the model we added in WebLLM's `get-started <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started>`__:

.. code:: typescript

  const myAppConfig: AppConfig = {
    model_list: [
      // Other records here omitted...
      {
        "model": "https://huggingface.co/my-hf-account/my-redpajama3b-weight-huggingface-repo/resolve/main/",
        "model_id": "RedPajama-INCITE-Instruct-3B-v1",
        "model_lib": "https://raw.githubusercontent.com/my-gh-account/my-repo/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm",
        "required_features": ["shader-f16"],
      },
    ]
  }

  const selectedModel = "RedPajama-INCITE-Instruct-3B-v1";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { appConfig: appConfig },
  );

Now, running the ``get-started`` example will use the ``RedPajama`` model you just added.
See `get-started's README <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started#webllm-get-started-app>`__
on how to run it.
