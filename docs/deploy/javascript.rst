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
        "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC/resolve/main/",
        "local_id": "Llama-2-7b-chat-hf-q4f32_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      },
      {
        "model_url": "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/resolve/main/",
        "local_id": "Mistral-7B-Instruct-v0.2-q4f16_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm",
        "required_features": ["shader-f16"],
      },
      // Add your own models here...
    ]
  }
  const selectedModel = "Llama-2-7b-chat-hf-q4f32_1"
  // const selectedModel = "Mistral-7B-Instruct-v0.1-q4f16_1"
  await chat.reload(selectedModel, undefined, myAppConfig);

Just like any other platforms, to run a model with on WebLLM, you need:

1. **Model weights** converted to MLC format (e.g. `Llama-2-7b-hf-q4f32_1-MLC 
   <https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC/tree/main>`_.): downloaded through ``model_url``
2. **Model library** that comprises the inference logic (see repo `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__): downloaded through ``model_lib_url``.

Verify Installation for Adding Models
-------------------------------------

In sections below, we walk you through two examples of adding models to WebLLM. Before proceeding,
please verify installation of ``mlc_chat`` and ``tvm``:

**Step 1. Verify mlc_chat**

We use the python package ``mlc_chat`` to compile models. This can be installed by 
following :ref:`install-mlc-packages`, either by building from source, or by
installing the prebuilt package. Verify ``mlc_chat`` installation in command line via:

.. code:: bash

    $ mlc_chat --help
    # You should see help information with this line
    usage: MLC LLM Command Line Interface. [-h] {compile,convert_weight,gen_config}

.. note::
    If it runs into error ``command not found: mlc_chat``, try ``python -m mlc_chat --help``.

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
- Adding ``Llama2-uncensored`` when MLC supports ``Llama2``


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
    mlc_chat convert_weight ./dist/models/WizardMath-7B-V1.1/ \
        --quantization q4f16_1 \
        -o dist/WizardMath-7B-V1.1-q4f16_1-MLC

**Step 2 Generate MLC Chat Config**

Use ``mlc_chat gen_config`` to generate ``mlc-chat-config.json`` and process tokenizers.
See :ref:`compile-command-specification` for specification of ``gen_config``.

.. code:: shell

    mlc_chat gen_config ./dist/models/WizardMath-7B-V1.1/ \
        --quantization q4f16_1 --conv-template wizard_coder_or_math \
        -o dist/WizardMath-7B-V1.1-q4f16_1-MLC/

For the ``conv-template``, `conv_template.cc <https://github.com/mlc-ai/mlc-llm/blob/main/cpp/conv_templates.cc>`__
contains a full list of conversation templates that MLC provides.

If the model you are adding requires a new conversation template, you would need to add your own.
Follow `this PR <https://github.com/mlc-ai/mlc-llm/pull/1402>`__ as an example. Besides, you also need to add the new template to ``/path/to/web-llm/src/conversation.ts``.
We look up the template to use with the ``conv_template`` field in ``mlc-chat-config.json``.

For more details, please see :ref:`configure-mlc-chat-json`.

.. note:: 

  If you added your conversation template in ``src/conversation.ts``, you need to build WebLLM
  from source following the instruction in
  `the WebLLM repo's README <https://github.com/mlc-ai/web-llm?tab=readme-ov-file#build-webllm-package-from-source>`_. 

  Alternatively, you could use the ``"custom"`` conversation template so that you can pass in
  your own ``ConvTemplateConfig`` in runtime without having to build the package from source.

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

We simply specify the Huggingface link as ``model_url``, while reusing the ``model_lib_url`` for 
``Mistral-7B``. Note that we need the suffix to be ``/resolve/main/``.

.. code:: typescript

  const myAppConfig: AppConfig = {
    model_list: [
      // Other records here omitted...
      {
        // Substitute model_url with the one you created `my-huggingface-account/my-wizardMath-weight-huggingface-repo`
        "model_url": "https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC/resolve/main/",
        "local_id": "WizardMath-7B-V1.1-q4f16_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm",
        "required_features": ["shader-f16"],
      },
    ]
  }

  const selectedModel = "WizardMath-7B-V1.1-q4f16_1"
  await chat.reload(selectedModel, undefined, myAppConfig);

Now, running the ``get-started`` example will use the ``WizardMath`` model you just added.
See `get-started's README <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started#webllm-get-started-app>`__
on how to run it. 


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

In this section, we walk you through adding ``RedPajama-INCITE-Chat-3B-v1`` to the
`get-started <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started>`__ example.

This section largely replicates :ref:`compile-model-libraries`. See that page for
more details, specifically the ``WebGPU`` option.

**Step 0. Install dependencies**

To compile model libraries for webgpu, you need to :ref:`build mlc_chat from source <mlcchat_build_from_source>`.
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
    mlc_chat convert_weight ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
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
    mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
        --quantization q4f16_1 --conv-template redpajama_chat \
        -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
    # 2. compile: compile model library with specification in mlc-chat-config.json
    mlc_chat compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
        --device webgpu -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm

.. note::
    When compiling larger models like ``Llama-2-7B``, you may want to add ``--prefill_chunk_size 1024`` or
    lower ``context_window_size`` to decrease memory usage. Otherwise, during runtime,
    you may run into issues like:

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
        "model_url": "https://huggingface.co/my-hf-account/my-redpajama3b-weight-huggingface-repo/resolve/main/",
        "local_id": "RedPajama-INCITE-Instruct-3B-v1",
        "model_lib_url": "https://raw.githubusercontent.com/my-gh-account/my-repo/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm",
        "required_features": ["shader-f16"],
      },
    ]
  }

  const selectedModel = "RedPajama-INCITE-Instruct-3B-v1"
  await chat.reload(selectedModel, undefined, myAppConfig);

Now, running the ``get-started`` example will use the ``RedPajama`` model you just added.
See `get-started's README <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started#webllm-get-started-app>`__
on how to run it. 