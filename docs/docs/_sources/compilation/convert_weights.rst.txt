.. _convert-weights-via-MLC:

Convert Model Weights
=====================

To run a model with MLC LLM,
we need to convert model weights into MLC format (e.g. `RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/tree/main>`_.)
This page walks us through the process of adding a model variant with ``mlc_llm convert_weight``, which
takes a huggingface model as input and converts/quantizes into MLC-compatible weights.

Specifically, we add RedPjama-INCITE-**Instruct**-3B-v1, while MLC already
provides a model library for RedPjama-INCITE-**Chat**-3B-v1, which we can reuse.

This can be extended to, e.g.:

- Add ``OpenHermes-Mistral`` when MLC already supports Mistral
- Add ``Llama-2-uncensored`` when MLC already supports Llama-2

.. note::
    Before you proceed, make sure you followed :ref:`install-tvm-unity`, a required
    backend to compile models with MLC LLM.

    Please also follow the instructions in :ref:`deploy-cli` / :ref:`deploy-python-engine` to obtain
    the CLI app / Python API that can be used to chat with the compiled model.


.. contents:: Table of Contents
    :depth: 1
    :local:

.. _verify_installation_for_compile:

0. Verify installation
----------------------

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


1. Clone from HF and convert_weight
-----------------------------------

You can be under the mlc-llm repo, or your own working directory. Note that all platforms
can share the same compiled/quantized weights. See :ref:`compile-command-specification`
for specification of ``convert_weight``.

.. code:: shell

    # Create directory
    mkdir -p dist/models && cd dist/models
    # Clone HF weights
    git lfs install
    git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
    cd ../..
    # Convert weight
    mlc_llm convert_weight ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ \
        --quantization q4f16_1 \
        -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC

.. _generate_mlc_chat_config:

2. Generate MLC Chat Config
---------------------------

Use ``mlc_llm gen_config`` to generate ``mlc-chat-config.json`` and process tokenizers.
See :ref:`compile-command-specification` for specification of ``gen_config``.

.. code:: shell

    mlc_llm gen_config ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ \
        --quantization q4f16_1 --conv-template redpajama_chat \
        -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/


.. note::
    The file ``mlc-chat-config.json`` is crucial in both model compilation
    and runtime chatting. Here we only care about the latter case.

    You can **optionally** customize
    ``dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/mlc-chat-config.json`` (checkout :ref:`configure-mlc-chat-json` for more detailed instructions).
    You can also simply use the default configuration.

    `conversation_template <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/conversation_template>`__
    directory contains a full list of conversation templates that MLC provides. If the model you are adding
    requires a new conversation template, you would need to add your own.
    Follow `this PR <https://github.com/mlc-ai/mlc-llm/pull/2163>`__ as an example. However,
    adding your own template would require you :ref:`build mlc_llm from source <mlcchat_build_from_source>` in order for it
    to be recognized by the runtime.

By now, you should have the following files.

.. code:: shell

    ~/mlc-llm > ls dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC
        mlc-chat-config.json                             # ===> the chat config
        ndarray-cache.json                               # ===> the model weight info
        params_shard_0.bin                               # ===> the model weights
        params_shard_1.bin
        ...
        tokenizer.json                                   # ===> the tokenizer files
        tokenizer_config.json

.. _distribute-compiled-models:

(Optional) 3. Upload weights to HF
----------------------------------

Optionally, you can upload what we have to huggingface.

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
<https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/tree/main>`_, but
for **Instruct** instead of **Chat**.

Good job, you have successfully distributed the model you compiled.
Next, we will talk about how we can consume the model weights in applications.

Download the Distributed Models
-------------------------------

You can now use the existing mlc tools such as chat/serve/package with the converted weights.

.. code:: shell

    mlc_llm chat HF://my-huggingface-account/my-redpajama3b-weight-huggingface-repo
