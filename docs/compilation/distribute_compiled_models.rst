.. _distribute-compiled-models:

Distribute Compiled Models
==========================


This page describes how to distribute the model you compiled so others can use the model in MLC chat runtime.
For demonstration purposes, we show how to compile the `RedPajama-3B instruct model <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`_
(which has different weights from the RedPajama chat model we see in :ref:`compile-models-via-MLC`).


.. contents:: Table of Contents
    :depth: 1
    :local:

Step 1. Compile the model
-------------------------

We first compile the RedPajama-3B instruct model, with the same commands in :ref:`compile-models-via-MLC`.

We first clone HF weights and convert/quantize it to MLC-compatible weights.

.. code:: shell

    mkdir -p dist/rp_instruct_q4f16_1 && mkdir dist/models && cd dist/models
    git lfs install
    git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
    cd ../..
    mlc_chat convert_weight ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ --quantization q4f16_1 -o dist/rp_instruct_q4f16_1/params

.. tabs::

    .. group-tab:: Metal

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_instruct_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json (for Intel Mac, .dylib instead of .so)
            mlc_chat compile ./dist/rp_instruct_q4f16_1/params/mlc-chat-config.json --device metal -o dist/rp_instruct_q4f16_1/rp_instruct_q4f16_1.so

    .. group-tab:: Linux - CUDA

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_instruct_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_instruct_q4f16_1/params/mlc-chat-config.json --device cuda -o dist/rp_instruct_q4f16_1/rp_instruct_q4f16_1.so

    .. group-tab:: Vulkan

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_instruct_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json (for Windows, .dll instead of .so)
            mlc_chat compile ./dist/rp_instruct_q4f16_1/params/mlc-chat-config.json --device vulkan -o dist/rp_instruct_q4f16_1/rp_instruct_q4f16_1.so

Then check the compilation artifact ready on the disk.

.. code:: shell

    ~/mlc-llm > ls dist/rp_instruct_q4f16_1
        rp_instruct_q4f16_1.so                           # ===> the model library
        params                                           # ===> containing the model weights, tokenizer and chat config

    ~/mlc-llm > ls dist/rp_instruct_q4f16_1/params
        mlc-chat-config.json                             # ===> the chat config
        ndarray-cache.json                               # ===> the model weight info
        params_shard_0.bin                               # ===> the model weights
        params_shard_1.bin
        ...
        tokenizer.json                                   # ===> the tokenizer files
        tokenizer_config.json


Step 2. Update MLC Chat Configuration JSON
------------------------------------------

You can **optionally** customize the chat config file
``dist/rp_instruct_q4f16_1/params/mlc-chat-config.json`` (checkout :ref:`configure-mlc-chat-json` for more detailed instructions).
You can also simply use the default configuration and skip this step.

For demonstration purposes, we update ``mean_gen_len`` to 32 and ``max_gen_len`` to 64.
We also update ``conv_template`` to ``"LM"`` because the model is instruction-tuned.


Step 3. Upload Weights to HF
----------------------------

As a next step, we need to upload the model weights.
We only need to upload the files in ``dist/rp_instruct_q4f16_1/params``.
If you also want to host the compiled models on Hugging Face, you can follow the instructions below:

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-redpajama3b-weight-huggingface-repo
    cd my-redpajama3b-weight-huggingface-repo
    cp path/to/mlc-llm/dist/rp_instruct_q4f16_1/params/* .
    git add . && git commit -m "Add redpajama-3b instruct model weights"
    git push origin main

This would result in something like this `example distributed RedPajama-3B chat model repository
<https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1/tree/main>`_.


.. _distribute-compiled-models-step-4:

Step 4. Distribute / Reuse Model library
----------------------------------------

An MLC chat app (e.g. Python API, ``mlc_chat_cli``, web-llm, etc.) needs the model library to run the model.
In the case of RedPajama-3B instruct model, we already have a prebuilt model library for RedPajama-3B chat model that shares the
same model architecture and quantization mode as the instruct model.

In this case, we can simply reuse ``dist/rp_q4f16_1/rp_q4f16_1.so`` compiled in :ref:`compile-models-via-MLC`. That is,
``rp_q4f16_1.so`` and ``rp_instruct_q4f16_1.so`` are the same.

In cases where the compiled model has a new architecture or you customized some logics, feel free to open a PR in the
`binary-mlc-llm-libs repo <https://github.com/mlc-ai/binary-mlc-llm-libs>`_, or in your own repo so that others can access.


..  REPOPULATE BELOW AFTER WE UPLOADING PREBUILT WEIGHTS AND UPDATING RUNTIME
    ---------------------------------
    Good job, you have successfully distributed the model you compiled.
    Next, we will talk about how we can consume the model weights in applications.

    Download the Distributed Models and Run in CLI
    ----------------------------------------------

    The steps needed to run models in CLI are similar to the steps to download the prebuilt model weights and libraries.

    .. code:: shell

        # Clone prebuilt libs so we can reuse them:
        mkdir -p dist/prebuilt
        git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib

        # Or download the model library (only needed if we do not reuse the model lib):
        cd dist/prebuilt/lib
        wget url-to-my-model-lib
        cd ../../..

        # Download the model weights
        cd dist/prebuilt
        git clone https://huggingface.co/my-huggingface-account/my-redpajama3b-weight-huggingface-repo RedPajama-INCITE-Instruct-3B-v1-q4f16_1
        cd ../..
        # Run CLI
        mlc_chat_cli --model RedPajama-INCITE-Instruct-3B-v1-q4f16_1
