.. _distribute-compiled-models:

Distribute Compiled Models
==========================


This page describes how to distribute the model you compiled so others can use the model in MLC chat runtime.
For demonstration purposes, we show how to compile the `RedPajama-3B instruct model <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`_
(which has different weights from the RedPajama chat model).


If you have not compiled the RedPajama-3B instruct model,
you can use the following command to compile it:

.. tabs::

    .. group-tab:: Metal

        .. code:: shell

            python3 -m mlc_llm.build --hf-path togethercomputer/RedPajama-INCITE-Instruct-3B-v1 --target metal --quantization q4f16_1

    .. group-tab:: Linux - CUDA

        .. code:: shell

            python3 -m mlc_llm.build --hf-path togethercomputer/RedPajama-INCITE-Instruct-3B-v1 --target cuda --quantization q4f16_1

    .. group-tab:: Vulkan

        .. code:: shell

            python3 -m mlc_llm.build --hf-path togethercomputer/RedPajama-INCITE-Instruct-3B-v1 --target vulkan --quantization q4f16_1


.. contents:: Table of Contents
    :depth: 1
    :local:

Step 1. Check the Build Artifact
--------------------------------

To begin with, we can check that we have the compilation artifact ready on the disk.

.. code:: shell

    ~/mlc-llm > ls dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1
        RedPajama-INCITE-Instruct-3B-v1-q4f16_1-metal.so  # ===> the model library
        mod_cache_before_build_metal.pkl                  # ===> a cached file for future builds
        params                                            # ===> containing the model weights, tokenizer and chat config

    ~/mlc-llm > ls dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1/params
        mlc-chat-config.json                              # ===> the chat config
        ndarray-cache.json                                # ===> the model weight info
        params_shard_0.bin                                # ===> the model weights
        params_shard_1.bin
        ...
        tokenizer.json                                    # ===> the tokenizer files
        tokenizer_config.json

You are expected to see the same folder structure for the model you compiled.

Step 2. Update MLC Chat Configuration JSON
------------------------------------------

You can **optionally** customize the chat config file
``dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1/params/mlc-chat-config.json`` (checkout :ref:`configure-mlc-chat-json` for more detailed instructions).
You can also simply use the default configuration and skip this step.

For demonstration purpose, we update ``mean_gen_len`` to 32 and ``max_gen_len`` to 64.
We also update ``conv_template`` to ``"LM"`` because the model is instruction-tuned.


.. _distribute-model-step3-specify-model-lib:

Step 3. Specify the Model Lib
-----------------------------

An MLC chat app needs to look for the model library to run the model.
In the case of RedPajama-3B instruct model, we already have a prebuilt model lib for RedPajama-3B chat model that shares the
same model architecture and quantization mode as the instruct model.
We can edit ``dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1/params/mlc-chat-config.json``
and update the value of field ``model_lib`` to ``"RedPajama-INCITE-Chat-3B-v1-q4f16_1"``.

.. note::

    We recommend reusing the model lib for the same architecture with different weight variants.
    You can leverage the ``--reuse-lib`` in the compilation command to specify the library you want to reuse or edit the chat config afterward.
    Reusing model lib allows us to run the model on existing MLC apps (e.g. iOS) that requires static packaging.

    For example, if you have compiled RedPajama-3B chat model before, then you can use the following command to compile the instruct model,
    which reuses the compiled chat model library:

    .. code:: shell

        python3 -m mlc_llm.build --hf-path togethercomputer/RedPajama-INCITE-Instruct-3B-v1 --reuse-lib RedPajama-INCITE-Chat-3B-v1-q4f16_1 --target [your target] --quantization q4f16_1

    In this way, `mlc_llm.build` does not produce the model library for the instruct model, and in `mlc-chat-config.json`
    the ``model_lib`` field is set to ``RedPajama-INCITE-Chat-3B-v1-q4f16_1``.

    Please note that only models with same architecture and compiled with same quantization modes can reuse and share model library.


We should distribute the generated model lib if we want to build a new model architecture or try out customized compilation optimizations.
In this case, we should keep the ``model_lib`` field as ``"RedPajama-INCITE-Instruct-3B-v1-q4f16_1"``.
You can upload the model library ``dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-metal.so``
and ask others to download it to  `dist/prebuilt/lib` directory so the CLI app can pick it up.


Step 4. Upload the Compiled Model Weights
-----------------------------------------

As a next step, we need to upload the model weights.
We only need to upload the files in ``dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1/params``.
If you also want to host the compiled models on Hugging Face, you can follow the instructions below:

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-redpajama3b-weight-huggingface-repo
    cd my-redpajama3b-weight-huggingface-repo
    cp path/to/mlc-llm/dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1/params/* .
    git add . && git commit -m "Add redpajama-3b instruct model weights"
    git push origin main

Here we provide an `example distributed RedPajama-3B instruct model repository <https://huggingface.co/mlc-ai/RedPajama-INCITE-Instruct-3B-v1-q4f16_1/tree/main>`_ which you can refer to.

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


Download the Distributed Models and Run in iOS App
--------------------------------------------------

For iOS app, model libraries are statically packed into the app at the time of app building.
Therefore, the iOS app supports running any models whose model libraries are integrated into the app.
You can check the :ref:`list of supported model libraries <prebuilt-models-ios>`.

To download and run the compiled RedPajama-3B instruct model on iPhone, we need to reuse the integrated ``RedPajama-INCITE-Chat-3B-v1-q4f16_1`` model library.
Please revisit :ref:`distribute-model-step3-specify-model-lib` and make sure the ``model_lib`` field of `mlc-chat-config.json` is set to ``RedPajama-INCITE-Chat-3B-v1-q4f16_1``.

Now we can download the model weights in iOS app and run the model by following the steps below:

.. tabs::

    .. tab:: Step 1

        Open "MLCChat" app, click "Add model variant".

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-distribute-1.jpeg
            :align: center
            :width: 30%

    .. tab:: Step 2

        Paste the repository URL of the model built on your own, and click "Add".

        You can refer to the link in the image as an example.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-distribute-2.jpeg
            :align: center
            :width: 30%

    .. tab:: Step 3

        After adding the model, you can download your model from the URL by clicking the download button.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-distribute-3.jpeg
            :align: center
            :width: 30%

    .. tab:: Step 4

        When the download is finished, click into the model and enjoy.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-distribute-4.jpeg
            :align: center
            :width: 30%

.. for a blank line

|
