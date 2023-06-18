.. _distribute-compiled-models:

Distribute Compiled Models
==========================


This page describes how to distribute the model you compiled so others can use the model in MLC chat runtime.
For demonstration purposes, we show how to compile a llama-7b variant
`OpenLLaMA-7B <https://huggingface.co/openlm-research/open_llama_7b>`_ model.
We assume the model is already compiled.

.. note::

    If you have not compiled the model you want to distribute, please go back to the :doc:`model compilation page </compilation/compile_models>` to compile the model.

.. contents:: Table of Contents
    :depth: 1
    :local:

Step 1. Check the build artifact
--------------------------------

To begin with, we can check that we have the compilation artifact ready on the disk.

.. code:: shell

    ~/mlc-llm > ls dist/open-llama-7b-q3f16_0
        open-llama-7b-q3f16_0-metal.so                   # ===> the model library
        mod_cache_before_build_metal.pkl                 # ===> a cached file for future builds
        params                                           # ===> containing the model weights, tokenizer and chat config

    ~/mlc-llm > ls dist/open-llama-7b-q3f16_0/params
        mlc-chat-config.json                             # ===> the chat config
        ndarray-cache.json                               # ===> the model weight info
        params_shard_0.bin                               # ===> the model weights
        params_shard_1.bin
        ...
        tokenizer.json                                   # ===> the tokenizer files
        tokenizer_config.json

You are expected to see the same folder structure for the model you compiled.

Step 2. Update MLC Chat Configuration JSON
------------------------------------------

You can **optionally** customize the chat config file
``dist/open-llama-7b-q3f16_0/params/mlc-chat-config.json`` (checkout :ref:`configure-mlc-chat-json` for more detailed instructions).
Youc an also simply use the default configuration and skip this step.

Step 3. Specify the Model Lib
-----------------------------

An MLC chat app needs to look for the model library to run the model.
In the case of llama-7b, we already have a prebuilt model lib for vicuna that shares the
same model architecture and quantization mode.
We can edit ``dist/open-llama-7b-q3f16_0/params/mlc-chat-config.json`` and update the value of field ``model_lib`` to ``"vicuna-v1-7b-q3f16_0"``.

.. note::

    We recommend reusing the model lib for the same architecture with different weight variants.
    You can leverage the ``--reuse-lib`` in the build to specify the library you want to reuse or edit the chat config afterward.
    Reusing model lib allows us to run the model on existing MLC apps (e.g. iOS) that requires static packaging.


We should distribute the generated model lib if we want to build a new model architecture or try out customized compilation optimizations.
In this case, we should keep the ``model_lib`` field as ``"open-llama-7b-q3f16_0"``.
You can upload the model library ``dist/open-llama-7b-q3f16_0/open-llama-7b-q3f16_0-metal.so``
and ask others to download it to  `dist/prebuilt/lib` directory so the CLI app can pick it up.


Step 4. Upload the Compiled Model Weights
-----------------------------------------

As a next step, we need to upload the model weights.
We only need to upload the files in ``dist/open-llama-7b-q3f16_0/params``.
If you also want to host the compiled models on Hugging Face, you can follow the instructions below:

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-openllama7b-weight-huggingface-repo
    cd my-openllama7b-weight-huggingface-repo
    cp path/to/mlc-llm/dist/open-llama-7b-q3f16_0/params/* .
    git commit -m "Add open-llama-7b model weights"
    git push


---------------------------------

Good job, you have successfully distributed the model you compiled.
Next, we will talk about how to we can consume the model weights

Download the Distributed Models and Run in CLI
----------------------------------------------

The steps needed to run models in CLI are similar to the steps to download the prebuilt model weights and libraries.

.. code:: shell

    # clone prebuilt libs so we can reuse them
    mkdir -p dist/prebuilt
    git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib

    # Download the model library (only needed if we are not reusing the model lib)
    cd dist/prebuilt/lib
    wget url-to-my-model-lib
    cd ../../..

    # Download the model weights
    cd dist/prebuilt
    git clone https://huggingface.co/my-huggingface-account/my-openllama7b-weight-huggingface-repo open-llama-7b-q3f16_0
    cd ../..
    # Run CLI
    mlc_chat_cli --local-id open-llama-7b-q3f16_0


Download the Distributed Models and Run in iOS App
--------------------------------------------------

For iOS app, model libraries are packed into the app at the time of app building.
Therefore, the iOS app supports running any models whose model libraries are integrated into the app.
You can check the :ref:`list of supported model libraries <prebuilt-models-ios>`.

To download and run the compiled OpenLLaMA model on iPhone, we need to reuse the integrated `vicuna-v1-7b-q3f16_0` model library, because both OpenLLaMA and Vicuna are LLaMA-family models.

To reuse the `vicuna-v1-7b-q3f16_0` model library, we make sure we already updated the MLC Chat config: open ``dist/open-llama-7b-q3f16_0/params/mlc-chat-config.json``,
update the value of field ``model_lib`` to ``"vicuna-v1-7b-q3f16_0"``.

After replacing the ``model_lib`` value, upload the updated `mlc-chat-config.json` to the location where you host your model weights.
For our example here, we use git to push the change to `my-openllama7b-weight-huggingface-repo`.

Now we can download the model weights in iOS app and run the model by following the steps below:

.. tabs::

    .. tab:: Step 1

        Open "MLCChat" app, click "Add model variant".

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-1.png
            :align: center
            :width: 30%

    .. tab:: Step 2

        Paste the repository URL of the model built on your own, and click Add.

        You can refer to the link in the image as an example.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-2.png
            :align: center
            :width: 30%

    .. tab:: Step 3

        After adding the model, you can download your model from the URL by clicking the download button.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-3.png
            :align: center
            :width: 30%

    .. tab:: Step 4

        When the download is finished, click into the model and enjoy.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-4.png
            :align: center
            :width: 30%

.. for a blank line

|
