Distribute Compiled Models
==========================

When you want to run the model compiled by yourself on mobile devices and/or web browser, you need to distribute the model you compiled to the Internet (for example, as a repository in Hugging Face), so that the applications released by MLC LLM can download your model from the Internet location.

This page introduces how to distribute the model you compiled.
For demonstration purpose, here we want to distribute the compiled `OpenLLaMA-7B <https://huggingface.co/openlm-research/open_llama_7b>`_ model.
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

You can **optionally** :doc:`update the MLCChat configuration JSON </get_started/mlc_chat_config>` file ``dist/open-llama-7b-q3f16_0/params/mlc-chat-config.json``. Open and update the JSON file and customize it according to your needs.

You can also use the default configuration, in which case no action is needed in this step.

Step 3. Host the Model Library
------------------------------

Now we upload the model library to an Internet location.
That is to say, for our OpenLLaMA example here, we need to host ``dist/open-llama-7b-q3f16_0/open-llama-7b-q3f16_0-metal.so`` on the Internet.

Any publicly available Internet location is good to host the model library.
The prebuilt model libraries are hosted in `a GitHub repo <https://github.com/mlc-ai/binary-mlc-llm-libs>`_.
If you also want to host your model library on GitHub, you can follow the instructions below:

.. code:: shell

    # First, please create a repository on GitHub.
    # With the repository created, run
    git clone https://github.com/my-github-account/my-model-library-github-repo.git
    cd my-model-library-github-repo
    cp path/to/mlc-llm/dist/open-llama-7b-q3f16_0/open-llama-7b-q3f16_0-metal.so .
    git commit -m "Add open-llama-7b Metal library"
    git push

Step 4. Host the Compiled Model Weights
---------------------------------------

Now we upload the compiled model weights to Internet.
That is to say, for our OpenLLaMA example here, we need to host ``dist/open-llama-7b-q3f16_0/params`` on the Internet.

Any publicly available Internet location is good to host the compiled model weights.
MLC LLM uses Hugging Face repositories to host the prebuilt model weights.
If you also want to host your model library on Hugging Face, you can follow the instructions below:

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
Next, we will talk about how to download and run models with the distributed library and weights.

Download the Distributed Models and Run in CLI
----------------------------------------------

The steps needed to run models in CLI are similar to the steps to download the prebuilt model weights and libraries.

.. code:: shell

    mkdir -p dist/prebuilt/lib
    # Download the model library
    git clone https://github.com/my-github-account/my-model-library-github-repo.git
    cp my-model-library-github-repo/* dist/prebuilt/lib/
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

To reuse the `vicuna-v1-7b-q3f16_0` model library, we need to update the MLC Chat config: open ``dist/open-llama-7b-q3f16_0/params/mlc-chat-config.json``, replace the value of field ``model_lib`` to ``"vicuna-v1-7b-q3f16_0"``.

After replacing the ``model_lib`` value, upload the updated `mlc-chat-config.json` to the location where you host your model weights. For our example here, we use git to push the change to `my-openllama7b-weight-huggingface-repo`.

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
