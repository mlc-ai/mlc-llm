Getting Vicuna Weights
======================

.. contents:: Table of Contents
   :local:
   :depth: 2

`Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`_ is an open-source chatbot trained by fine-tuning `LLaMA <https://ai.facebook.com/blog/large-language-model-llama-meta-ai/>`_ on `ShartGPT <https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered>`_ data.

Please note that the official Vicuna weights are delta weights applied to the LLaMA weights in order to comply with the LLaMA license. Users are responsible for applying these delta weights themselves.

In this tutorial, we will show how to apply the delta weights to LLaMA weights to get Vicuna weights.

Install FastChat
----------------

FastChat offers convenient utility functions for applying the delta to LLaMA weights. You can easily install it using pip.

.. code-block:: bash

    pip install fschat

Download HuggingFace LLaMA Weights
----------------------------------

The HuggingFace LLaMA weights are hosted using Git-LFS. Therefore, it is necessary to install Git-LFS first (you can ignore this step if git-lfs is already installed).

.. code-block:: bash

    conda install git-lfs
    git lfs install

Then download the weights (both the LLaMA weight and Vicuna delta weight):

.. code-block:: bash

    git clone https://huggingface.co/decapoda-research/llama-7b-hf
    git clone https://huggingface.co/lmsys/vicuna-7b-delta-v1.1


There is a name misalignment issue in the LLaMA weights and Vicuna delta weights.
Please follow these steps to modify the content of the "config.json" file:

.. code-block:: bash

    sed -i 's/LLaMAForCausalLM/LlamaForCausalLM/g' llama-7b-hf/config.json

Then use ``fschat`` to apply the delta to LLaMA weights

.. code-block:: bash

    python3 -m fastchat.model.apply_delta \
        --base-model-path llama-7b-hf \
        --target-model-path vicuna-7b-v1.1 \
        --delta-path vicuna-7b-delta-v1.1

You will get the Vicuna weights in ``vicuna-7b-v1.1`` folder, which can be used as input of MLC-LLM to further compile models.


(Optional) Move Vicuna Weights to dist folder
---------------------------------------------

The default model path of MLC-LLM is ``dist`` folder. Therefore, it is recommended to move the Vicuna weights to ``dist`` folder.

.. code-block:: bash

    mkdir -p dist/models
    mv vicuna-7b-v1.1 dist/models/vicuna-7b-v1.1
