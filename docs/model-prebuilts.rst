.. _Model Prebulits:

Model Prebuilts
===============

.. contents:: Table of Contents
    :depth: 3

MLC-LLM is a universal solution for deploying different language models, any language models that can be described in `TVM Relax <https://mlc.ai/chapter_graph_optimization/index.html>`__ (which is a general representation for Neural Networks, and can be imported from models written in PyTorch) should recognized by MLC-LLM and thus deployed to different backends with the help of TVM Unity.

The community has already supported several LLM architectures, including LLaMA, GPT-NeoX, and GPT-J. We eagerly anticipate further contributions from the community to expand the range of supported model architectures, with the goal of democratizing the deployment of LLMs.

.. _off-the-shelf-models:

Off-the-Shelf Models
--------------------

Below is a list of off-the-shelf prebuilt models compiled by MLC-LLM community. These prebuilt models are hosted on Hugging Face, eliminating the need for users to compile them on their own. Each model is accompanied by detailed configurations. These models have undergone extensive testing on various devices, and their kernel performance has been fine-tuned by developers with the help of TVM.

.. list-table:: Off-the-Shelf Models
  :widths: 15 15 15 15
  :header-rows: 1

  * - Model code
    - Model Series
    - Quantization Mode
    - Hugging Face repo
  * - `vicuna-v1-7b-q4f32_0`
    - `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
    - * Weight storage data type: int4
      * Running data type: float32
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q4f32_0>`__
  * - `vicuna-v1-7b-q4f16_0`
    - `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
    - * Weight storage data type: int4
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q4f16_0>`__
  * - `RedPajama-INCITE-Chat-3B-v1-q4f32_0`
    - `RedPajama <https://www.together.xyz/blog/redpajama>`__
    - * Weight storage data type: int4
      * Running data type: float32
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_0>`__
  * - `RedPajama-INCITE-Chat-3B-v1-q4f16_0`
    - `RedPajama <https://www.together.xyz/blog/redpajama>`__
    - * Weight storage data type: int4
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0>`__
  * - `rwkv-raven-1b5-q8f16_0`
    - `RWKV <https://github.com/BlinkDL/RWKV-LM>`__
    - * Weight storage data type: uint8
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-1b5-q8f16_0>`__
  * - `rwkv-raven-3b-q8f16_0`
    - `RWKV <https://github.com/BlinkDL/RWKV-LM>`__
    - * Weight storage data type: uint8
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-3b-q8f16_0>`__
  * - `rwkv-raven-7b-q8f16_0`
    - `RWKV <https://github.com/BlinkDL/RWKV-LM>`__
    - * Weight storage data type: uint8
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-7b-q8f16_0>`__

You can check `MLC-LLM pull requests <https://github.com/mlc-ai/mlc-llm/pulls?q=is%3Aopen+is%3Apr+label%3Anew-models>`__ to track the ongoing efforts of new models. We encourage users to upload their compiled models to Hugging Face and share them with the community.

.. _supported-model-architectures:

Supported Model Architectures
-----------------------------

MLC-LLM supports the following model architectures:

.. list-table:: Supported Model Architectures
  :widths: 15 15 15 15
  :header-rows: 1

  * - Category Code
    - Series
    - Model Definition
    - Variants
  * - ``llama``
    - `LLaMa <https://github.com/facebookresearch/llama>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/llama.py>`__
    - * `Alpaca <https://github.com/tatsu-lab/stanford_alpaca>`__
      * `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
      * `Guanaco <https://github.com/artidoro/qlora>`__
      * `OpenLLaMA <https://github.com/openlm-research/open_llama>`__
  * - ``gpt-neox``
    - `GPT-NeoX <https://github.com/EleutherAI/gpt-neox>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gpt_neox.py>`__
    - * `RedPajama <https://www.together.xyz/blog/redpajama>`__
      * `Dolly <https://github.com/databrickslabs/dolly>`__
      * `Pythia <https://huggingface.co/EleutherAI/pythia-1.4b>`__
  * - ``gptj``
    - `GPT-J <https://github.com/kingoflolz/mesh-transformer-jax>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gptj.py>`__
    - * `MOSS <https://github.com/OpenLMLab/MOSS>`__
  * - ``rwkv``
    - `RWKV <https://github.com/BlinkDL/RWKV-LM>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/rwkv.py>`__
    - * `RWKV-raven <https://github.com/BlinkDL/RWKV-LM>

For models within these model architectures, you can check the :doc:`/tutorials/compile-models` on how to compile models. Please create a new issue if you want to request a new model architecture. Our tutorial :doc:`/tutorials/bring-your-own-models` introduces how to bring a new  model architecture to MLC-LLM.

.. _contribute-models-to-mlc-llm:

Contribute Models to MLC-LLM
----------------------------

Ready to contribute your compiled models/new model architectures? Awesome! Please check :ref:`contribute-new-models` on how to contribute new models to MLC-LLM.
