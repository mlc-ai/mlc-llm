Model Zoo
=========

.. contents:: Table of Contents
    :depth: 3

MLC-LLM is a universal solution for deploying different language models, any language models that can be described in `TVM Relax <https://mlc.ai/chapter_graph_optimization/index.html>`__ (which is a general representation for Neural Networks, and can be imported from models written in PyTorch) should recognized by MLC-LLM and thus deployed to different backends with the help of TVM Unity.

The community has already supported several LLM architectures, including LLaMA, GPT-NeoX, and MOSS. We eagerly anticipate further contributions from the community to expand the range of supported model architectures, with the goal of democratizing the deployment of LLMs.

List of Officially Supported Models
-----------------------------------

Below is a list of models officially supported by MLC-LLM community. Pre-built models are hosted on Hugging Face, eliminating the need for users to compile them on their own. Each model is accompanied by detailed configurations. These models have undergone extensive testing on various devices, and their kernel performance has been fine-tuned by developers with the help of TVM.

.. list-table:: Supported Models
  :widths: 15 15 15 15
  :header-rows: 1

  * - Model code
    - Model Series
    - Quantization Mode
    - Hugging Face repo
  * - `vicuna-v1-7b-q4f32_0`
    - `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
    - * Weight storage data type: int4
      * running data type: float32
      * symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q4f32_0>`__
  * - `vicuna-v1-7b-q4f16_0`
    - `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
    - * Weight storage data type: int4
      * running data type: float16
      * symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q4f16_0>`__
  * - `RedPajama-INCITE-Chat-3B-v1-q4f32_0`
    - `RedPajama <https://www.together.xyz/blog/redpajama>`__
    - * Weight storage data type: int4
      * running data type: float32
      * symmetric quantization 
    - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_0>`__
  * - `RedPajama-INCITE-Chat-3B-v1-q4f16_0`
    - `RedPajama <https://www.together.xyz/blog/redpajama>`__
    - * Weight storage data type: int4
      * running data type: float16
      * symmetric quantization 
    - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0>`__

You can see `MLC-LLM pull requests <https://github.com/mlc-ai/mlc-llm/pulls?q=is%3Aopen+is%3Apr+label%3Anew-models>`__ to check the ongoing efforts of new models.

Want to try different models?
-----------------------------

Please check :doc:`/tutorials/compile-models` on how to compile models with supported model architectures, and :doc:`/tutorials/bring-your-own-models` on how to bring a new LLM model architecture.

Contribute to MLC-LLM Model Zoo
-------------------------------

Awesome! Please check our :doc:`/contribute/community` on how to contribute to MLC-LLM.
