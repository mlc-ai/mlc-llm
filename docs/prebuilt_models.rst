.. _Model Prebuilts:

Model Prebuilts
==================

.. contents:: Table of Contents
    :depth: 3
    :local:

MLC-LLM is a universal solution for deploying different language models. Any language models that can be described in `TVM Relax <https://mlc.ai/chapter_graph_optimization/index.html>`__ (a general representation for Neural Networks and can be imported from models written in PyTorch) can be recognized by MLC-LLM and thus deployed to different backends with the help of :doc:`TVM Unity </install/tvm>`.

The community has already supported several LLM architectures (LLaMA, GPT-NeoX, etc.) and have prebuilt some models (Vicuna, RedPajama, etc.) which you can use off the shelf.
With the goal of democratizing the deployment of LLMs, we eagerly anticipate further contributions from the community to expand the range of supported model architectures.

This page contains the list of prebuilt models for our CLI (command line interface) app, iOS and Android apps.
The models have undergone extensive testing on various devices, and their performance has been optimized by developers with the help of TVM.

.. _prebuilt-models-cli:

Prebuilt Models for CLI
-----------------------

.. list-table::
  :widths: 15 15 15 15
  :header-rows: 1

  * - Model code
    - Model Series
    - Quantization Mode
    - Hugging Face repo
  * - `Llama-2-7b-q4f16_1`
    - `Llama <https://ai.meta.com/llama/>`__
    - * Weight storage data type: int4
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1>`__
  * - `vicuna-v1-7b-q3f16_0`
    - `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
    - * Weight storage data type: int3
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0>`__
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

To download and run one model with CLI, follow the instructions below:

.. code:: shell

  # Create conda environment and install CLI if you have not installed.
  conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-cli-nightly
  conda activate mlc-chat-venv
  conda install git git-lfs
  git lfs install

  # Download prebuilt model binary libraries from GitHub if you have not downloaded.
  mkdir -p dist/prebuilt
  git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib

  # Download prebuilt model weights and run CLI.
  cd dist/prebuilt
  git clone https://huggingface.co/mlc-ai/mlc-chat-[model-code]
  cd ../..
  mlc_chat_cli --local-id [model-code]

  # e.g.,
  # cd dist/prebuilt
  # git clone https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-7b-q8f16_0
  # cd ../..
  # mlc_chat_cli --local-id rwkv-raven-7b-q8f16_0


.. _prebuilt-models-ios:

Prebuilt Models for iOS
-----------------------

.. list-table:: Prebuilt models for iOS
  :widths: 15 15 15 15
  :header-rows: 1

  * - Model code
    - Model Series
    - Quantization Mode
    - Hugging Face repo
  * - `Llama-2-7b-q3f16_1`
    - `Llama <https://ai.meta.com/llama/>`__
    - * Weight storage data type: int3
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q3f16_1>`__
  * - `vicuna-v1-7b-q3f16_0`
    - `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
    - * Weight storage data type: int3
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0>`__
  * - `RedPajama-INCITE-Chat-3B-v1-q4f16_0`
    - `RedPajama <https://www.together.xyz/blog/redpajama>`__
    - * Weight storage data type: int4
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0>`__

The `downloadable iOS app <https://apps.apple.com/us/app/mlc-chat/id6448482937>`_ has builtin RedPajama-3B model support.
To add a model to the iOS app, follow the steps below:

.. collapse:: Click to show instructions

  .. tabs::

      .. tab:: Step 1

          Open "MLCChat" app, click "Add model variant".

          .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-1.png
              :align: center
              :width: 30%

      .. tab:: Step 2

          Paste the repository URL of the model built on your own, and click "Add".

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

The iOS app has integrated with the following model libraries, which can be directly reused when you want to run a model you compiled in iOS, as long as the model is in the supported model family and is compiled with supported quantization mode.
For example, if you compile `OpenLLaMA-7B <https://github.com/openlm-research/open_llama>`_ with quantization mode ``q3f16_0``, then you can run the compiled OpenLLaMA model on iPhone without rebuilding the iOS app by reusing the `vicuna-v1-7b-q3f16_0` model library. Please check the :doc:`model distribution page </compilation/distribute_compiled_models>` for detailed instructions.

.. list-table:: Prebuilt model libraries which are integrated in the iOS app
  :widths: 15 15 15
  :header-rows: 1

  * - Model library name
    - Model Family
    - Quantization Mode
  * - `vicuna-v1-7b-q3f16_0`
    - LLaMA
    - * Weight storage data type: int3
      * Running data type: float16
      * Symmetric quantization
  * - `RedPajama-INCITE-Chat-3B-v1-q4f16_0`
    - GPT-NeoX
    - * Weight storage data type: int4
      * Running data type: float16
      * Symmetric quantization


.. _prebuilt-models-android:

Prebuilt Models for Android
---------------------------

.. list-table:: Prebuilt models for Android
  :widths: 15 15 15 15
  :header-rows: 1

  * - Model code
    - Model Series
    - Quantization Mode
    - Hugging Face repo
  * - `vicuna-v1-7b-q4f16_1`
    - `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
    - * Weight storage data type: int4
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int4>`__
  * - `RedPajama-INCITE-Chat-3B-v1-q4f16_0`
    - `RedPajama <https://www.together.xyz/blog/redpajama>`__
    - * Weight storage data type: int4
      * Running data type: float16
      * Symmetric quantization
    - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0>`__

------------------

You can check `MLC-LLM pull requests <https://github.com/mlc-ai/mlc-llm/pulls?q=is%3Aopen+is%3Apr+label%3Anew-models>`__ to track the ongoing efforts of new models. We encourage users to upload their compiled models to Hugging Face and share with the community.

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
    - * `Llama-2 <https://ai.meta.com/llama/>`__
      * `Alpaca <https://github.com/tatsu-lab/stanford_alpaca>`__
      * `Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
      * `Guanaco <https://github.com/artidoro/qlora>`__
      * `OpenLLaMA <https://github.com/openlm-research/open_llama>`__
      * `Gorilla <https://huggingface.co/gorilla-llm/gorilla-7b-hf-delta-v0>`__
      * `WizardLM <https://github.com/nlpxucan/WizardLM>`__
      * `YuLan-Chat <https://github.com/RUC-GSAI/YuLan-Chat>`__
  * - ``gpt-neox``
    - `GPT-NeoX <https://github.com/EleutherAI/gpt-neox>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gpt_neox.py>`__
    - * `RedPajama <https://www.together.xyz/blog/redpajama>`__
      * `Dolly <https://github.com/databrickslabs/dolly>`__
      * `Pythia <https://huggingface.co/EleutherAI/pythia-1.4b>`__
  * - ``gptj``
    - `GPT-J <https://huggingface.co/EleutherAI/gpt-j-6b>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gptj.py>`__
    - * `MOSS <https://github.com/OpenLMLab/MOSS>`__
  * - ``rwkv``
    - `RWKV <https://github.com/BlinkDL/RWKV-LM>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/rwkv.py>`__
    - * `RWKV-raven <https://github.com/BlinkDL/RWKV-LM>`__
  * - ``minigpt``
    - `MiniGPT <https://huggingface.co/Vision-CAIR/MiniGPT-4>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/minigpt.py>`__
    -
  * - ``gpt_bigcode``
    - `GPTBigCode <https://huggingface.co/docs/transformers/model_doc/gpt_bigcode>`__
    - `Relax Code <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gpt_bigcode.py>`__
    - * `StarCoder <https://huggingface.co/bigcode/starcoder>`__
      * `WizardCoder <https://huggingface.co/WizardLM/WizardCoder-15B-V1.0>`__
      * `SantaCoder <https://huggingface.co/bigcode/gpt_bigcode-santacoder>`__

For models structured in these model architectures, you can check the :doc:`model compilation page </compilation/compile_models>` on how to compile models.
Please `create a new issue <https://github.com/mlc-ai/mlc-llm/issues/new/choose>`_ if you want to request a new model architecture.
Our tutorial :doc:`Define New Models </tutorials/customize/define_new_models>` introduces how to bring a new model architecture to MLC-LLM.

.. _contribute-models-to-mlc-llm:

Contribute Models to MLC-LLM
----------------------------

Ready to contribute your compiled models/new model architectures? Awesome! Please check :ref:`contribute-new-models` on how to contribute new models to MLC-LLM.
