Model Prebuilts from Old Flow (Deprecated)
==========================================

**This page records the model libraries weights compiled under the old workflow (non-SLM).**

**We will remove this page soon.**

.. contents:: Table of Contents
    :depth: 3
    :local:

Overview
--------

MLC-LLM is a universal solution for deploying different language models. Any models that can be described in `TVM Relax <https://mlc.ai/chapter_graph_optimization/index.html>`__ 
(a general representation for Neural Networks and can be imported from models written in PyTorch) can be recognized by MLC-LLM and thus deployed to different backends with the 
help of :doc:`TVM Unity </install/tvm>`.

There are two ways to run a model on MLC-LLM:

1. Compile your own models following :doc:`the model compilation page </compilation/compile_models>`.
2. Use off-the-shelf prebuilts models following this current page.

This page focuses on the second option:

- Documenting :ref:`how to use prebuilts <deprecated-using-model-prebuilts>` for various platforms, and
- Tracking what current :ref:`prebuilt models we provide <deprecated-supported-model-architectures>`.

Prerequisite: Model Libraries and Compiled Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to run a specific model on MLC-LLM, you need:

**1. A model library:** a binary file containing the end-to-end functionality to inference a model (e.g. ``Llama-2-7b-chat-hf-q4f16_1-cuda.so``). See the full list of all precompiled model libraries `here <https://github.com/mlc-ai/binary-mlc-llm-libs>`__.

**2. Compiled weights:** a folder containing multiple files that store the compiled and quantized weights of a model (e.g. https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1).  See the full list of all precompiled weights `here <https://huggingface.co/mlc-ai>`__.

.. _deprecated-using-model-prebuilts:

Using Prebuilt Models for Different Platforms
---------------------------------------------

We quickly go over how to use prebuilt models for each platform. You can find detailed instruction on each platform's corresponding page.

.. _deprecated-using-prebuilt-models-cli:


Prebuilt Models on CLI / Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more, please see :doc:`the CLI page </deploy/cli>`, and the :doc:`the Python page </deploy/python>`.

.. collapse:: Click to show details

  First create the conda environment if you have not done so.

    .. code:: shell

      conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-cli-nightly
      conda activate mlc-chat-venv
      conda install git git-lfs
      git lfs install

  Download the prebuilt model libraries from github.

    .. code:: shell

      mkdir -p dist/prebuilt
      git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib

  Download the prebuilt model weights from hugging face for the model variant you want.

    .. code:: shell

      # Say we want to run rwkv-raven-7b-q8f16_0
      cd dist/prebuilt
      git clone https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-7b-q8f16_0
      cd ../..

      # The format being:
      # cd dist/prebuilt
      # git clone https://huggingface.co/mlc-ai/mlc-chat-[model-code]
      # cd ../..
      # mlc_chat_cli --model [model-code]

  Run the model with CLI:

    .. code:: shell

      # For CLI
      mlc_chat_cli --model rwkv-raven-7b-q8f16_0

  To run the model with Python API, see :doc:`the Python page </deploy/python>` (all other downloading steps are the same as CLI).


.. for a blank line

|

.. _deprecated-using-prebuilt-models-ios:

Prebuilt Models on iOS
^^^^^^^^^^^^^^^^^^^^^^

For more, please see :doc:`the iOS page </deploy/ios>`.

.. collapse:: Click to show details

  The `iOS app <https://apps.apple.com/us/app/mlc-chat/id6448482937>`_ has builtin RedPajama-3B and Llama-2-7b support. 

  All prebuilt models with an entry in ``iOS`` in the :ref:`model library table <deprecated-model-library-tables>` are supported by iOS. Namely, we have:

  .. list-table:: Prebuilt model libraries integrated in the iOS app
    :widths: 15 15 15
    :header-rows: 1

    * - Model library name
      - Model Family
      - Quantization Mode
    * - `Llama-2-7b-chat-hf-q3f16_1`
      - LLaMA
      - * Weight storage data type: int3
        * Running data type: float16
        * Symmetric quantization
    * - `vicuna-v1-7b-q3f16_0`
      - LLaMA
      - * Weight storage data type: int3
        * Running data type: float16
        * Symmetric quantization
    * - `RedPajama-INCITE-Chat-3B-v1-q4f16_1`
      - GPT-NeoX
      - * Weight storage data type: int4
        * Running data type: float16
        * Symmetric quantization

  As for prebuilt model weights, the ones we have integrated into app are listed below:

  .. list-table:: Tested prebuilt model weights for iOS
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
    * - `RedPajama-INCITE-Chat-3B-v1-q4f16_1`
      - `RedPajama <https://www.together.xyz/blog/redpajama>`__
      - * Weight storage data type: int4
        * Running data type: float16
        * Symmetric quantization
      - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1>`__
  
  To run a model variant you compiled on your own, you can directly reuse the above
  integrated prebuilt model libraries, as long as the model shares the
  architecture and is compiled with the same quantization mode.
  For example, if you compile `OpenLLaMA-7B <https://github.com/openlm-research/open_llama>`_
  with quantization mode ``q3f16_0``, then you can run the compiled OpenLLaMA model on iPhone
  without rebuilding the iOS app by reusing the `vicuna-v1-7b-q3f16_0` model library.
  Then you can upload the compiled weights to hugging face so that you can download
  the weights in the app as shown below (for more on uploading to hugging face,
  please check :ref:`distribute-compiled-models`).
  
  To add a model to the iOS app, follow the steps below:

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

.. _deprecated-prebuilt-models-android:

Prebuilt Models on Android
^^^^^^^^^^^^^^^^^^^^^^^^^^

For more, please see :doc:`the Android page </deploy/android>`.

.. collapse:: Click to show details

  The apk for demo Android app includes the following models. To add more, check out the Android page.

  .. list-table:: Prebuilt Models for Android
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
    * - `RedPajama-INCITE-Chat-3B-v1-q4f16_1`
      - `RedPajama <https://www.together.xyz/blog/redpajama>`__
      - * Weight storage data type: int4
        * Running data type: float16
        * Symmetric quantization
      - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1>`__
.. for a blank line

|

.. _deprecated-supported-model-architectures:

Level 1: Supported Model Architectures (The All-In-One Table)
-------------------------------------------------------------

For each model architecture (e.g. Llama), there are multiple variants (e.g. CodeLlama, WizardLM). The variants share the same code for inference and only differ in their weights. In other words, running CodeLlama and WizardLM can use the same model library file (specified in Level 2 tables), but different precompiled weights (specified in Level 3 tables). Note that we have not provided prebuilt weights for all model variants.

Each entry below hyperlinks to the corresponding level 2 and level 3 tables.

MLC-LLM supports the following model architectures:

.. list-table:: Supported Model Architectures
  :widths: 10 10 15 15
  :header-rows: 1

  * - Model Architecture
    - Support
    - Available MLC Prebuilts
    - Unavailable in MLC Prebuilts
  * - `LLaMA <https://github.com/facebookresearch/llama>`__
    - * :ref:`Prebuilt Model Library <deprecated-llama_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/llama.py>`__
    - * :ref:`Llama-2 <deprecated-llama2_variant_table>`
      * :ref:`Code Llama <deprecated-code_llama_variant_table>`
      * :ref:`Vicuna <deprecated-vicuna_variant_table>`
      * :ref:`WizardLM <deprecated-WizardLM_variant_table>` 
      * :ref:`WizardMath <deprecated-wizard_math_variant_table>`
      * :ref:`OpenOrca Platypus2 <deprecated-open_orca_variant_table>`
      * :ref:`FlagAlpha Llama-2 Chinese <deprecated-flag_alpha_llama2_variant_table>` 
      * :ref:`georgesung Llama-2 Uncensored <deprecated-llama2_uncensored_variant_table>`
    - * `Alpaca <https://github.com/tatsu-lab/stanford_alpaca>`__
      * `Guanaco <https://github.com/artidoro/qlora>`__
      * `OpenLLaMA <https://github.com/openlm-research/open_llama>`__
      * `Gorilla <https://huggingface.co/gorilla-llm/gorilla-7b-hf-delta-v0>`__
      * `YuLan-Chat <https://github.com/RUC-GSAI/YuLan-Chat>`__
      * `WizardCoder (new) <https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder>`__
  * - `GPT-NeoX <https://github.com/EleutherAI/gpt-neox>`__
    - * :ref:`Prebuilt Model Library <deprecated-gpt_neox_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gpt_neox.py>`__
    - * :ref:`RedPajama <deprecated-red_pajama_variant_table>` 
    - * `Dolly <https://github.com/databrickslabs/dolly>`__
      * `Pythia <https://huggingface.co/EleutherAI/pythia-1.4b>`__
      * `StableCode <https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b>`__
  * - `GPT-J <https://huggingface.co/EleutherAI/gpt-j-6b>`__
    - * Prebuilt not compiled yet
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gptj.py>`__
    - 
    - * `MOSS <https://github.com/OpenLMLab/MOSS>`__
  * - `RWKV <https://github.com/BlinkDL/RWKV-LM>`__
    - * :ref:`Prebuilt Model Library <deprecated-rwkv_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/rwkv.py>`__
    - * :ref:`RWKV-raven <deprecated-rwkv_raven_variant_table>` 
    - 
  * - `MiniGPT <https://huggingface.co/Vision-CAIR/MiniGPT-4>`__
    - * Prebuilt not compiled yet
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/minigpt.py>`__
    - 
    - * `MiniGPT-4 <https://huggingface.co/Vision-CAIR/MiniGPT-4>`__
  * - `GPTBigCode <https://huggingface.co/docs/transformers/model_doc/gpt_bigcode>`__
    - * :ref:`Prebuilt Model Library <deprecated-gpt_big_code_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/gpt_bigcode.py>`__
    - * :ref:`WizardCoder (old) <deprecated-wizard_coder_variant_table>` 
    - * `StarCoder <https://huggingface.co/bigcode/starcoder>`__
      * `SantaCoder <https://huggingface.co/bigcode/gpt_bigcode-santacoder>`__
  * - `ChatGLM <https://github.com/THUDM/ChatGLM-6B/blob/main/README_en.md>`__
    - * Prebuilt not compiled yet
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/chatglm.py>`__
    - 
    - * `ChatGLM2 <https://huggingface.co/THUDM/chatglm2-6b>`__
      * `CodeGeeX2 <https://huggingface.co/THUDM/codegeex2-6b>`__
  * - `StableLM <https://huggingface.co/stabilityai>`__
    - * Prebuilt not compiled yet
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/relax_model/stablelm_3b.py>`__
    - 
    - * `StableLM <https://huggingface.co/collections/stabilityai/stable-lm-650852cfd55dd4e15cdcb30a>`__

If the model variant you are interested in uses one of these model architectures we support,
(but we have not provided the prebuilt weights yet), you can check out 
:doc:`/compilation/convert_weights` and :doc:`/compilation/compile_models` on how to compile your own models.
Afterwards, you may follow :ref:`distribute-compiled-models` to upload your prebuilt
weights to hugging face, and submit a PR that adds an entry to this page,
contributing to the community.

For models structured in an architecture we have not supported yet, you could:

- Either `create a [Model Request] issue <https://github.com/mlc-ai/mlc-llm/issues/new?assignees=&labels=new-models&projects=&template=model-request.md&title=%5BModel+Request%5D+>`__ which automatically shows up on our `Model Request Tracking Board <https://github.com/orgs/mlc-ai/projects/2>`__.

- Or follow our tutorial :doc:`Define New Models </compilation/define_new_models>`, which introduces how to bring a new model architecture to MLC-LLM.


.. _deprecated-model-library-tables:

Level 2: Model Library Tables (Precompiled Binary Files)
--------------------------------------------------------

As mentioned earlier, each model architecture corresponds to a different model library file. That is, you cannot use the same model library file to run ``RedPajama`` and ``Llama-2``. However, you can use the same ``Llama`` model library file to run ``Llama-2``, ``WizardLM``, ``CodeLlama``, etc, but just with different weight files (from tables in Level 3).

Each table below demonstrates the pre-compiled model library files for each model architecture. This is categorized by:

- **Size**: each size of model has its own distinct model library file (e.g. 7B or 13B number of parameters)

- **Platform**: the backend that the model library is intended to be run on (e.g. CUDA, ROCm, iphone, etc.)

- **Quantization scheme**: the model library file also differs due to the quantization scheme used. For more on this, please see the :doc:`model compilation page </compilation/compile_models>` (e.g. ``q3f16_1`` vs. ``q4f16_1``)

Each entry links to the specific model library file found in `this github repo <https://github.com/mlc-ai/binary-mlc-llm-libs>`__.

.. _deprecated-llama_library_table:

Llama
^^^^^
.. list-table:: Llama
  :widths: 8 8 8 8 8 8 8 8 8 8
  :header-rows: 1
  :stub-columns: 1

  * -
    - CUDA
    - ROCm
    - Vulkan

      (Linux)
    - Vulkan

      (Windows)
    - Metal

      (M1/M2)
    - Metal

      (Intel)
    - iOS
    - webgpu
    - mali
  * - 7B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-cuda.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-rocm.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-vulkan.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-vulkan.dll>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-metal.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-metal_x86_64.dylib>`__
    - `q3f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q3f16_1-iphone.tar>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-webgpu.wasm>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f32_1-webgpu.wasm>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf-q4f16_1-mali.so>`__
  * - 13B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-cuda.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-rocm.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-vulkan.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-vulkan.dll>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-metal.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-metal_x86_64.dylib>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-webgpu.wasm>`__
    
      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f32_1-webgpu.wasm>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf-q4f16_1-mali.so>`__
  * - 34B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/CodeLlama-34b-hf-q4f16_1-cuda.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/CodeLlama-34b-hf-q4f16_1-vulkan.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/CodeLlama-34b-hf-q4f16_1-vulkan.dll>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/CodeLlama-34b-hf-q4f16_1-metal.so>`__
    - 
    - 
    - 
    - 
  * - 70B
    - 
    - 
    - 
    - 
    - `q3f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-70b-chat-hf-q3f16_1-metal.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-70b-chat-hf-q4f16_1-metal.so>`__
    - 
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-70b-chat-hf-q4f16_1-webgpu.wasm>`__
    - 

.. _deprecated-gpt_neox_library_table:
  
GPT-NeoX (RedPajama-INCITE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table:: GPT-NeoX (RedPajama-INCITE)
  :widths: 8 8 8 8 8 8 8 8 8 8
  :header-rows: 1
  :stub-columns: 1

  * -
    - CUDA
    - ROCm
    - Vulkan

      (Linux)
    - Vulkan

      (Windows)
    - Metal

      (M1/M2)
    - Metal

      (Intel)
    - iOS
    - webgpu
    - mali
  * - 3B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-rocm.so>`__
    - `q4f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-vulkan.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.so>`__
    - `q4f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-vulkan.dll>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.dll>`__
    - `q4f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-metal.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal.so>`__
    - `q4f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-metal_x86_64.dylib>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal_x86_64.dylib>`__
    - `q4f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-iphone.tar>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar>`__
    - `q4f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-webgpu-v1.wasm>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm>`__

      `q4f32_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f32_0-webgpu-v1.wasm>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f32_1-webgpu.wasm>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-mali.so>`__

.. _deprecated-rwkv_library_table:

RWKV
^^^^
.. list-table:: RWKV
  :widths: 8 8 8 8 8 8 8 8 8 8
  :header-rows: 1
  :stub-columns: 1

  * -
    - CUDA
    - ROCm
    - Vulkan

      (Linux)
    - Vulkan

      (Windows)
    - Metal

      (M1/M2)
    - Metal

      (Intel)
    - iOS
    - webgpu
    - mali
  * - 1B5
    -
    -
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-1b5-q8f16_0-vulkan.so>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-1b5-q8f16_0-vulkan.dll>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-1b5-q8f16_0-metal.so>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-1b5-q8f16_0-metal_x86_64.dylib>`__
    -
    -
    -
  * - 3B
    -
    -
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-3b-q8f16_0-vulkan.so>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-3b-q8f16_0-vulkan.dll>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-3b-q8f16_0-metal.so>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-3b-q8f16_0-metal_x86_64.dylib>`__
    -
    -
    -
  * - 7B
    -
    -
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-7b-q8f16_0-vulkan.so>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-7b-q8f16_0-vulkan.dll>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-7b-q8f16_0-metal.so>`__
    - `q8f16_0 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/rwkv-raven-7b-q8f16_0-metal_x86_64.dylib>`__
    -
    -
    -

.. _deprecated-gpt_big_code_library_table:

GPTBigCode
^^^^^^^^^^
Note that these all links to model libraries for WizardCoder (the older version released in Jun. 2023). 
However, any GPTBigCode model variants should be able to reuse these (e.g. StarCoder, SantaCoder).

.. list-table:: GPTBigCode
  :widths: 8 8 8 8 8 8 8 8 8 8
  :header-rows: 1
  :stub-columns: 1

  * -
    - CUDA
    - ROCm
    - Vulkan

      (Linux)
    - Vulkan

      (Windows)
    - Metal

      (M1/M2)
    - Metal

      (Intel)
    - iOS
    - webgpu
    - mali
  * - 15B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f16_1-cuda.so>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f32_1-cuda.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f16_1-vulkan.so>`__
      
      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f32_1-vulkan.so>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f16_1-vulkan.dll>`__
    
      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f32_1-vulkan.dll>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f16_1-metal.so>`__
    - 
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f16_1-webgpu.wasm>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/WizardCoder-15B-V1.0-q4f32_1-webgpu.wasm>`__
    - 
  
.. _deprecated-model-variant-tables:

Level 3: Model Variant Tables (Precompiled Weights)
---------------------------------------------------

Finally, for each model variant, we provide the precompiled weights we uploaded to hugging face.

Each precompiled weight is categorized by its model size (e.g. 7B vs. 13B) and the quantization scheme (e.g. ``q3f16_1`` vs. ``q4f16_1``). We note that the weights are **platform-agnostic**.

Each model variant also loads its conversation configuration from a pre-defined :ref:`conversation template<load-predefined-conv-template>`. Note that multiple model variants can share a common conversation template.

Some of these files are uploaded by our community contributors--thank you!

.. _deprecated-llama2_variant_table:

`Llama-2 <https://ai.meta.com/llama/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``llama-2``

.. list-table:: Llama-2
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q3f16_1 <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q3f16_1>`__
      * `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1>`__
      * `q4f32_1 <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f32_1>`__

  * - 13B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f16_1>`__
      * `q4f32_1 <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f32_1>`__

  * - 70B
    - * `q3f16_1 <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-70b-chat-hf-q3f16_1>`__
      * `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-Llama-2-70b-chat-hf-q4f16_1>`__

.. _deprecated-code_llama_variant_table:

`Code Llama <https://about.fb.com/news/2023/08/code-llama-ai-for-coding/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``codellama_completion``

.. list-table:: Code Llama
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 (Base) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-7b-hf-q4f16_1>`__
      * `q4f16_1 (Instruct) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-7b-Instruct-hf-q4f16_1>`__
      * `q4f16_1 (Python) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-7b-Python-hf-q4f16_1>`__

  * - 13B
    - * `q4f16_1 (Base) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-13b-hf-q4f16_1>`__
      * `q4f16_1 (Instruct) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-13b-Instruct-hf-q4f16_1>`__
      * `q4f16_1 (Python) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-13b-Python-hf-q4f16_1>`__

  * - 34B
    - * `q4f16_1 (Base) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-34b-hf-q4f16_1>`__
      * `q4f16_1 (Instruct) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-34b-Instruct-hf-q4f16_1>`__
      * `q4f16_1 (Python) <https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-34b-Python-hf-q4f16_1>`__


.. _deprecated-vicuna_variant_table:

`Vicuna <https://lmsys.org/blog/2023-03-30-vicuna/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``vicuna_v1.1``

.. list-table:: Vicuna
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q3f16_0 <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0>`__
      * `q4f32_0 <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q4f32_0>`__
      * `int3 (demo) <https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int3>`__
      * `int4 (demo) <https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int4>`__


.. _deprecated-WizardLM_variant_table:

`WizardLM <https://github.com/nlpxucan/WizardLM>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``vicuna_v1.1``

.. list-table:: WizardLM
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 13B
    - * `q4f16_1 (V1.2) <https://huggingface.co/mlc-ai/mlc-chat-WizardLM-13B-V1.2-q4f16_1>`__
      * `q4f32_1 (V1.2) <https://huggingface.co/mlc-ai/mlc-chat-WizardLM-13B-V1.2-q4f32_1>`__

  * - 70B
    - * `q3f16_1 (V1.0) <https://huggingface.co/mlc-ai/mlc-chat-WizardLM-70B-V1.0-q3f16_1>`__
      * `q4f16_1 (V1.0) <https://huggingface.co/mlc-ai/mlc-chat-WizardLM-70B-V1.0-q4f16_1>`__


.. _deprecated-wizard_math_variant_table:

`WizardMath <https://github.com/nlpxucan/WizardLM/tree/main/WizardMath>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``wizard_coder_or_math``

.. list-table:: WizardMath
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-WizardMath-7B-V1.0-q4f16_1>`__
      * `q4f32_1 <https://huggingface.co/mlc-ai/mlc-chat-WizardMath-7B-V1.0-q4f32_1>`__
  * - 13B
    - `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-WizardMath-13B-V1.0-q4f16_1>`__
  * - 70B
    - `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-WizardMath-70B-V1.0-q4f16_1>`__


.. _deprecated-open_orca_variant_table:

`OpenOrca Platypus2 <https://huggingface.co/Open-Orca/OpenOrca-Platypus2-13B>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``llama-2``

.. list-table:: OpenOrca Platypus2
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 13B
    - `q4f16_1 <https://huggingface.co/DavidSharma/mlc-chat-OpenOrca-Platypus2-13B-q4f16_1>`__


.. _deprecated-flag_alpha_llama2_variant_table:

`FlagAlpha Llama-2 Chinese <https://github.com/FlagAlpha/Llama2-Chinese>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``llama-2``

.. list-table:: FlagAlpha Llama-2 Chinese
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-FlagAlpha-Llama2-Chinese-7b-Chat-q4f16_1>`__
      * `q4f32_1 <https://huggingface.co/mlc-ai/mlc-chat-FlagAlpha-Llama2-Chinese-7b-Chat-q4f32_1>`__


.. _deprecated-llama2_uncensored_variant_table:

`Llama2 uncensored (georgesung) <https://huggingface.co/georgesung/llama2_7b_chat_uncensored>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``llama-default``

.. list-table:: Llama2 uncensored
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-georgesung-llama2-7b-chat-uncensored-q4f16_1>`__
      * `q4f32_1 <https://huggingface.co/mlc-ai/mlc-chat-georgesung-llama2-7b-chat-uncensored-q4f32_1>`__

.. _deprecated-red_pajama_variant_table:

`RedPajama <https://www.together.xyz/blog/redpajama>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``LM``

.. list-table:: Red Pajama
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 3B
    - * `q4f16_0 (Instruct) <https://huggingface.co/mlc-ai/RedPajama-INCITE-Instruct-3B-v1-q4f16_0>`__
      * `q4f16_0 (Chat) <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0>`__
      * `q4f16_1 (Chat) <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1>`__
      * `q4f32_0 (Chat) <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_0>`__


.. _deprecated-rwkv_raven_variant_table:

`RWKV-raven <https://github.com/BlinkDL/RWKV-LM>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``rwkv``

.. list-table:: RWKV-raven
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 1B5
    - `q8f16_0 <https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-1b5-q8f16_0>`__

  * - 3B
    - `q8f16_0 <https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-3b-q8f16_0>`__

  * - 7B
    - `q8f16_0 <https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-7b-q8f16_0>`__


.. _deprecated-wizard_coder_variant_table:

`WizardCoder <https://github.com/nlpxucan/WizardLM>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``wizard_coder_or_math``

.. list-table:: WizardCoder
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 15B
    - `q4f16_1 <https://huggingface.co/mlc-ai/mlc-chat-WizardCoder-15B-V1.0-q4f16_1>`__

------------------


.. _deprecated-contribute-models-to-mlc-llm:

Contribute Models to MLC-LLM
----------------------------

Ready to contribute your compiled models/new model architectures? Awesome! Please check :ref:`contribute-new-models` on how to contribute new models to MLC-LLM.
