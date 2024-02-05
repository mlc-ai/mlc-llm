.. _Model Prebuilts:

Model Prebuilts
==================

.. contents:: Table of Contents
    :depth: 3
    :local:

.. _model-prebuilts-overview:

Overview
--------

MLC-LLM is a universal solution for deploying different language models. Any models that can be described in `TVM Relax <https://mlc.ai/chapter_graph_optimization/index.html>`__ 
(a general representation for Neural Networks and can be imported from models written in PyTorch) can be recognized by MLC-LLM and thus deployed to different backends with the 
help of :doc:`TVM Unity </install/tvm>`.

There are two ways to run a model on MLC-LLM (this page focuses on the second one):

1. Compile your own models following :doc:`the model compilation page </compilation/compile_models>`.
2. Use off-the-shelf prebuilt models following this current page.

In order to run a specific model on MLC-LLM, you need:

**1. A model library:** a binary file containing the end-to-end functionality to inference a model (e.g. ``Llama-2-7b-chat-hf-q4f16_1-cuda.so``).
See the full list of all precompiled model libraries `here <https://github.com/mlc-ai/binary-mlc-llm-libs>`__.

**2. Compiled weights:** a folder containing multiple files that store the compiled and quantized weights of a model
(e.g. https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC).  See the full list of all precompiled weights `here <https://huggingface.co/mlc-ai>`__.

In this page, we first quickly go over :ref:`how to use prebuilts <using-model-prebuilts>` for different platforms,
then track what current :ref:`prebuilt models we provide <supported-model-architectures>`.


.. _using-model-prebuilts:

Using Prebuilt Models for Different Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We quickly go over how to use prebuilt models for each platform. You can find detailed instruction on each platform's corresponding page.

.. _using-prebuilt-models-cli:

**Prebuilt Models on CLI / Python**

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

      mkdir dist/
      git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs

  Run the model with CLI:

    .. code:: shell

      mlc_chat chat HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC


  To run the model with Python API, see :doc:`the Python page </deploy/python>` (all other downloading steps are the same as CLI).


.. for a blank line

|

.. _using-prebuilt-models-ios:

**Prebuilt Models on iOS**

For more, please see :doc:`the iOS page </deploy/ios>`.

.. collapse:: Click to show details

  The `iOS app <https://apps.apple.com/us/app/mlc-chat/id6448482937>`_ has builtin RedPajama-3B and Mistral-7B-Instruct-v0.2 support. 

  All prebuilt models with an entry in ``iOS`` in the :ref:`model library table <model-library-tables>` are supported by iOS. Namely, we have:

  .. list-table:: Prebuilt Models for iOS
    :widths: 15 15 15 15
    :header-rows: 1

    * - Model Code
      - Model Series
      - Quantization Mode
      - MLC HuggingFace Weights Repo
    * - `Mistral-7B-Instruct-v0.2-q3f16_1`
      - `Mistral <https://mistral.ai/>`__
      - * Weight storage data type: int3
        * Running data type: float16
        * Symmetric quantization
      - `link <https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC>`__
    * - `RedPajama-INCITE-Chat-3B-v1-q4f16_1`
      - `RedPajama <https://github.com/togethercomputer/RedPajama-Data>`__
      - * Weight storage data type: int4
        * Running data type: float16
        * Symmetric quantization
      - `link <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC>`__
    * - `phi-2-q4f16_1`
      - `Microsoft Phi-2 <https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/>`__
      - * Weight storage data type: int4
        * Running data type: float16
        * Symmetric quantization
      - `link <https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC>`__
.. for a blank line

|

.. _prebuilt-models-android:

**Prebuilt Models on Android**

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
      - `link <https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC>`__
    * - `RedPajama-INCITE-Chat-3B-v1-q4f16_1`
      - `RedPajama <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC>`__
      - * Weight storage data type: int4
        * Running data type: float16
        * Symmetric quantization
      - `link <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1>`__
.. for a blank line

|

.. _supported-model-architectures:

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
    - * :ref:`Prebuilt Model Library <llama_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_chat/model/llama>`__
    - * :ref:`Llama-2-chat <llama2_variant_table>`
    - * `Code Llama <https://huggingface.co/codellama>`__
      * `Vicuna <https://huggingface.co/lmsys/vicuna-7b-v1.5>`__
      * `WizardLM <https://github.com/nlpxucan/WizardLM/tree/main/WizardLM>`__
      * `WizardCoder (new) <https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder>`__
      * `OpenOrca Platypus2 <https://huggingface.co/Open-Orca/OpenOrca-Platypus2-13B>`__
      * `FlagAlpha Llama-2 Chinese <https://github.com/FlagAlpha/Llama2-Chinese>`__
      * `georgesung Llama-2 Uncensored <https://huggingface.co/georgesung/llama2_7b_chat_uncensored>`__
      * `Alpaca <https://github.com/tatsu-lab/stanford_alpaca>`__
      * `Guanaco <https://github.com/artidoro/qlora>`__
      * `OpenLLaMA <https://github.com/openlm-research/open_llama>`__
      * `Gorilla <https://huggingface.co/gorilla-llm/gorilla-7b-hf-delta-v0>`__
      * `YuLan-Chat <https://github.com/RUC-GSAI/YuLan-Chat>`__
  * - `Mistral <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__
    - * :ref:`Prebuilt Model Library <mistral_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_chat/model/mistral>`__
    - * :ref:`Mistral-7B-Instruct-v0.2 <mistralInstruct_variant_table>`
      * :ref:`NeuralHermes-2.5-Mistral-7B <neuralHermes_variant_table>`
      * :ref:`OpenHermes-2.5-Mistral-7B <openHermes_variant_table>`
      * :ref:`WizardMath-7B-V1.1 <wizardMathV1.1_variant_table>`
    - 
  * - `GPT-NeoX <https://github.com/EleutherAI/gpt-neox>`__
    - * :ref:`Prebuilt Model Library <gpt_neox_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_chat/model/gpt_neox>`__
    - * :ref:`RedPajama <red_pajama_variant_table>` 
    - * `Dolly <https://github.com/databrickslabs/dolly>`__
      * `Pythia <https://huggingface.co/EleutherAI/pythia-1.4b>`__
      * `StableCode <https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b>`__
  * - `GPTBigCode <https://huggingface.co/docs/transformers/model_doc/gpt_bigcode>`__
    - * :ref:`Prebuilt Model Library <gpt_big_code_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_chat/model/gpt_bigcode>`__
    - 
    - * `StarCoder <https://huggingface.co/bigcode/starcoder>`__
      * `SantaCoder <https://huggingface.co/bigcode/gpt_bigcode-santacoder>`__
      * `WizardCoder (old) <https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder>`__
  * - `Phi <https://huggingface.co/microsoft/phi-2>`__
    - * :ref:`Prebuilt Model Library <phi_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_chat/model/phi>`__
    - * :ref:`Phi-1_5 <phi_variant_table>`
      * :ref:`Phi-2 <phi_variant_table>`
    - 
  * - `GPT2  <https://huggingface.co/docs/transformers/model_doc/gpt2>`__
    - * :ref:`Prebuilt Model Library <gpt2_library_table>`
      * `MLC Implementation <https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_chat/model/gpt2>`__
    - * :ref:`GPT2 <gpt2_variant_table>`
    - 

If the model variant you are interested in uses one of these model architectures we support,
(but we have not provided the prebuilt weights yet), you can check out 
:doc:`/compilation/convert_weights` on how to convert the weights.
Afterwards, you may follow :ref:`distribute-compiled-models` to upload your prebuilt
weights to hugging face, and submit a PR that adds an entry to this page,
contributing to the community.

For models structured in an architecture we have not supported yet, you could:

- Either `create a [Model Request] issue <https://github.com/mlc-ai/mlc-llm/issues/new?assignees=&labels=new-models&projects=&template=model-request.md&title=%5BModel+Request%5D+>`__ which
  automatically shows up on our `Model Request Tracking Board <https://github.com/orgs/mlc-ai/projects/2>`__.

- Or follow our tutorial :doc:`Define New Models </compilation/define_new_models>`, which introduces how to bring a new model architecture to MLC-LLM.


.. _model-library-tables:

Level 2: Model Library Tables (Precompiled Binary Files)
--------------------------------------------------------

As mentioned earlier, each model architecture corresponds to a different model library file. That is, you cannot use the same model library file to run ``RedPajama`` and ``Llama-2``. However, you can use the same ``Llama`` model library file to run ``Llama-2``, ``WizardLM``, ``CodeLlama``, etc, but just with different weight files (from tables in Level 3).

Each table below demonstrates the pre-compiled model library files for each model architecture. This is categorized by:

- **Size**: each size of model has its own distinct model library file (e.g. 7B or 13B number of parameters)

- **Platform**: the backend that the model library is intended to be run on (e.g. CUDA, ROCm, iphone, etc.)

- **Quantization scheme**: the model library file also differs due to the quantization scheme used. For more on this, please see the :doc:`quantization page </compilation/configure_quantization>`
  (e.g. ``q3f16_1`` vs. ``q4f16_1``).

Each entry links to the specific model library file found in `this github repo <https://github.com/mlc-ai/binary-mlc-llm-libs>`__.

If the model library you found is not available as a prebuilt, you can compile it yourself by following :doc:`the model compilation page </compilation/compile_models>`,
and submit a PR to the repo `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__ afterwards.

.. _llama_library_table:

Llama
^^^^^
.. list-table:: Llama
  :widths: 8 8 8 8 8 8 8 8 8 8 8
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

      (M Chip)
    - Metal

      (Intel)
    - iOS
    - Android
    - webgpu
    - mali
  * - 7B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-cuda.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-vulkan.so>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-vulkan.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-metal.so>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-metal.so>`__
    - 
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-android.tar>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-android.tar>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-webgpu.wasm>`__
    - 
  * - 13B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf/Llama-2-13b-chat-hf-q4f16_1-cuda.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf/Llama-2-13b-chat-hf-q4f16_1-vulkan.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf/Llama-2-13b-chat-hf-q4f16_1-metal.so>`__
    - 
    - 
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-13b-chat-hf/Llama-2-13b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm>`__
    - 
  * - 34B
    - 
    - 
    - 
    - 
    - 
    - 
    - 
    - 
    - 
    -
  * - 70B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-70b-chat-hf/Llama-2-70b-chat-hf-q4f16_1-cuda.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-70b-chat-hf/Llama-2-70b-chat-hf-q4f16_1-vulkan.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-70b-chat-hf/Llama-2-70b-chat-hf-q4f16_1-metal.so>`__
    - 
    - 
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Llama-2-70b-chat-hf/Llama-2-70b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm>`__
    - 

.. _mistral_library_table:
  
Mistral
^^^^^^^
.. list-table:: Mistral
  :widths: 8 8 8 8 8 8 8 8 8 8 8
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

      (M Chip)
    - Metal

      (Intel)
    - iOS
    - Android
    - webgpu
    - mali
  * - 7B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-cuda.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-vulkan.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-metal.so>`__
    - 
    - `q3f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q3f16_1-iphone.tar>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-android.tar>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm>`__
    -


.. _gpt_neox_library_table:
  
GPT-NeoX (RedPajama-INCITE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table:: GPT-NeoX (RedPajama-INCITE)
  :widths: 8 8 8 8 8 8 8 8 8 8 8
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

      (M Chip)
    - Metal

      (Intel)
    - iOS
    - Android
    - webgpu
    - mali
  * - 3B
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so>`__
  
      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f32_1-cuda.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.so>`__
  
      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f32_1-vulkan.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal.so>`__
  
      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f32_1-metal.so>`__
    - 
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f16_1-android.tar>`__

      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f32_1-android.tar>`__
    - `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx2k-webgpu.wasm>`__
  
      `q4f32_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx2k-webgpu.wasm>`__
    -

.. _gpt_big_code_library_table:

GPTBigCode
^^^^^^^^^^

.. list-table:: GPTBigCode
  :widths: 8 8 8 8 8 8 8 8 8 8 8
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

      (M Chip)
    - Metal

      (Intel)
    - iOS
    - Android
    - webgpu
    - mali
  * - 15B
    - 
    - 
    - 
    - 
    - 
    - 
    - 
    - 
    - 
    - 

.. _phi_library_table:
  
Phi
^^^
.. list-table:: Phi
  :widths: 8 8 8 8 8 8 8 8 8 8 8
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

      (M Chip)
    - Metal

      (Intel)
    - iOS
    - Android
    - webgpu
    - mali
  * - Phi-2
   
      (2.7B)
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q0f16-cuda.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q4f16_1-cuda.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q0f16-vulkan.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q4f16_1-vulkan.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q0f16-metal.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q4f16_1-metal.so>`__
    - 
    - 
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q0f16-ctx2k-webgpu.wasm>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-2/phi-2-q4f16_1-ctx2k-webgpu.wasm>`__
    -
  * - Phi-1.5
  
      (1.3B)
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q0f16-cuda.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q4f16_1-cuda.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q0f16-vulkan.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q4f16_1-vulkan.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q0f16-metal.so>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q4f16_1-metal.so>`__
    - 
    - 
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q0f16-ctx2k-webgpu.wasm>`__

      `q4f16_1 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/phi-1_5/phi-1_5-q4f16_1-ctx2k-webgpu.wasm>`__
    -

.. _gpt2_library_table:
  
GPT2
^^^^
.. list-table:: GPT2
  :widths: 8 8 8 8 8 8 8 8 8 8 8
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

      (M Chip)
    - Metal

      (Intel)
    - iOS
    - Android
    - webgpu
    - mali
  * - GPT2 
  
      (124M)
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2/gpt2-q0f16-cuda.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2/gpt2-q0f16-vulkan.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2/gpt2-q0f16-metal.so>`__
    - 
    - 
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2/gpt2-q0f16-ctx1k-webgpu.wasm>`__
    -
  * - GPT2-med
  
      (355M)
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2-medium/gpt2-medium-q0f16-cuda.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2-medium/gpt2-medium-q0f16-vulkan.so>`__
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2-medium/gpt2-medium-q0f16-metal.so>`__
    - 
    - 
    - 
    - `q0f16 <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/gpt2-medium/gpt2-medium-q0f16-ctx1k-webgpu.wasm>`__
    -

.. _model-variant-tables:

Level 3: Model Variant Tables (Precompiled Weights)
---------------------------------------------------

Finally, for each model variant, we provide the precompiled weights we uploaded to hugging face.

Each precompiled weight is categorized by its model size (e.g. 7B vs. 13B) and the quantization scheme (e.g. ``q3f16_1`` vs. ``q4f16_1``). We note that the weights are **platform-agnostic**.

Each model variant also loads its conversation configuration from a pre-defined :ref:`conversation template<load-predefined-conv-template>`. Note that multiple model variants can share a common conversation template.

Some of these files are uploaded by our community contributors--thank you!

.. _llama2_variant_table:

`Llama-2 <https://ai.meta.com/llama/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``llama-2``

.. list-table:: Llama-2
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 (Chat) <https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC>`__
      * `q4f32_1 (Chat) <https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC>`__

  * - 13B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/Llama-2-13b-chat-hf-q4f16_1-MLC>`__

  * - 70B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/Llama-2-70b-chat-hf-q4f16_1-MLC>`__

.. _mistralinstruct_variant_table:

`Mistral <https://huggingface.co/docs/transformers/main/en/model_doc/mistral>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``mistral_default``

.. list-table:: Mistral
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q3f16_1 (Instruct) <https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q3f16_1-MLC>`__
      * `q4f16_1 (Instruct) <https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC>`__

.. _neuralhermes_variant_table:

`NeuralHermes-2.5-Mistral <https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``neural_hermes_mistral``

.. list-table:: Neural Hermes
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC>`__

.. _openhermes_variant_table:

`OpenHermes-2-Mistral <https://huggingface.co/teknium/OpenHermes-2-Mistral-7B>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``open_hermes_mistral``

.. list-table:: Open Hermes
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/OpenHermes-2.5-Mistral-7B-q4f16_1-MLC>`__



.. _wizardmathv1.1_variant_table:

`WizardMath V1.1 <https://github.com/nlpxucan/WizardLM/tree/main/WizardMath>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``wizard_coder_or_math``

.. list-table:: WizardMath
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 7B
    - * `q4f16_1 <https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC>`__


.. _red_pajama_variant_table:

`RedPajama <https://www.together.xyz/blog/redpajama>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``redpajama_chat``

.. list-table:: Red Pajama
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - 3B
    - * `q4f16_1 (Chat) <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC>`__
      * `q4f32_1 (Chat) <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC>`__


.. _phi_variant_table:

`Phi <https://huggingface.co/microsoft/phi-2>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``phi-2``

.. list-table:: Phi
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - Phi-2 (2.7B)
    - * `q0f16 <https://huggingface.co/mlc-ai/phi-2-q0f16-MLC>`__
      * `q4f16_1 <https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC>`__
  * - Phi-1.5 (1.3B)
    - * `q0f16 <https://huggingface.co/mlc-ai/phi-1_5-q0f16-MLC>`__
      * `q4f16_1 <https://huggingface.co/mlc-ai/phi-1_5-q4f16_1-MLC>`__


.. _gpt2_variant_table:

`GPT2 <https://huggingface.co/docs/transformers/model_doc/gpt2>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation template: ``gpt2``

.. list-table:: GPT2
  :widths: 30 30
  :header-rows: 1

  * - Size
    - Hugging Face Repo Link
  * - GPT2 (124M)
    - * `q0f16 <https://huggingface.co/mlc-ai/gpt2-q0f16-MLC>`__
  * - GPT2-medium (355M)
    - * `q0f16 <https://huggingface.co/mlc-ai/gpt2-medium-q0f16-MLC>`__


------------------


.. _contribute-models-to-mlc-llm:

Contribute Models to MLC-LLM
----------------------------

Ready to contribute your compiled models/new model architectures? Awesome! Please check :ref:`contribute-new-models` on how to contribute new models to MLC-LLM.
