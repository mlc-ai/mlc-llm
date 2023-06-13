👋 Welcome to MLC LLM
===================================

`Discord <https://discord.gg/9Xpy2HGBuD>`_ | `Demo <https://mlc.ai/mlc-llm>`_ | `GitHub <https://github.com/mlc-ai/mlc-llm>`_

🚧 This document is currently undergoing heavy construction.

Machine Learning Compilation for LLM (MLC LLM) is a universal deployment solution that enables LLMs to run efficiently on consumer devices, leveraging native hardware acceleration like GPUs.

.. table:: MLC LLM: A universal deployment solution for large language models
   :widths: 200, 250, 250, 250, 250
   :align: center

   +-------------------+-------------------+-------------------+-------------------+---------------------+
   |                   |  AMD GPU          | NVIDIA GPU        | Apple M1/M2 GPU   | Intel GPU           |
   +===================+===================+===================+===================+=====================+
   |   Linux / Win     | ✅ Vulkan, ROCm   | ✅ Vulkan, CUDA   | ❌                | ✅ Vulkan           |
   +-------------------+-------------------+-------------------+-------------------+---------------------+
   |   macOS           | ✅ Metal          | ❌                | ✅ Metal          | ✅ Metal            |
   +-------------------+-------------------+-------------------+-------------------+---------------------+
   |   Web Browser     | ✅ WebGPU         | ✅ WebGPU         | ✅ WebGPU         | ✅ WebGPU           |
   +-------------------+-------------------+-------------------+-------------------+---------------------+
   |   iOS / iPadOS    | ✅ Metal on Apple M1/M2 GPU                                                     |
   +-------------------+-------------------+-------------------+-------------------+---------------------+
   |   Android         | ✅ OpenCL on Adreno GPU               | 🚧  OpenCL on Mali GPU                  |
   +-------------------+-------------------+-------------------+-------------------+---------------------+


Get Started
-----------

:doc:`get_started/mlcchat_terminologies`


Project Structure
-----------------

The project comprises three independent submodules: model definition, model compilation, and runtimes.

.. figure:: _static/img/project-structure.svg
   :width: 600
   :align: center
   :alt: Project Structure

   Three independent submodules in MLC LLM

.. ➀➁➂➃➄➅➆➇➈➉
.. ➊➋➌➍➎➏➐➑➒➓

**➀ Model definition in Python.** MLC offers a variety of pre-defined architectures, such as Llama (e.g., Vicuna, OpenLlama, Llama, Wizard), GPT-NeoX (e.g., RedPajama, Dolly), RNNs (e.g., RWKV), and GPT-J (e.g., MOSS). Model developers could solely define the model in pure Python, without having to touch code generation and runtime.

**➁ Model compilation in Python.** :doc:`TVM Unity <install/tvm>` compiler are configured in pure python, and it quantizes and exports the Python-based model to :ref:`model lib <model_lib>` and quantized :ref:`model weights <model_weights>`. Quantization and optimization algorithms can be developed in pure Python to compress and accelerate LLMs for specific usecases.

**➂ Platform-native runtimes.** Variants of MLCChat are provided on each platform: **C++** for command line, **Javascript** for web, **Swift** for iOS, and **Java** for Android, configurable with a JSON :ref:`chat config <chat_config>`. App developers only need to familiarize with the platform-naive runtimes to integrate MLC-compiled LLMs into their projects.

Tutorials
---------

.. tabs ::

   .. tab :: ➂ Use Compiled Models

      The following tutorials introduce how to run MLC-compiled models with MLCChat or in platform-native projects.

      - :doc:`tutorials/runtime/cpp` for command line
      - :doc:`tutorials/runtime/javascript` for WebLLM
      - :doc:`tutorials/runtime/android` for Android
      - :doc:`tutorials/runtime/ios` for iOS
      - :doc:`tutorials/runtime/python` for running models in Python

      **Note.** :doc:`TVM Unity <install/tvm>` compiler is not a dependency to running any MLC-compiled model.

   .. tab :: ➁ Compile Models

      :doc:`TVM Unity <install/tvm>` is required to compile models.

      - :doc:`tutorials/compilation/model_compilation_walkthrough`
      - :doc:`tutorials/compilation/compiler_artifacts`
      - :doc:`tutorials/compilation/configure_targets`
      - :doc:`tutorials/compilation/configure_quantization`

   .. tab :: ➀ Define Model Architectures

      :doc:`TVM Unity <install/tvm>` is required to define new model architectures.

      - :doc:`tutorials/customize/define_new_models`

   .. tab :: MLC-Prebuilt LLMs

      - :doc:`tutorials/prebuilts/prebuilt_models`

.. - Machine Learning Compilation Basics: `Machine Learning Compilation <https://mlc.ai/>`__


Community
---------

- :doc:`Community guideline <community/guideline>` 
- :doc:`FAQs <community/faq>` 

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:

   get_started/mlcchat_terminologies.rst

.. toctree::
   :maxdepth: 1
   :caption: Use Compiled Models
   :hidden:

   tutorials/runtime/cpp.rst
   tutorials/runtime/javascript.rst
   tutorials/runtime/android.rst
   tutorials/runtime/ios.rst
   tutorials/runtime/python.rst

.. toctree::
   :maxdepth: 1
   :caption: Compile Models
   :hidden:

   tutorials/compilation/model_compilation_walkthrough.rst
   tutorials/compilation/compiler_artifacts.rst
   tutorials/compilation/configure_targets.rst
   tutorials/compilation/configure_quantization.rst

.. toctree::
   :maxdepth: 1
   :caption: Define Model Architectures
   :hidden:

   tutorials/customize/define_new_models.rst

.. toctree::
   :maxdepth: 1
   :caption: Prebuilt Models
   :hidden:

   tutorials/prebuilts/prebuilt_models.rst

.. toctree::
   :maxdepth: 1
   :caption: Installation and Dependency
   :hidden:

   install/tvm.rst

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   community/guideline.rst
   community/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: Other tutorials

   install/software-dependencies.rst
   tutorials/deploy-models.rst
   tutorials/compile-models.rst
   tutorials/bring-your-own-models.rst
   tutorials/customize-conversation.rst
   tutorials/customize.rst
