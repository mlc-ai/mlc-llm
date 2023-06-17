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

.. _get_started:

Get Started --- Try out MLC LLM on your device
----------------------------------------------

We have prepared packages for you to try out MLC LLM locally, and you can try out prebuilt models on your device:

.. tabs::

  .. tab:: iOS

    The MLC LLM app is now accessible on the App Store at no cost. You can download and explore it by simply clicking the button below:

    .. image:: https://linkmaker.itunes.apple.com/assets/shared/badges/en-us/appstore-lrg.svg
      :width: 135
      :target: https://apps.apple.com/us/app/mlc-chat/id6448482937

    Once the app is installed, you can download the models and then engage in chat with the model without requiring an internet connection.

    Memory requirements vary across different models. The Vicuna-7B model necessitates an iPhone device with a minimum of 6GB RAM, whereas the RedPajama-3B model can run on an iPhone with at least 4GB RAM.

    .. figure:: https://mlc.ai/blog/img/redpajama/ios.gif
      :width: 300
      :align: center

      MLC LLM on iOS

  .. tab:: Android

    The MLC LLM Android app is free and available for download and can be tried out by simply clicking the button below:

    .. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
      :width: 135
      :target: https://github.com/mlc-ai/binary-mlc-llm-libs/raw/main/mlc-chat.apk

    Once the app is installed, you can engage in a chat with the model without the need for an internet connection:

    Memory requirements vary across different models. The Vicuna-7B model necessitates an Android device with a minimum of 6GB RAM, whereas the RedPajama-3B model can run on an Android device with at least 4GB RAM.

    .. figure:: https://mlc.ai/blog/img/android/android-recording.gif
      :width: 300
      :align: center

      MLC LLM on Android

  .. tab:: CLI (Linux/MacOS/Windows)

    To utilize the models on your PC, we highly recommend trying out the CLI version of MLC LLM.

    We have prepared Conda packages for MLC Chat CLI. If you haven't installed Conda yet, please refer to :doc:`this tutorial </install/conda>` to install Conda.
    To install MLC Chat CLI, simply run the following command:

    .. code:: bash

      conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-nightly
      conda activate mlc-chat-venv

    If you are using Windows or Linux and want to use your native GPU, please follow the instructions in :doc:`/install/gpu` tutorial to prepare the environment.

    .. figure:: https://mlc.ai/blog/img/redpajama/cli.gif
      :width: 300
      :align: center

      MLC LLM on CLI

  .. tab:: Web Browser

    With the advancements of WebGPU, we can now run LLM directly on web browsers. You have the opportunity to experience the web version of MLC LLM through `WebLLM <https://mlc.ai/webllm>`__.

    Once the parameters have been fetched and stored in the local cache, you can begin interacting with the model without the need for an internet connection.

    You can use `WebGPU Report <https://webgpureport.org/>`__ to verify the functionality of WebGPU on your browser.

    .. figure:: https://mlc.ai/blog/img/redpajama/web.gif
      :width: 300
      :align: center

      MLC LLM on Web

Project Overview
----------------

The project consists of three distinct submodules: model definition, model compilation, and runtimes.

.. figure:: /_static/img/project-structure.svg
   :width: 600
   :align: center
   :alt: Project Structure

   Three independent submodules in MLC LLM

.. ➀➁➂➃➄➅➆➇➈➉
.. ➊➋➌➍➎➏➐➑➒➓

**➀ Model definition in Python.** MLC offers a variety of pre-defined architectures, such as Llama (e.g., Vicuna, OpenLlama, Llama, Wizard), GPT-NeoX (e.g., RedPajama, Dolly), RNNs (e.g., RWKV), and GPT-J (e.g., MOSS). Model developers could solely define the model in pure Python, without having to touch code generation and runtime.

**➁ Model compilation in Python.** :doc:`TVM Unity </install/tvm>` compiler are configured in pure python, and it quantizes and exports the Python-based model to :ref:`model lib <model_lib>` and quantized :ref:`model weights <model_weights>`. Quantization and optimization algorithms can be developed in pure Python to compress and accelerate LLMs for specific usecases.

**➂ Platform-native runtimes.** Variants of MLCChat are provided on each platform: **C++** for command line, **Javascript** for web, **Swift** for iOS, and **Java** for Android, configurable with a JSON :ref:`chat config <chat_config>`. App developers only need to familiarize with the platform-naive runtimes to integrate MLC-compiled LLMs into their projects.

Customize MLC-Chat Configuration
--------------------------------

The behavior of the chat can be customized by modifying the chat configuration file. To learn more about customizing the chat configuration JSON, you can refer to the following tutorials which provide a detailed walkthrough:

- :doc:`/tutorials/runtime/mlc_chat_config`

Model Prebuilts
---------------

To use different pre-built models, you can refer to the following tutorials:

- :doc:`/tutorials/prebuilts/prebuilt_models`

Misc
----

If you find MLC LLM useful in your work, please consider citing the project using the following format:

.. code:: bibtex

   @software{mlc-llm,
      author = {MLC team},
      title = {{MLC-LLM}},
      url = {https://github.com/mlc-ai/mlc-llm},
      year = {2023}
   }

The underlying compiler techniques employed by MLC LLM are outlined in the following papers:

.. collapse:: References (Click to expand)

   .. code:: bibtex

      @inproceedings{tensorir,
         author = {Feng, Siyuan and Hou, Bohan and Jin, Hongyi and Lin, Wuwei and Shao, Junru and Lai, Ruihang and Ye, Zihao and Zheng, Lianmin and Yu, Cody Hao and Yu, Yong and Chen, Tianqi},
         title = {TensorIR: An Abstraction for Automatic Tensorized Program Optimization},
         year = {2023},
         isbn = {9781450399166},
         publisher = {Association for Computing Machinery},
         address = {New York, NY, USA},
         url = {https://doi.org/10.1145/3575693.3576933},
         doi = {10.1145/3575693.3576933},
         booktitle = {Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
         pages = {804–817},
         numpages = {14},
         keywords = {Tensor Computation, Machine Learning Compiler, Deep Neural Network},
         location = {Vancouver, BC, Canada},
         series = {ASPLOS 2023}
      }

      @inproceedings{metaschedule,
         author = {Shao, Junru and Zhou, Xiyou and Feng, Siyuan and Hou, Bohan and Lai, Ruihang and Jin, Hongyi and Lin, Wuwei and Masuda, Masahiro and Yu, Cody Hao and Chen, Tianqi},
         booktitle = {Advances in Neural Information Processing Systems},
         editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
         pages = {35783--35796},
         publisher = {Curran Associates, Inc.},
         title = {Tensor Program Optimization with Probabilistic Programs},
         url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf},
         volume = {35},
         year = {2022}
      }

      @inproceedings{tvm,
         author = {Tianqi Chen and Thierry Moreau and Ziheng Jiang and Lianmin Zheng and Eddie Yan and Haichen Shen and Meghan Cowan and Leyuan Wang and Yuwei Hu and Luis Ceze and Carlos Guestrin and Arvind Krishnamurthy},
         title = {{TVM}: An Automated {End-to-End} Optimizing Compiler for Deep Learning},
         booktitle = {13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)},
         year = {2018},
         isbn = {978-1-939133-08-3},
         address = {Carlsbad, CA},
         pages = {578--594},
         url = {https://www.usenix.org/conference/osdi18/presentation/chen},
         publisher = {USENIX Association},
         month = oct,
      }
..
|


If you are interested in using Machine Learning Compilation in practice, we highly recommend the following course:

- `Machine Learning Compilation <https://mlc.ai/>`__

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:

   tutorials/runtime/mlc_chat_config.rst

.. toctree::
   :maxdepth: 1
   :caption: Build Apps
   :hidden:

   tutorials/runtime/terminologies.rst
   tutorials/runtime/cpp.rst
   tutorials/runtime/javascript.rst
   tutorials/runtime/rest.rst
   tutorials/app_build/cli.rst
   tutorials/app_build/ios.rst
   tutorials/app_build/android.rst

.. toctree::
   :maxdepth: 1
   :caption: Compile Models
   :hidden:

   tutorials/compilation/compile_models.rst
   tutorials/compilation/distribute_compiled_models.rst
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
   install/conda.rst
   install/gpu.rst

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   community/guideline.rst
   community/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: Other tutorials
   :hidden:

   tutorials/deploy-models.rst
   tutorials/bring-your-own-models.rst
