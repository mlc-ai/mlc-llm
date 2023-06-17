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

Get Started
-----------

We have prepared a comprehensive tutorial to assist you in getting started with MLC LLM. This tutorial aims to familiarize you with the terminologies commonly used in MLC LLM and provide you with a clear understanding of the overall workflow, and help you navigate through the documentation.

- :doc:`get_started/index`

.. _runtime_apis:

Run models with MLC-Chat APIs
-----------------------------

The following tutorials will guide you on running MLC-compiled models using MLCChat APIs in various programming languages.
These tutorials aim to help you create your own applications based on these models.

.. tabs ::

   .. tab :: C++
      - :doc:`tutorials/runtime/cpp`
   .. tab :: Javascript
      - :doc:`tutorials/runtime/javascript`
   .. tab :: REST
      - :doc:`tutorials/runtime/rest`

.. _compile_models:

Compile Models
--------------

:doc:`TVM Unity </install/tvm>` is required to compile models.

- :doc:`tutorials/compilation/model_compilation_walkthrough`
- :doc:`tutorials/compilation/compiler_artifacts`
- :doc:`tutorials/compilation/configure_targets`
- :doc:`tutorials/compilation/configure_quantization`

.. _define_new_models:

Define New Model Architectures
------------------------------

:doc:`TVM Unity </install/tvm>` is required to define new model architectures.

- :doc:`tutorials/bring-your-own-models`

MLC-Prebuilt LLMs
-----------------

- :doc:`tutorials/prebuilts/prebuilt_models`

Community
---------

- :doc:`Community guideline <community/guideline>` 
- :doc:`FAQs <community/faq>` 

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

   get_started/index.rst
   tutorials/runtime/mlc_chat_config.rst

.. toctree::
   :maxdepth: 1
   :caption: Run models with MLC-Chat APIs
   :hidden:

   tutorials/runtime/cpp.rst
   tutorials/runtime/javascript.rst
   tutorials/runtime/rest.rst

.. toctree::
   :maxdepth: 1
   :caption: Compile Models
   :hidden:

   tutorials/compilation/get-vicuna-weight.rst
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
   install/conda.rst
   install/gpu.rst
   install/cli.rst
   install/ios.rst
   install/android.rst

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   community/guideline.rst
   community/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: Other tutorials

   tutorials/deploy-models.rst
   tutorials/compile-models.rst
   tutorials/bring-your-own-models.rst
