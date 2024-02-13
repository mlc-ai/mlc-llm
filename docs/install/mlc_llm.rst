.. _install-mlc-packages:

Install MLC LLM Python Package
==============================

.. contents:: Table of Contents
    :local:
    :depth: 2

MLC LLM Python Package can be installed directly from a prebuilt developer package, or built from source.

Option 1. Prebuilt Package
--------------------------

We provide nightly built pip wheels for MLC-LLM via pip.
Select your operating system/compute platform and run the command in your terminal:

.. note::
    ❗ Whenever using Python, it is highly recommended to use **conda** to manage an isolated Python environment to avoid missing dependencies, incompatible versions, and package conflicts.

.. tabs::

    .. tab:: Linux

        .. tabs::

            .. tab:: CPU

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

            .. tab:: CUDA 11.7

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu117 mlc-ai-nightly-cu117

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu118 mlc-ai-nightly-cu118

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu121 mlc-ai-nightly-cu121

            .. tab:: CUDA 12.2

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu122 mlc-ai-nightly-cu122

            .. tab:: ROCm 5.6

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-rocm56 mlc-ai-nightly-rocm56
    
            .. tab:: ROCm 5.7

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-rocm57 mlc-ai-nightly-rocm57

            .. tab:: Vulkan

                Supported in all Linux packages.

        .. note::

            If encountering issues with GLIBC not found, please install the latest glibc in conda:

            .. code-block:: bash

                conda install -c conda-forge libgcc-ng

            Besides, we would recommend using Python 3.11; so if you are creating a new environment,
            you could use the following command:

            .. code-block:: bash

                conda create --name mlc-prebuilt  python=3.11

    .. tab:: macOS

        .. tabs::

            .. tab:: CPU + Metal

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

        .. note::

            Always check if conda is installed properly in macOS using the command below:

            .. code-block:: bash

                conda info | grep platform

            It should return "osx-64" for Mac with Intel chip, and "osx-arm64" for Mac with Apple chip.

    .. tab:: Windows

        .. tabs::

            .. tab:: CPU + Vulkan

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

        .. note::
            If encountering the error below:

            .. code-block:: bash

                FileNotFoundError: Could not find module 'path\to\site-packages\tvm\tvm.dll' (or one of its dependencies). Try using the full path with constructor syntax.

            It is likely `zstd`, a dependency to LLVM, was missing. Please use the command below to get it installed:

            .. code-block:: bash

                conda install zstd


Then you can verify installation in command line:

.. code-block:: bash

    python -c "import mlc_chat; print(mlc_chat)"
    # Prints out: <module 'mlc_chat' from '/path-to-env/lib/python3.11/site-packages/mlc_chat/__init__.py'>

|

.. _mlcchat_build_from_source:

Option 2. Build from Source
---------------------------

We also provide options to build mlc runtime libraries ``mlc_chat`` from source.
This step is useful when you want to make modification or obtain a specific version of mlc runtime.


**Step 1. Set up build dependency.** To build from source, you need to ensure that the following build dependencies are satisfied:

* CMake >= 3.24
* Git
* `Rust and Cargo <https://www.rust-lang.org/tools/install>`_, required by Hugging Face's tokenizer
* One of the GPU runtimes:

    * CUDA >= 11.8 (NVIDIA GPUs)
    * Metal (Apple GPUs)
    * Vulkan (NVIDIA, AMD, Intel GPUs)

.. code-block:: bash
    :caption: Set up build dependencies in Conda

    # make sure to start with a fresh environment
    conda env remove -n mlc-chat-venv
    # create the conda environment with build dependency
    conda create -n mlc-chat-venv -c conda-forge \
        "cmake>=3.24" \
        rust \
        git \
        python=3.11
    # enter the build environment
    conda activate mlc-chat-venv

.. note::
    For runtime, :doc:`TVM Unity </install/tvm>` compiler is not a dependency for MLCChat CLI or Python API. Only TVM's runtime is required, which is automatically included in `3rdparty/tvm <https://github.com/mlc-ai/mlc-llm/tree/main/3rdparty>`_.
    However, if you would like to compile your own models, you need to follow :doc:`TVM Unity </install/tvm>`.

**Step 2. Configure and build.** A standard git-based workflow is recommended to download MLC LLM, after which you can specify build requirements with our lightweight config generation tool:

.. code-block:: bash
    :caption: Configure and build

    # clone from GitHub
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/
    # create build directory
    mkdir -p build && cd build
    # generate build configuration
    python3 ../cmake/gen_cmake_config.py
    # build mlc_llm libraries
    cmake .. && cmake --build . --parallel $(nproc) && cd ..

.. note::
    If you are using CUDA and your compute capability is above 80, then it is require to build with
    ``set(USE_FLASHINFER ON)``. Otherwise, you may run into ``Cannot find PackedFunc`` issue during
    runtime.
    
    To check your CUDA compute capability, you can use ``nvidia-smi --query-gpu=compute_cap --format=csv``.

**Step 3. Install via Python.** We recommend that you install ``mlc_chat`` as a Python package, giving you 
access to ``mlc_chat.compile``, ``mlc_chat.ChatModule``, and the CLI.
There are two ways to do so:

    .. tabs ::

       .. code-tab :: bash Install via environment variable

          export PYTHONPATH=/path-to-mlc-llm/python:$PYTHONPATH

       .. code-tab :: bash Install via pip local project

          conda activate your-own-env
          which python # make sure python is installed, expected output: path_to_conda/envs/your-own-env/bin/python
          cd /path-to-mlc-llm/python
          pip install -e .

**Step 4. Validate installation.** You may validate if MLC libarires and mlc_chat CLI is compiled successfully using the following command:

.. code-block:: bash
    :caption: Validate installation

    # expected to see `libmlc_llm.so` and `libtvm_runtime.so`
    ls -l ./build/
    # expected to see help message
    mlc_chat chat -h

Finally, you can verify installation in command line. You should see the path you used to build from source with:

.. code:: bash

   python -c "import mlc_chat; print(mlc_chat)"
