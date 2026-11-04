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
    Please make sure your conda environment has Python and pip installed.

.. tabs::

    .. tab:: Linux

        .. tabs::

            .. tab:: CPU

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

            .. tab:: CUDA 12.8

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu128 mlc-ai-nightly-cu128

            .. tab:: CUDA 13.0

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu130 mlc-ai-nightly-cu130

            .. tab:: ROCm 6.1

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm61 mlc-ai-nightly-rocm61

            .. tab:: ROCm 6.2

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm62 mlc-ai-nightly-rocm62

            .. tab:: Vulkan

                Supported in all Linux packages. Checkout the following instructions
                to install the latest vulkan loader to avoid vulkan not found issue.

                .. code-block:: bash

                    conda install -c conda-forge gcc libvulkan-loader

        .. note::
            We need git-lfs in the system, you can install it via

            .. code-block:: bash

                conda install -c conda-forge git-lfs

            If encountering issues with GLIBC not found, please install the latest glibc in conda:

            .. code-block:: bash

                conda install -c conda-forge libstdcxx-ng

            Besides, we would recommend using Python 3.13; so if you are creating a new environment,
            you could use the following command:

            .. code-block:: bash

                conda create --name mlc-prebuilt  python=3.13

    .. tab:: macOS

        .. tabs::

            .. tab:: CPU + Metal

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

        .. note::

            Always check if conda is installed properly in macOS using the command below:

            .. code-block:: bash

                conda info | grep platform

            It should return "osx-64" for Mac with Intel chip, and "osx-arm64" for Mac with Apple chip.
            We need git-lfs in the system, you can install it via

            .. code-block:: bash

                conda install -c conda-forge git-lfs

    .. tab:: Windows

        .. tabs::

            .. tab:: CPU + Vulkan

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

        .. note::
            Please make sure your conda environment comes with python and pip.
            Make sure you also install the following packages,
            vulkan loader, clang, git and git-lfs to enable proper automatic download
            and jit compilation.

            .. code-block:: bash

                conda install -c conda-forge clang libvulkan-loader git-lfs git

            If encountering the error below:

            .. code-block:: bash

                FileNotFoundError: Could not find module 'path\to\site-packages\tvm\tvm.dll' (or one of its dependencies). Try using the full path with constructor syntax.

            It is likely `zstd`, a dependency to LLVM, was missing. Please use the command below to get it installed:

            .. code-block:: bash

                conda install zstd


Then you can verify installation in command line:

.. code-block:: bash

    python -c "import mlc_llm; print(mlc_llm)"
    # Prints out: <module 'mlc_llm' from '/path-to-env/lib/python3.13/site-packages/mlc_llm/__init__.py'>

|

.. _mlcchat_build_from_source:

Option 2. Build from Source
---------------------------

We also provide options to build mlc runtime libraries ``mlc_llm`` from source.
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
        python=3.13
    # enter the build environment
    conda activate mlc-chat-venv

.. note::
    For runtime, :doc:`TVM </install/tvm>` compiler is not a dependency for MLCChat CLI or Python API. Only TVM's runtime is required, which is automatically included in `3rdparty/tvm <https://github.com/mlc-ai/mlc-llm/tree/main/3rdparty>`_.
    However, if you would like to compile your own models, you need to follow :doc:`TVM </install/tvm>`.

**Step 2. Configure and build.** A standard git-based workflow is recommended to download MLC LLM, after which you can specify build requirements with our lightweight config generation tool:

.. code-block:: bash
    :caption: Configure and build

    # clone from GitHub
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/
    # create build directory
    mkdir -p build && cd build
    # generate build configuration
    python ../cmake/gen_cmake_config.py
    # build mlc_llm libraries
    cmake .. && make -j $(nproc) && cd ..

**Step 3. Set up Python environment and install package.** Before using MLC LLM from Python, ensure your environment has all necessary dependencies. The core required package is `apache-tvm-ffi` [citation:8][citation:12]. Additionally, `numpy` is required for tensor operations [citation:4]. You can install them using pip:

.. code-block:: bash
    :caption: Install Python dependencies

    pip install numpy apache-tvm-ffi

After dependencies are installed, we recommend that you install ``mlc_llm`` as a Python package, giving you access to ``mlc_llm.compile``, ``mlc_llm.MLCEngine``, and the CLI. There are two ways to do so:

    .. tabs ::

       .. code-tab :: bash Install via environment variable

          export MLC_LLM_SOURCE_DIR=/path-to-mlc-llm
          export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
          alias mlc_llm="python -m mlc_llm"

       .. code-tab :: bash Install via pip local project

          conda activate your-own-env
          which python # make sure python is installed, expected output: path_to_conda/envs/your-own-env/bin/python
          cd /path-to-mlc-llm/python
          pip install -e .

**Step 4. Configure library path.** For the Python package to locate the compiled runtime libraries (e.g., `libtvm_runtime.so` or `libmlc_llm.so`), you need to set the appropriate library path environment variable. The MLC LLM library loading logic searches these paths by default [citation:2]. If the libraries are not found, you must configure the path manually.

.. code-block:: bash
    :caption: Set library path for runtime

    # On Linux, use LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
    # On macOS, use DYLD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/build/lib:$DYLD_LIBRARY_PATH

.. note::
    The need to set ``LD_LIBRARY_PATH`` or ``DYLD_LIBRARY_PATH`` can often be avoided by using the `pip install -e .` method from **Step 3**, as it handles library discovery more robustly. Use the environment variable approach primarily for troubleshooting or when using the `PYTHONPATH` method.

**Step 5. Validate installation.** You may validate if MLC libraries and ``mlc_llm`` CLI are compiled successfully and the Python environment is correctly configured using the following commands:

.. code-block:: bash
    :caption: Validate installation

    # expected to see `libmlc_llm.so` and `libtvm_runtime.so`
    ls -l ./build/

    # Verify TVM runtime and MLC LLM Python module can be imported
    python -c "import tvm; print(tvm.__version__)"
    python -c "from mlc_llm import MLCEngine; print('MLC-LLM imported successfully')"

    # expected to see help message
    mlc_llm chat -h

Finally, you can verify installation in command line. You should see the path you used to build from source with:

.. code:: bash

   python -c "import mlc_llm; print(mlc_llm)"
