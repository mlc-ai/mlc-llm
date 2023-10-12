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
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

            .. tab:: CUDA 11.7

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly-cu117 mlc-ai-nightly-cu117

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly-cu118 mlc-ai-nightly-cu118

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly-cu121 mlc-ai-nightly-cu121

            .. tab:: CUDA 12.2

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly-cu122 mlc-ai-nightly-cu122

            .. tab:: ROCm 5.6

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly-rocm56 mlc-ai-nightly-rocm56
    
            .. tab:: ROCm 5.7

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly-rocm57 mlc-ai-nightly-rocm57

            .. tab:: Vulkan

                Supported in all Linux packages.

        .. note::

            If encountering issues with GLIBC not found, please install the latest glibc in conda:

            .. code-block:: bash

                conda install -c conda-forge libgcc-ng

    .. tab:: macOS

        .. tabs::

            .. tab:: CPU + Metal

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

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
                    python3 -m pip install --pre --force-reinstall -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

        .. note::
            If encountering the error below:

            .. code-block:: bash

                FileNotFoundError: Could not find module 'path\to\site-packages\tvm\tvm.dll' (or one of its dependencies). Try using the full path with constructor syntax.

            It is likely `zstd`, a dependency to LLVM, was missing. Please `download <https://github.com/facebook/zstd/releases/tag/v1.5.5>`__ the precompiled binary, rename it to `zstd.dll` and copy to the same folder as `tvm.dll`.


Option 2. Build from Source
---------------------------

Upcoming.