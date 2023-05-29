.. _Installation and Setup:

=================
Install and Setup
=================

.. contents:: Table of Contents
    :depth: 3

.. _tvm-unity-install:

Install TVM Unity
-----------------

`TVM Unity <https://discuss.tvm.apache.org/t/establish-tvm-unity-connection-a-technical-strategy/13344>`__, the latest development in Apache TVM, is required to build MLC LLM. To install TVM Unity, there are two options available:

- installing a prebuilt developer package, or
- building TVM Unity from source.

.. _tvm-unity-install-prebuilt-package:

Option 1. Prebuilt Package
==========================

To help our community to use Apache TVM Unity, a nightly prebuilt developer package is provided with everything packaged.
Please visit the installation page for installation instructions: https://mlc.ai/package/.

**Set up $TVM_HOME after installation.** This environment variable is not used by TVM itself but by downstream applications like `MLC LLM <https://mlc.ai/mlc-llm>`_, and thus it would be more convenient to set it up earlier:

.. code-block:: bash

    # Locate TVM python package
    >>> python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.11/site-packages/tvm/__init__.py
    # Set up the environment variable
    >>> export TVM_HOME=/some-path/lib/python3.11/site-packages/tvm/

.. note::
    If installed properly, ``libtvm.{so|dylib|dll}`` should exist right under ``$TVM_HOME``.


.. _tvm-unity-build-from-source:

Option 2. Build from Source
===========================

**Step 1. Set up build dependency.** To build from source, you need to ensure that the following build dependencies are met:

- CMake >= 3.18
- LLVM >= 15
- Git
- (Optional) CUDA >= 11.8 (targeting NVIDIA GPUs)
- (Optional) Metal (targeting Apple GPUs such as M1 and M2)
- (Optional) Vulkan (targeting NVIDIA, AMD, Intel and mobile GPUs)
- (Optional) OpenCL (targeting NVIDIA, AMD, Intel and mobile GPUs)

.. note::
    - To target NVIDIA GPUs, either CUDA or Vulkan is required (CUDA is recommended);
    - For AMD and Intel GPUs, Vulkan is necessary;
    - When targeting Apple (macOS, iOS, iPadOS), Metal is a mandatory dependency;
    - Some Android devices only support OpenCL, but most of them support Vulkan.

To easiest way to manage dependency is via conda, which maintains a set of toolchains including LLVM across platforms. To create the environment of those build dependencies, one may simply use:

.. code-block:: bash
    :caption: Set up build dependencies in conda

    # make sure to start with a fresh environment
    conda env remove -n tvm-unity-build
    # create the conda environment with build dependency
    conda create -n tvm-unity-build -c conda-forge \
        "llvmdev>=15" \
        "cmake>=3.18" \
        git
    # enter the build environment
    conda activate tvm-unity-build

**Step 2. Configure and build.** Standard git-based workflow are recommended to download Apache TVM Unity, and then specify build requirements in ``config.cmake``:

.. code-block:: bash
    :caption: Download TVM Unity from GitHub

    # clone from GitHub
    git clone --recursive git@github.com:mlc-ai/relax.git tvm-unity && cd tvm-unity
    # create the build directory
    rm -rf build && mkdir build && cd build
    # specify build requirements in `config.cmake`
    cp ../cmake/config.cmake .
    vim config.cmake

.. note::
    We are temporarily using `mlc-ai/relax <https://github.com/mlc-ai/relax>`_ instead, which comes with several temporary outstanding changes that we will upstream to Apache TVM's `unity branch <https://github.com/apache/tvm/tree/unity>`_.

While ``config.cmake`` is well-documented, below are flags of the most interest:

.. code-block:: cmake
    :caption: Configure build in ``config.cmake``

    #### Edit `/path-tvm-unity/build/config.cmake`
    # Can be one of `Debug`, `RelWithDebInfo` (recommended) and `Release`
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    set(USE_LLVM   ON)  # LLVM is a must-dependency
    set(USE_CUDA   OFF) # Turn on if needed
    set(USE_METAL  OFF) # Turn on if needed
    set(USE_VULKAN OFF) # Turn on if needed
    set(USE_OpenCL OFF) # Turn on if needed

Once ``config.cmake`` is edited accordingly, kick off build with the commands below

.. code-block:: bash
    :caption: Build ``libtvm`` using cmake and cmake

    cmake ..
    make -j$(nproc)

A success build should produce ``libtvm`` and ``libtvm_runtime`` under ``/path-tvm-unity/build/`` directory.

.. note::
    To troubleshoot the build, output from cmake is usually quite helpful.

**Step 3. Set up environment variables.**
The following two environment variables are generally required for TVM-based applications:

.. code-block:: bash
    :caption: Setting up environment variables for TVM

    # make sure $TVM_HOME/build/libtvm.{so|dylib|dll} exists
    export TVM_HOME=/path-tvm/
    # make TVM's Python binding discoverable by Python interpreter
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

Validate Installation
=====================

Using a compiler infrastructure with multiple language bindings could be error-prone.
Therefore, it is highly recommended to validate TVM Unity installation before use.

**Step 1. Locate TVM Python package.** The following command can help confirm that TVM is properly installed as a python package and provide the location of the TVM python package:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.11/site-packages/tvm/__init__.py

**Step 2. Confirm which TVM library is used.** When maintaining multiple build or installation of TVM, it becomes important to double check if the python package is using the proper ``libtvm`` with the following command:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm._ffi.base._LIB)"
    <CDLL '/some-path/lib/python3.11/site-packages/tvm/libtvm.dylib', handle 95ada510 at 0x1030e4e50>

**Step 3. Reflect TVM build option.** Sometimes when downstream application fails, it could likely be some mistakes with a wrong TVM commit, or wrong build flags. To find it out, the following commands will be helpful:

.. code-block:: bash

    >>> python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
    ... # Omitted less relevant options
    GIT_COMMIT_HASH: 4f6289590252a1cf45a4dc37bce55a25043b8338
    HIDE_PRIVATE_SYMBOLS: ON
    USE_LLVM: llvm-config --link-static
    LLVM_VERSION: 15.0.7
    USE_VULKAN: OFF
    USE_CUDA: OFF
    CUDA_VERSION: NOT-FOUND
    USE_OPENCL: OFF
    USE_METAL: ON
    USE_ROCM: OFF

.. note::
    ``GIT_COMMIT_HASH`` indicates the exact commit of the TVM build, and it can be found on GitHub via `<https://github.com/mlc-ai/relax/commit/$GIT_COMMIT_HASH>`_.

**Step 4. Check device detection.** Sometimes it could be helpful to understand if TVM could detect your device at all with the following commands:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.metal().exist)"
    True # or False
    >>> python -c "import tvm; print(tvm.cuda().exist)"
    False # or True
    >>> python -c "import tvm; print(tvm.vulkan().exist)"
    False # or True

Please note that the commands above verify the presence of an actual device on the local machine for the TVM runtime (not the compiler) to execute properly. However, TVM compiler can perform compilation tasks without requiring a physical device. As long as the necessary toolchain, such as NVCC, is available, TVM supports cross-compilation even in the absence of an actual device.


Install MLC-LLM CLI
-------------------

MLC-LLM CLI is a command-line interface for MLC-LLM, which enables user to chat with the bot in terminal. Please refer to :ref:`prepare-weight-library` for installation instructions.