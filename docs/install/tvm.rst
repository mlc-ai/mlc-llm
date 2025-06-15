.. _install-tvm-unity:

Install TVM Unity Compiler
==========================

.. contents:: Table of Contents
    :local:
    :depth: 2

`TVM Unity <https://discuss.tvm.apache.org/t/establish-tvm-unity-connection-a-technical-strategy/13344>`__, the latest development in Apache TVM, is required to build MLC LLM. Its features include:

- High-performance CPU/GPU code generation instantly without tuning;
- Dynamic shape and symbolic shape tracking by design;
- Supporting both inference and training;
- Productive python-first compiler implementation. As a concrete example, MLC LLM compilation is implemented in pure python using its API.

TVM Unity can be installed directly from a prebuilt developer package, or built from source.

.. _tvm-unity-prebuilt-package:

Option 1. Prebuilt Package
--------------------------

A nightly prebuilt Python package of Apache TVM Unity is provided.

.. note::
    ❗ Whenever using Python, it is highly recommended to use **conda** to manage an isolated Python environment to avoid missing dependencies, incompatible versions, and package conflicts.

.. tabs::

   .. tab:: Linux

      .. tabs::

         .. tab:: CPU

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

         .. tab:: CUDA 12.2

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu122

         .. tab:: CUDA 12.3

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu123

         .. tab:: ROCm 6.1

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-rocm61

         .. tab:: ROCm 6.2

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-rocm62

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
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

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
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

      .. note::
        Make sure you also install vulkan loader and clang to avoid vulkan
        not found error or clang not found(needed for jit compile)

        .. code-block:: bash

            conda install -c conda-forge clang libvulkan-loader

        If encountering the error below:

        .. code-block:: bash

            FileNotFoundError: Could not find module 'path\to\site-packages\tvm\tvm.dll' (or one of its dependencies). Try using the full path with constructor syntax.

        It is likely `zstd`, a dependency to LLVM, was missing. Please use the command below to get it installed:

        .. code-block:: bash

            conda install zstd

.. _tvm-unity-build-from-source:

Option 2. Build from Source
---------------------------

While it is generally recommended to always use the prebuilt TVM Unity, if you require more customization, you may need to build it from source. **NOTE.** this should only be attempted if you are familiar with the intricacies of C++, CMake, LLVM, Python, and other related systems.

.. collapse:: Details

    **Step 1. Set up build dependency.** To build from source, you need to ensure that the following build dependencies are met:

    - CMake >= 3.24
    - LLVM >= 15
      - For please install LLVM>=17 for ROCm 6.1 and LLVM>=18 for ROCm 6.2.
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
        conda env remove -n tvm-build-venv
        # create the conda environment with build dependency
        conda create -n tvm-build-venv -c conda-forge \
            "llvmdev>=15" \
            "cmake>=3.24" \
            git \
            python=3.11
        # enter the build environment
        conda activate tvm-build-venv

    **Step 2. Configure and build.** Standard git-based workflow are recommended to download Apache TVM Unity, and then specify build requirements in ``config.cmake``:

    .. code-block:: bash
        :caption: Download TVM Unity from GitHub

        # clone from GitHub
        git clone --recursive https://github.com/mlc-ai/relax.git tvm-unity && cd tvm-unity
        # create the build directory
        rm -rf build && mkdir build && cd build
        # specify build requirements in `config.cmake`
        cp ../cmake/config.cmake .

    .. note::
        We are temporarily using `mlc-ai/relax <https://github.com/mlc-ai/relax>`_ instead, which comes with several temporary outstanding changes that we will upstream to Apache TVM's `unity branch <https://github.com/apache/tvm/tree/unity>`_.

    We want to specifically tweak the following flags by appending them to the end of the configuration file:

    .. code-block:: bash
        :caption: Configure build in ``config.cmake``

        # controls default compilation flags
        echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
        # LLVM is a must dependency
        echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
        echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
        # GPU SDKs, turn on if needed
        echo "set(USE_CUDA   OFF)" >> config.cmake
        echo "set(USE_METAL  OFF)" >> config.cmake
        echo "set(USE_VULKAN OFF)" >> config.cmake
        echo "set(USE_OPENCL OFF)" >> config.cmake
        # FlashInfer related, requires CUDA w/ compute capability 80;86;89;90
        echo "set(USE_FLASHINFER OFF)" >> config.cmake
        echo "set(FLASHINFER_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
        echo "set(CMAKE_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake

    .. note::
        ``HIDE_PRIVATE_SYMBOLS`` is a configuration option that enables the ``-fvisibility=hidden`` flag. This flag helps prevent potential symbol conflicts between TVM and PyTorch. These conflicts arise due to the frameworks shipping LLVMs of different versions.

        `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_ controls default compilation flag:

        - ``Debug`` sets ``-O0 -g``
        - ``RelWithDebInfo`` sets ``-O2 -g -DNDEBUG`` (recommended)
        - ``Release`` sets ``-O3 -DNDEBUG``

    .. note::
        If you are using CUDA and your compute capability is above 80, then it is require to build with
        ``set(USE_FLASHINFER ON)``. Otherwise, you may run into ``Cannot find Function`` issue during
        runtime.

        To check your CUDA compute capability, you can use ``nvidia-smi --query-gpu=compute_cap --format=csv``.

    Once ``config.cmake`` is edited accordingly, kick off build with the commands below:

    .. code-block:: bash
        :caption: Build ``libtvm`` using cmake and cmake

        cmake .. && cmake --build . --parallel $(nproc)

    A success build should produce ``libtvm`` and ``libtvm_runtime`` under ``/path-tvm-unity/build/`` directory.

    Leaving the build environment ``tvm-build-venv``, there are two ways to install the successful build into your environment:

    .. tabs ::

       .. code-tab :: bash Install via environment variable

          export PYTHONPATH=/path-to-tvm-unity/python:$PYTHONPATH

       .. code-tab :: bash Install via pip local project

          conda activate your-own-env
          conda install python # make sure python is installed
          cd /path-to-tvm-unity/python
          pip install -e .

.. `|` adds a blank line

|

.. _tvm-unity-validate:

Validate TVM Installation
-------------------------

Using a compiler infrastructure with multiple language bindings could be error-prone.
Therefore, it is highly recommended to validate TVM Unity installation before use.

**Step 1. Locate TVM Python package.** The following command can help confirm that TVM is properly installed as a python package and provide the location of the TVM python package:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.11/site-packages/tvm/__init__.py

**Step 2. Confirm which TVM library is used.** When maintaining multiple build or installation of TVM, it becomes important to double check if the python package is using the proper ``libtvm`` with the following command:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.base._LIB)"
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
    ``GIT_COMMIT_HASH`` indicates the exact commit of the TVM build, and it can be found on GitHub via ``https://github.com/mlc-ai/relax/commit/$GIT_COMMIT_HASH``.

**Step 4. Check device detection.** Sometimes it could be helpful to understand if TVM could detect your device at all with the following commands:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.metal().exist)"
    True # or False
    >>> python -c "import tvm; print(tvm.cuda().exist)"
    False # or True
    >>> python -c "import tvm; print(tvm.vulkan().exist)"
    False # or True

Please note that the commands above verify the presence of an actual device on the local machine for the TVM runtime (not the compiler) to execute properly. However, TVM compiler can perform compilation tasks without requiring a physical device. As long as the necessary toolchain, such as NVCC, is available, TVM supports cross-compilation even in the absence of an actual device.
