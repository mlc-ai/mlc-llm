.. _deploy-cli:

CLI and C++ API
===============

MLCChat CLI is the command line tool to run MLC-compiled LLMs out of the box. You may install it from the prebuilt package we provide, or compile it from source.

.. contents:: Table of Contents
  :local:
  :depth: 2

Option 1. Conda Prebuilt
~~~~~~~~~~~~~~~~~~~~~~~~

The prebuilt package supports Metal on macOS and Vulkan on Linux and Windows, and can be installed via Conda one-liner.

To use other GPU runtimes, e.g. CUDA, please instead :ref:`build it from source <mlcchat_build_from_source>`.

.. code:: shell

    conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-nightly
    conda activate mlc-chat-venv
    mlc_chat_cli --help

After installation, activating ``mlc-chat-venv`` environment in Conda will give the ``mlc_chat_cli`` command available.

.. note::
    The prebuilt package supports **Metal** on macOS and **Vulkan** on Linux and Windows. It is possible to use other GPU runtimes such as **CUDA** by compiling MLCChat CLI from source.

.. _mlcchat_build_from_source:

Option 2. Build MLC Runtime from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provid options to build mlc runtime libraries and ``mlc_chat_cli`` from source.
This step is useful when you want to directly obtain a version of mlc runtime library
and the cli. Please click the details below to see the instruction.

.. collapse:: Details

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
            git
        # enter the build environment
        conda activate mlc-chat-venv

    .. note::
        :doc:`TVM Unity </install/tvm>` compiler is not a dependency to MLCChat CLI. Only its runtime is required, which is automatically included in `3rdparty/tvm <https://github.com/mlc-ai/mlc-llm/tree/main/3rdparty>`_.

    **Step 2. Configure and build.** A standard git-based workflow is recommended to download MLC LLM, after which you can specify build requirements with our lightweight config generation tool:

    .. code-block:: bash
        :caption: Configure and build

        # clone from GitHub
        git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/
        # create build directory
        mkdir -p build && cd build
        # generate build configuration
        python3 ../cmake/gen_cmake_config.py
        # build `mlc_chat_cli`
        cmake .. && cmake --build . --parallel $(nproc) && cd ..
        # check if the file is generated successfully
        ls -l .

    **Step 3. Validate installation.** You may validate if MLCChat CLI is compiled successfully using the following command:

    .. code-block:: bash
        :caption: Validate installation

        # expected to see `mlc_chat_cli`, `libmlc_llm` and `libtvm_runtime`
        ls -l ./build/
        # expected to see help message
        ./build/mlc_chat_cli --help

.. `|` adds a blank line

|

Run Models through MLCChat CLI
------------------------------

Once ``mlc_chat_cli`` is installed, you are able to run any MLC-compiled model on comamnd line.

**Ensure Model Exists.** As the input to ``mlc_chat_cli``, it is always good to double check if the compiled model exists.

.. collapse:: Details

  .. tabs ::

     .. tab :: Check prebuilt models

        If you downloaded prebuilt models from MLC LLM, by default:

        - Model lib should be placed at ``./dist/prebuilt/lib/$(local_id)-$(arch).$(suffix)``.
        - Model weights and chat config are located under ``./dist/prebuilt/mlc-chat-$(local_id)/``.

        .. collapse:: Example

          .. code:: shell

            >>> ls -l ./dist/prebuilt/lib
            vicuna-v1-7b-q3f16_0-metal.so  # Format: $(local_id)-$(arch).$(suffix)
            vicuna-v1-7b-q3f16_0-vulkan.so
            ...
            >>> ls -l ./dist/prebuilt/mlc-chat-vicuna-v1-7b-q3f16_0  # Format: ./dist/prebuilt/mlc-chat-$(local_id)/
            # chat config:
            mlc-chat-config.json
            # model weights:
            ndarray-cache.json
            params_shard_*.bin
            ...

     .. tab :: Check compiled models

        If you have compiled models using MLC LLM, by default:

        - Model libraries should be placed at ``./dist/$(local_id)/$(local_id)-$(arch).$(suffix)``.
        - Model weights and chat config are located under ``./dist/$(local_id)/params/``.

        .. collapse:: Example

          .. code:: shell

            >>> ls -l ./dist/vicuna-v1-7b-q3f16_0/ # Format: ./dist/$(local_id)/
            vicuna-v1-7b-q3f16_0-metal.so  # Format: $(local_id)-$(arch).$(suffix)
            ...
            >>> ls -l ./dist/vicuna-v1-7b-q3f16_0/params  # Format: ``./dist/$(local_id)/params/``
            # chat config:
            mlc-chat-config.json
            # model weights:
            ndarray-cache.json
            params_shard_*.bin
            ...

|

**Run the Model.** Next run ``mlc_chat_cli`` in command line:

.. code:: shell

  # `local_id` is `$(model_name)-$(quantize_mode)`
  # In this example, `model_name` is `vicuna-v1-7b`, and `quantize_mode` is `q3f16_0`
  >>> mlc_chat_cli --local-id vicuna-v1-7b-q3f16_0
  Use MLC config: "....../mlc-chat-config.json"
  Use model weights: "....../ndarray-cache.json"
  Use model library: "....../vicuna-v1-7b-q3f16_0-metal.so"
  ...

Have fun chatting with MLC-compiled LLM!

Advanced: Build Apps with C++ API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MLC-compiled models can be integrated into **any C++ project** using TVM's C/C++ API without going through the command line.

**Step 1. Create libmlc_llm.** Both static and shared libraries are available via the :ref:`CMake instructions <mlcchat_build_from_source>`, and the downstream developer may include either one into the C++ project according to needs.

**Step 2. Calling into the model in your C++ Project.** Use ``tvm::runtime::Module`` API from TVM runtime to interact with MLC LLM without MLCChat.

.. note::
    `DLPack <https://dmlc.github.io/dlpack/latest/c_api.html>`_ that comes with TVM is an in-memory representation of tensors in deep learning. It is widely adopted in
    `NumPy <https://numpy.org/devdocs/reference/generated/numpy.from_dlpack.html>`_,
    `PyTorch <https://pytorch.org/docs/stable/dlpack.html>`_,
    `JAX <https://jax.readthedocs.io/en/latest/jax.dlpack.html>`_,
    `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/>`_,
    etc.

Using MLCChat APIs in Your Own Programs
---------------------------------------

Below is a minimal example of using MLCChat C++ APIs.

.. code:: c++

  #define TVM_USE_LIBBACKTRACE 0
  #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

  #include <tvm/runtime/packed_func.h>
  #include <tvm/runtime/module.h>
  #include <tvm/runtime/registry.h>

  // DLPack is a widely adopted in-memory representation of tensors in deep learning.
  #include <dlpack/dlpack.h>

  void ChatModule(
    const DLDeviceType& device_type, // from dlpack.h
    int device_id, // which one if there are multiple devices, usually 0
    const std::string& path_model_lib,
    const std::string& path_weight_config
  ) {
    // Step 0. Make sure the following files exist:
    // - model lib  : `$(path_model_lib)`
    // - chat config: `$(path_weight_config)/mlc-chat-config.json`
    // - weights    : `$(path_weight_config)/ndarray-cache.json`
    using tvm::runtime::PackedFunc;

    // Step 1. Call `mlc.llm_chat_create`
    // This method will exist if `libmlc_llm` is successfully loaded or linked as a shared or static library.
    const PackedFunc* llm_chat_create = tvm::runtime::Registry::Get("mlc.llm_chat_create");
    assert(llm_chat_create != nullptr);
    tvm::runtime::Module mlc_llm = (*llm_chat_create)(
      static_cast<int>(device_type),
      device_id,
    );
    // Step 2. Obtain all available functions in `mlc_llm`
    PackedFunc prefill = mlc_llm->GetFunction("prefill");
    PackedFunc decode = mlc_llm->GetFunction("decode");
    PackedFunc stopped = mlc_llm->GetFunction("stopped");
    PackedFunc get_message = mlc_llm->GetFunction("get_message");
    PackedFunc reload = mlc_llm->GetFunction("reload");
    PackedFunc get_role0 = mlc_llm->GetFunction("get_role0");
    PackedFunc get_role1 = mlc_llm->GetFunction("get_role1");
    PackedFunc runtime_stats_text = mlc_llm->GetFunction("runtime_stats_text");
    PackedFunc reset_chat = mlc_llm->GetFunction("reset_chat");
    PackedFunc process_system_prompts = mlc_llm->GetFunction("process_system_prompts");
    // Step 3. Load the model lib containing optimized tensor computation
    tvm::runtime::Module model_lib = tvm::runtime::Module::LoadFromFile(path_model_lib);
    // Step 4. Inform MLC LLM to use `model_lib`
    reload(model_lib, path_weight_config);
  }

.. note::

  MLCChat CLI can be considered as a `single-file <https://github.com/mlc-ai/mlc-llm/blob/main/cpp/cli_main.cc>`_ project serving a good example of using MLC LLM in any C++ project.


**Step 3. Set up compilation flags.** To properly compile the code above, you will have to set up compiler flags properly in your own C++ project:

- Make sure the following directories are included where ``TVM_HOME`` is ``/path/to/mlc-llm/3rdparty/tvm``:

  - TVM runtime: ``${TVM_HOME}/include``,
  - Header-only DLPack: ``${TVM_HOME}/3rdparty/dlpack/include``,
  - Header-only DMLC core: ``${TVM_HOME}/3rdparty/dmlc-core/include``.

- Make sure to link either the static or the shared ``libtvm_runtime`` library, which is provided via :ref:`CMake <mlcchat_build_from_source>`.
