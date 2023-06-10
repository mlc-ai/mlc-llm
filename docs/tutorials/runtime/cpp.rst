Run Models in C++
=================

MLCChat CLI is the default runtime to run MLC-compiled models in comamnd line out of the box. Additionally, MLC-compiled models can be integrated into any C++ projects using TVM runtime without MLCChat CLI.

.. contents:: Table of Contents
   :depth: 3

Install MLCChat CLI
-------------------

MLCChat CLI is the command line tool to run MLC-compiled LLMs out of the box. You may install it from the prebuilt package we provide, or compile it from source.

Option 1. Conda Prebuilt
~~~~~~~~~~~~~~~~~~~~~~~~

The prebuilt package supports Metal on macOS and Vulkan on Linux and Windows, and can be installed via Conda one-liner.

To use other GPU runtimes, e.g. CUDA, please instead build it from source.

.. code:: shell

    conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-nightly
    conda activate mlc-chat-venv
    mlc_chat_cli --help

After installation, activating ``mlc-chat-venv`` environment in Conda will give the ``mlc_chat_cli`` command available.

.. note::
    The prebuilt package supports **Metal** on macOS and **Vulkan** on Linux and Windows. It is possible to use other GPU runtimes such as **CUDA** by compiling MLCChat CLI from source.

.. _mlcchat_build_from_source:

Option 2. Build from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

While it is always recommended to use prebuilt MLCChat, for more customization, one has to build ``mlc_chat_cli`` from source in the following steps:

.. collapse:: Details

    **Step 1. Set up build dependency.** To build from source, you need to ensure that the following build dependencies are met:

    * CMake >= 3.24
    * Git
    * Rust and Cargo, required by Huggingface's tokenizer
    * One of the GPU runtimes:

      * CUDA >= 11.8 (NVIDIA GPUs)
      * Metal (Apple GPUs)
      * Vulkan (NVIDIA, AMD, Intel GPUs)

    .. code-block:: bash
        :caption: Set up build dependencies in conda

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
        TVM Unity compiler is not a dependency to MLCChat CLI. Only its runtime is required, which is automatically included in `3rdparty/tvm <https://github.com/mlc-ai/mlc-llm/tree/main/3rdparty>`_.

    **Step 2. Configure and build.** Standard git-based workflow are recommended to download MLC LLM, and then specify build requirements with our small config generation tool:

    .. code-block:: bash
        :caption: Configure and build ``mlc_chat_cli``

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

  The input consists of three parts: :ref:`model lib <model_lib>` of optimized tensor computation, shards of quantized :ref:`model weights <model_weights>`, as well as a JSON configuration file :ref:`chat config <chat_config>`. They should be located under a directory uniquely specified by the model's ``local_id``, which is ``$(model_name)-$(quantize_mode)``, for example, "vicuna-v1-7b-q3f16_0" for int3-quantized Vicuna-7B, "RedPajama-INCITE-Chat-3B-v1-q4f16_0" for int4-quantized RedPajama-3B.

  .. tabs ::

     .. tab :: Check prebuilt models

        If you downloaded prebuilt models from MLC LLM, by default:

        - model lib should be placed under ``./dist/prebuilt/lib/$(local_id)-$(arch).$(suffix)``
        - model weights and chat config are located under ``./dist/prebuilt/mlc-chat-$(local_id)/``

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

        - model lib should be placed under ``./dist/$(local_id)/$(local_id)-$(arch).$(suffix)``
        - model weights and chat config are located under ``./dist/$(local_id)/params/``

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

Advanced Topic: Integrate Models in C++
---------------------------------------

MLC-compiled models can be integrated into any C++ project using TVM's C/C++ API without going through the command line.

**Step 1. Create ``libmlc_llm``.** Both static and shared libraries are available via the :ref:`CMake instructions <mlcchat_build_from_source>`, and the downstream developer may include either one into the C++ project depending on needs.

**Step 2. Calling into the model in your C++ Project.** Use ``tvm::runtime::Module`` API from TVM runtime to interact with MLC LLM without MLCChat.

.. note::
    `DLPack <https://dmlc.github.io/dlpack/latest/c_api.html>`_ that comes with TVM is an in-memory representation of tensors in deep learning. It is widely adopted in
    `NumPy <https://numpy.org/devdocs/reference/generated/numpy.from_dlpack.html>`_,
    `PyTorch <https://pytorch.org/docs/stable/dlpack.html>`_,
    `JAX <https://jax.readthedocs.io/en/latest/jax.dlpack.html>`_,
    `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/>`_,
    etc.

.. collapse:: Example code

  .. code:: c++

    #define TVM_USE_LIBBACKTRACE 0
    #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

    #include <tvm/runtime/packed_func.h>
    #include <tvm/runtime/module.h>
    #include <tvm/runtime/registry.h>

    // DLPack is a widely adopted in-memory representation of tensors in deep learning.
    #include <dlpack/dlpack.h>

    void ChatModule(
      const DLDevice& device,
      const std::string& path_model_lib,
      const std::string& path_weight_config
    ) {
      // PackedFunc is TVM's calling convention
      using tvm::runtime::PackedFunc;
      // Make sure the following files exist:
      // - model lib  : `$(path_model_lib)`
      // - chat config: `$(path_weight_config)/mlc-chat-config.json`
      // - weights    : `$(path_weight_config)/ndarray-cache.json`

      const PackedFunc* fcreate = tvm::runtime::Registry::Get("mlc.llm_chat_create");
      assert(fcreate != nullptr);
      llm_chat_ = (*fcreate)(
        static_cast<int>(kDLMetal), // kDLMetal comes from DLPack
        0
      );
      // Obtain all available functions in `libmlc_llm`
      tvm::runtime::Module chat_mod = mlc::llm::CreateChatModule(device);
      PackedFunc prefill_ = chat_mod->GetFunction("prefill");
      PackedFunc decode_ = chat_mod->GetFunction("decode");
      PackedFunc stopped_ = chat_mod->GetFunction("stopped");
      PackedFunc get_message_ = chat_mod->GetFunction("get_message");
      PackedFunc reload_ = chat_mod->GetFunction("reload");
      PackedFunc get_role0_ = chat_mod->GetFunction("get_role0");
      PackedFunc get_role1_ = chat_mod->GetFunction("get_role1");
      PackedFunc runtime_stats_text_ = chat_mod->GetFunction("runtime_stats_text");
      PackedFunc reset_chat_ = chat_mod->GetFunction("reset_chat");
      PackedFunc process_system_prompts_ = chat_mod->GetFunction("process_system_prompts");
      // Load model lib which contains optimized tensor computation
      tvm::runtime::Module model_lib = tvm::runtime::Module::LoadFromFile(path_model_lib);
      reload_(model_lib, path_weight_config);
    }

.. `|` adds a blank line

|

**Step 3. Set up compilation flags.** To properly compile the code above, you will have to set up compiler flags properly in your own C++ project:

- Make sure the following directories are included where ``TVM_HOME`` is ``/path/to/mlc-llm/3rdparty/tvm``:

  - TVM runtime: ``${TVM_HOME}/include``
  - Header-only DLPack: ``${TVM_HOME}/3rdparty/dlpack/include``
  - Header-only DMLC core: ``${TVM_HOME}/3rdparty/dmlc-core/include``

- Make sure to link either the static or the shared ``libtvm_runtime`` library, which is provided via :ref:`CMake <mlcchat_build_from_source>`.
