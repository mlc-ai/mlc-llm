C++ APIs
========

.. contents:: Table of Contents
   :local:
   :depth: 2

We provide MLC-Chat C++ APIs for developers to integrate MLC-Chat into their own C++ projects.

Advanced Topic: Integrate Models in C++
---------------------------------------

MLC-compiled models can be integrated into any C++ project using TVM's C/C++ API without going through the command line.

**Step 1. Create libmlc_llm.** Both static and shared libraries are available via the :ref:`CMake instructions <mlcchat_build_from_source>`, and the downstream developer may include either one into the C++ project depending on needs.

**Step 2. Calling into the model in your C++ Project.** Use ``tvm::runtime::Module`` API from TVM runtime to interact with MLC LLM without MLCChat.

.. note::
    `DLPack <https://dmlc.github.io/dlpack/latest/c_api.html>`_ that comes with TVM is an in-memory representation of tensors in deep learning. It is widely adopted in
    `NumPy <https://numpy.org/devdocs/reference/generated/numpy.from_dlpack.html>`_,
    `PyTorch <https://pytorch.org/docs/stable/dlpack.html>`_,
    `JAX <https://jax.readthedocs.io/en/latest/jax.dlpack.html>`_,
    `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/>`_,
    etc.

Using MLCChat APIs in your own programs
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

  - TVM runtime: ``${TVM_HOME}/include``
  - Header-only DLPack: ``${TVM_HOME}/3rdparty/dlpack/include``
  - Header-only DMLC core: ``${TVM_HOME}/3rdparty/dmlc-core/include``

- Make sure to link either the static or the shared ``libtvm_runtime`` library, which is provided via :ref:`CMake <mlcchat_build_from_source>`.
