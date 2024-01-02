.. _deploy-cli:

CLI and C++ API
===============

MLCChat CLI is the command line tool to run MLC-compiled LLMs out of the box. You may install it from the prebuilt package we provide, or compile it from the source.

.. contents:: Table of Contents
  :local:
  :depth: 2

Option 1. Conda Prebuilt
~~~~~~~~~~~~~~~~~~~~~~~~

The prebuilt package supports Metal on macOS and Vulkan on Linux and Windows, and can be installed via Conda one-liner.

To use other GPU runtimes, e.g. CUDA, please instead :ref:`build it from source <mlcchat_build_from_source>`.

.. code:: shell

    conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-cli-nightly
    conda activate mlc-chat-venv
    mlc_chat_cli --help

After installation, activating ``mlc-chat-venv`` environment in Conda will give the ``mlc_chat_cli`` command available.

.. note::
    The prebuilt package supports **Metal** on macOS and **Vulkan** on Linux and Windows. It is possible to use other GPU runtimes such as **CUDA** by compiling MLCChat CLI from the source.


Option 2. Build MLC Runtime from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide options to build mlc runtime libraries and ``mlc_chat_cli`` from source.
This step is useful if the prebuilt is unavailable on your platform, or if you would like to build a runtime
that supports other GPU runtime than the prebuilt version. We can build a customized version
of mlc chat runtime. You only need to do this if you choose not to use the prebuilt.

First, make sure you install TVM unity (following the instruction in :ref:`install-tvm-unity`).
You can choose to only pip install ``mlc-ai-nightly`` that comes with the tvm unity but skip ``mlc-chat-nightly``.
Then please follow the instructions in :ref:`mlcchat_build_from_source` to build the necessary libraries.

.. `|` adds a blank line

|

Run Models through MLCChat CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ``mlc_chat_cli`` is installed, you are able to run any MLC-compiled model on the command line.

To run a model with MLC LLM in any platform, you need:

1. **Model weights** converted to MLC format (e.g. `RedPajama-INCITE-Chat-3B-v1-MLC 
   <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-MLC/tree/main>`_.)
2. **Model library** that comprises the inference logic (see repo `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__).

There are two ways to obtain the model weights and libraries:

1. Compile your own model weights and libraries following :doc:`the model compilation page </compilation/compile_models>`.
2. Use off-the-shelf `prebuilt models weights <https://huggingface.co/mlc-ai>`__ and
   `prebuilt model libraries <https://github.com/mlc-ai/binary-mlc-llm-libs>`__ (see :ref:`Model Prebuilts` for details).

We use off-the-shelf prebuilt models in this page. However, same steps apply if you want to run
the models you compiled yourself.

**Step 1: Download prebuilt model weights and libraries**

Skip this step if you have already obtained the model weights and libraries.

.. code:: shell

  # Download pre-conveted weights
  git lfs install && mkdir dist/
  git clone https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC \
                                     dist/Llama-2-7b-chat-hf-q4f16_1-MLC

  # Download pre-compiled model library
  git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs


**Step 2: Run in command line**

To run ``mlc_chat_cli``, simply 

- Specify the path to the converted model weights to ``--model``
- Specify the path to the compiled model library to ``--model-lib-path``

.. code:: shell

  mlc_chat_cli --model dist/Llama-2-7b-chat-hf-q4f16_1-MLC \
               --model-lib-path dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-vulkan.so
               # CUDA on Linux: dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so
               # Metal on macOS: dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-metal.so
               # Same rule applies for other platforms

.. code:: shell

  >>> Use MLC config: "....../mlc-chat-config.json"
  >>> Use model weights: "....../ndarray-cache.json"
  >>> Use model library: "....../Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}"
  >>> ...

**Running other models**

Checkout the :doc:`/prebuilt_models` page to run other pre-compiled models.

For models other than the prebuilt ones we provided:

1. If the model is a variant to an existing model library (e.g. ``WizardMathV1.1`` and ``OpenHermes`` are variants of ``Mistral``),
   follow :ref:`convert-weights-via-MLC` to convert the weights and reuse existing model libraries.
2. Otherwise, follow :ref:`compile-model-libraries` to compile both the model library and weights.

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
