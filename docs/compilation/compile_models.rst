.. _compile-models-via-MLC:

Compile Models via MLC
======================

This page describes how to compile a model with MLC LLM. Model compilation takes model inputs, produces quantized model weights,
and optimizes the model library for a given platform. It enables users to bring their own new model weights, try different quantization modes,
and customize the overall model optimization flow.

.. note::
    Before you proceed, please make sure that you have installed :ref:`install-tvm-unity`, a required
    backend to compile models with MLC LLM. If you want to build for webgpu, please also complete :ref:`install-web-build`.
    
    Please also follow the instructions in :ref:`deploy-cli` / :ref:`deploy-python` to obtain the CLI app / Python API that can be used to chat with the compiled model.
    Finally, we strongly recommend you to read :ref:`project-overview` first to get familiarized with the high-level terminologies.


.. contents:: Table of Contents
    :depth: 1
    :local:

Verify Installation
-------------------

**Step 1. Verify mlc_chat**

We use the python package ``mlc_chat`` to compile models. This can be installed by 
following :ref:`install-mlc-packages`, either by building from source, or by
installing the prebuilt package. Verify ``mlc_chat`` installation in command line via:

.. code:: bash

    $ mlc_chat --help
    # You should see help information with this line
    usage: MLC LLM Command Line Interface. [-h] {compile,convert_weight,gen_config}

.. note::
    If it runs into error ``command not found: mlc_chat``, try ``python -m mlc_chat --help``.

**Step 2. Verify TVM**

To compile models, you also need to follow :ref:`install-tvm-unity`.
Here we verify ``tvm`` quickly with command line (for full verification, see :ref:`tvm-unity-validate`):

.. code:: bash

    $ python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.11/site-packages/tvm/__init__.py


Get Started
-----------

This section provides a step by step instructions to guide you through the compilation
process of ``RedPajama-INCITE-Chat-3B-v1`` as an example. You can select the platform
where you want to **run** your model from the tabs below and run the corresponding commands.
We strongly recommend you **start with Metal/CUDA/Vulkan** as it is easier to validate the compilation result on
your personal computer.

**Step 1. Clone HF weights and convert to MLC weights**

Regardless of the platform, we need to create directories and clone HF weights first.
You can be under the mlc-llm repo, or your own working directory.

.. code:: shell

    mkdir -p dist/rp_q4f16_1 && mkdir dist/models && cd dist/models
    git lfs install
    git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
    cd ../..

Then convert the HF weights into MLC-compatible weights. Note that all platforms can share the same compiled/quantized weights.

.. code:: shell

    mlc_chat convert_weight ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 -o dist/rp_q4f16_1/params

**Step 2. Generate mlc-chat-config and compile**

.. tabs::

    .. group-tab:: Linux - CUDA

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device cuda -o dist/rp_q4f16_1/rp_q4f16_1.so


    .. group-tab:: Metal

        For M-chip Mac:

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device metal -o dist/rp_q4f16_1/rp_q4f16_1.so

        For Intel Mac:

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device metal -o dist/rp_q4f16_1/rp_q4f16_1.dylib


    .. group-tab:: Vulkan

        For Linux: 

        .. code:: shell
            
            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device vulkan -o dist/rp_q4f16_1/rp_q4f16_1.so

        For Windows: 

        .. code:: shell
            
            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device vulkan -o dist/rp_q4f16_1/rp_q4f16_1.dll

    .. group-tab:: iOS/iPadOS

        You need a Mac to compile models for it.

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat --context-window-size 768 -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device iphone -o dist/rp_q4f16_1/rp_q4f16_1.tar

        .. note::
            If it runs into error

            .. code:: text

                Compilation error:
                xcrun: error: unable to find utility "metal", not a developer tool or in PATH
                xcrun: error: unable to find utility "metallib", not a developer tool or in PATH

            , please check and make sure you have Command Line Tools for Xcode installed correctly.
            You can use ``xcrun metal`` to validate: when it prints ``metal: error: no input files``, it means the Command Line Tools for Xcode is installed and can be found, and you can proceed with the model compiling.

    .. group-tab:: Android

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat --context-window-size 768 -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device android -o dist/rp_q4f16_1/rp_q4f16_1.tar

    .. group-tab:: WebGPU

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_chat gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 --conv-template redpajama_chat -o dist/rp_q4f16_1/params/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_chat compile ./dist/rp_q4f16_1/params/mlc-chat-config.json --device webgpu -o dist/rp_q4f16_1/rp_q4f16_1.wasm

        .. note::
            To compile for webgpu, you need to build from source when installing ``mlc_chat``. Besides, you also need to follow :ref:`install-web-build`.
            Otherwise, it would run into error

            .. code:: text

                RuntimeError: Cannot find libraries: wasm_runtime.bc

        .. note::
            For webgpu, when compiling larger models like ``Llama-2-7B``, you may want to add ``--prefill_chunk_size 1024`` or lower ``context_window_size`` to decrease memory usage.
            Otherwise, you may run into issues like:

            .. code:: text

                TypeError: Failed to execute 'createBuffer' on 'GPUDevice': Failed to read the 'size' property from
                'GPUBufferDescriptor': Value is outside the 'unsigned long long' value range.

**Step 3. Verify output and chat**

By executing the compile command above, we generate the model weights, model lib, and a chat config.
We can check the output with the commands below:

.. tabs::

    .. group-tab:: Linux - CUDA

        .. code:: shell

            ~/mlc-llm > ls dist/rp_q4f16_1
              rp_q4f16_1.so                                    # ===> the model library
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/rp_q4f16_1/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We can now chat with the model using the command line interface (CLI) app or the Python API.

        .. code:: shell

            python
            >>> from mlc_chat import ChatModule
            >>> cm = ChatModule(model="./dist/rp_q4f16_1/params", model_lib_path="./dist/rp_q4f16_1/rp_q4f16_1.so")
            >>> cm.generate("hi")
            'Hi! How can I assist you today?'

    .. group-tab:: Metal

        .. code:: shell

            ~/mlc-llm > ls dist/rp_q4f16_1
              rp_q4f16_1.so                                    # ===> the model library (will be .dylib for Intel Mac)
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/rp_q4f16_1/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We can now chat with the model using the command line interface (CLI) app or the Python API.

        .. code:: shell

            python
            >>> from mlc_chat import ChatModule
            >>> cm = ChatModule(model="./dist/rp_q4f16_1/params", model_lib_path="./dist/rp_q4f16_1/rp_q4f16_1.so")
            >>> cm.generate("hi")
            'Hi! How can I assist you today?'


    .. group-tab:: Vulkan

        .. code:: shell

            ~/mlc-llm > ls dist/rp_q4f16_1
              rp_q4f16_1.so                                    # ===> the model library (will be .dll for Windows)
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/rp_q4f16_1/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We can now chat with the model using the command line interface (CLI) app or the Python API.

        .. code:: shell

            python
            >>> from mlc_chat import ChatModule
            >>> cm = ChatModule(model="./dist/rp_q4f16_1/params", model_lib_path="./dist/rp_q4f16_1/rp_q4f16_1.so", device="vulkan")
            >>> cm.generate("hi")
            'Hi! How can I assist you today?'

    .. group-tab:: iOS/iPadOS

        .. code:: shell

            ~/mlc-llm > ls dist/rp_q4f16_1
              rp_q4f16_1.tar                                   # ===> the model library
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/rp_q4f16_1/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        The model lib ``dist/rp_q4f16_1/rp_q4f16_1.tar``
        will be packaged as a static library into the iOS app. Checkout :ref:`deploy-ios` for more details.

    .. group-tab:: Android

        .. code:: shell

            ~/mlc-llm > ls dist/rp_q4f16_1
              rp_q4f16_1.tar                                   # ===> the model library
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/rp_q4f16_1/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        The model lib ``dist/rp_q4f16_1/rp_q4f16_1.tar``
        will be packaged as a static library into the android app. Checkout :ref:`deploy-android` for more details.

    .. group-tab:: WebGPU

        .. code:: shell

            ~/mlc-llm > ls dist/rp_q4f16_1
              rp_q4f16_1.wasm                                  # ===> the model library
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/rp_q4f16_1/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        To use this in WebGPU runtime, checkout :ref:`webllm-runtime`.


Each compilation target produces a specific model library for the given platform. The model weight is shared across
different targets. If you are interested in distributing the model besides local execution, please checkout :ref:`distribute-compiled-models`.
You are also more than welcome to read the following sections for more details about the compilation.

Compile Commands for More Models
--------------------------------

This section lists compile commands for more models that you can try out. Note that this can be easily
generalized to any model variant, as long as mlc-llm supports the architecture.

.. tabs::

    .. tab:: Model: Llama-2-7B

        Please `request for access <https://huggingface.co/meta-llama>`_ to the Llama-2 weights from Meta first.
        After granted access, first create directory ``dist/models`` and download the model to the directory.
        For example, you can run the following code:

        .. code:: shell

            mkdir -p dist/llama_q4f16_1 && mkdir dist/models && cd dist/models
            cd dist/models
            git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
            cd ../..

        Then convert the HF weights into MLC-compatible weights. Note that all platforms
        can share the same compiled/quantized weights.

        .. code:: shell

            mlc_chat convert_weight ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 -o dist/llama_q4f16_1/params
        
        Afterwards, run the following command to generate mlc config and compile the model.

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --conv-template llama-2 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device cuda -o dist/llama_q4f16_1/llama_q4f16_1.so

            .. tab:: Metal

                For M-chip Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --conv-template llama-2 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device metal -o dist/llama_q4f16_1/llama_q4f16_1.so


                For Intel Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --conv-template llama-2 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device metal -o dist/llama_q4f16_1/llama_q4f16_1.dylib

            .. tab:: Vulkan

                For Linux: 

                .. code:: shell
                    
                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --conv-template llama-2 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device vulkan -o dist/llama_q4f16_1/llama_q4f16_1.so

                For Windows: 

                .. code:: shell
                    
                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --conv-template llama-2 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device vulkan -o dist/llama_q4f16_1/llama_q4f16_1.dll

            .. tab:: WebGPU

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --context-window-size 2048 --conv-template llama-2 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device webgpu -o dist/llama_q4f16_1/llama_q4f16_1.wasm

                .. note::
                    To compile for webgpu, you need to build from source when installing ``mlc_chat``. Besides, you also need to follow :ref:`install-web-build`.
                    Otherwise, it would run into error

                    .. code:: text

                        RuntimeError: Cannot find libraries: wasm_runtime.bc

            .. tab:: iPhone/iPad

                You need a Mac to compile models for it.

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --conv-template llama-2 --context-window-size 768 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device iphone -o dist/llama_q4f16_1/llama_q4f16_1.tar

            .. tab:: Android

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 --conv-template llama-2 --context-window-size 768 -o dist/llama_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/llama_q4f16_1/params/mlc-chat-config.json --device android -o dist/llama_q4f16_1/llama_q4f16_1.tar

    .. tab:: Mistral-7B-Instruct-v0.1

        Note that Mistral uses sliding window attention (SWA). Thus, instead of specifying
        ``context-window-size``, we specify ``sliding-window-size``.

        First create directory ``dist/models`` and download the model to the directory.
        For example, you can run the following code:

        .. code:: shell

            mkdir -p dist/mistral_q4f16_1 && mkdir dist/models && cd dist/models
            cd dist/models
            git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
            cd ../..

        Then convert the HF weights into MLC-compatible weights. Note that all platforms
        can share the same compiled/quantized weights.

        .. code:: shell

            mlc_chat convert_weight ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 -o dist/mistral_q4f16_1/params

        Afterwards, run the following command to generate mlc config and compile the model.

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --conv-template mistral_default -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device cuda -o dist/mistral_q4f16_1/mistral_q4f16_1.so

            .. tab:: Metal

                For M-chip Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --conv-template mistral_default -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device metal -o dist/mistral_q4f16_1/mistral_q4f16_1.so


                For Intel Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --conv-template mistral_default -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device metal -o dist/mistral_q4f16_1/mistral_q4f16_1.dylib

            .. tab:: Vulkan

                For Linux: 

                .. code:: shell
                    
                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --conv-template mistral_default -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device vulkan -o dist/mistral_q4f16_1/mistral_q4f16_1.so

                For Windows: 

                .. code:: shell
                    
                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --conv-template mistral_default -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device vulkan -o dist/mistral_q4f16_1/mistral_q4f16_1.dll

            .. tab:: WebGPU

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --prefill-chunk-size 1024 --conv-template mistral_default -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device webgpu -o dist/mistral_q4f16_1/mistral_q4f16_1.wasm

                .. note::
                    To compile for webgpu, you need to build from source when installing ``mlc_chat``. Besides, you also need to follow :ref:`install-web-build`.
                    Otherwise, it would run into error

                    .. code:: text

                        RuntimeError: Cannot find libraries: wasm_runtime.bc

                .. note::
                    For webgpu, when compiling larger models like ``Llama-2-7B``, you may want to add ``--prefill_chunk_size 1024`` or lower ``context_window_size`` to decrease memory usage.
                    Otherwise, you may run into issues like:

                    .. code:: text

                        TypeError: Failed to execute 'createBuffer' on 'GPUDevice': Failed to read the 'size' property from
                        'GPUBufferDescriptor': Value is outside the 'unsigned long long' value range.

            .. tab:: iPhone/iPad

                You need a Mac to compile models for it.

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --conv-template mistral_default --sliding-window-size 1024 --prefill-chunk-size 128  -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device iphone -o dist/mistral_q4f16_1/mistral_q4f16_1.tar

            .. tab:: Android

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/Mistral-7B-Instruct-v0.1/ --quantization q4f16_1 --conv-template mistral_default --sliding-window-size 1024 --prefill-chunk-size 128 -o dist/mistral_q4f16_1/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/mistral_q4f16_1/params/mlc-chat-config.json --device android -o dist/mistral_q4f16_1/mistral_q4f16_1.tar

    .. tab:: Other models

        First create directory ``dist/models`` and download the model to the directory.
        For example, you can run the following code:

        .. code:: shell

            mkdir -p dist/OUTPUT && mkdir dist/models && cd dist/models
            cd dist/models
            git clone https://huggingface.co/DISTRIBUTOR/HF_MODEL
            cd ../..

        Then convert the HF weights into MLC-compatible weights. Note that all platforms
        can share the same compiled/quantized weights.

        .. code:: shell

            mlc_chat convert_weight ./dist/models/HF_MODEL/ --quantization q4f16_1 -o dist/OUTPUT/params

        Afterwards, run the following command to generate mlc config and compile the model.

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device cuda -o dist/OUTPUT/OUTPUT.so

            .. tab:: Metal

                For M-chip Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device metal -o dist/OUTPUT/OUTPUT.so


                For Intel Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device metal -o dist/OUTPUT/OUTPUT.dylib

            .. tab:: Vulkan

                For Linux: 

                .. code:: shell
                    
                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device vulkan -o dist/OUTPUT/OUTPUT.so

                For Windows: 

                .. code:: shell
                    
                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device vulkan -o dist/OUTPUT/OUTPUT.dll

            .. tab:: WebGPU

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device webgpu -o dist/OUTPUT/OUTPUT.wasm

                .. note::
                    To compile for webgpu, you need to build from source when installing ``mlc_chat``. Besides, you also need to follow :ref:`install-web-build`.
                    Otherwise, it would run into error

                    .. code:: text

                        RuntimeError: Cannot find libraries: wasm_runtime.bc

                .. note::
                    For webgpu, when compiling larger models like ``Llama-2-7B``, you may want to add ``--prefill_chunk_size 1024`` or lower ``context_window_size`` to decrease memory usage.
                    Otherwise, you may run into issues like:

                    .. code:: text

                        TypeError: Failed to execute 'createBuffer' on 'GPUDevice': Failed to read the 'size' property from
                        'GPUBufferDescriptor': Value is outside the 'unsigned long long' value range.

            .. tab:: iPhone/iPad

                You need a Mac to compile models for it.

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE --context-window-size 768 -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device iphone -o dist/OUTPUT/OUTPUT.tar

            .. tab:: Android

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_chat gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE --context-window-size 768 -o dist/OUTPUT/params/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_chat compile ./dist/OUTPUT/params/mlc-chat-config.json --device android -o dist/OUTPUT/OUTPUT.tar


For each model and each backend, the above only provides the most recommended build command (which is the most optimized).
You can also try with different argument values (e.g., different quantization modes, context window size, etc.),
whose build results affect runtime memory requirement, and it is possible that they may not run as
fast and robustly as the provided one when running the model.

.. note::
    Uing 3-bit quantization usually can be overly aggressive and only works for limited settings.
    If you encounter issues where the compiled model does not perform as expected,
    consider utilizing a higher number of bits for quantization (e.g., 4-bit quantization).

If you are interested in distributing the model besides local execution, please checkout :ref:`distribute-compiled-models`.


.. _compile-command-specification:

Compile Command Specification
-----------------------------

As you have seen in the section above, the model compilation is split into three steps: convert weights, generate
``mlc-chat-config.json``, and compile the model. This section describes the list of options that can be used
during compilation.

1. Convert Weight
^^^^^^^^^^^^^^^^^

Weight conversion command follows the pattern below:

.. code:: text

    mlc_chat convert_weight \
        CONFIG \
        --quantization QUANTIZATION_MODE \
        [--model-type MODEL_TYPE] \
        [--device DEVICE] \
        [--source SOURCE] \
        [--source-format SOURCE_FORMAT] \
        --output OUTPUT

Note that ``CONFIG`` is a positional argument. Arguments wrapped with ``[ ]`` are optional.

--CONFIG                            It can be one of the following:

                                    1. Path to a HuggingFace model directory that contains a ``config.json`` or
                                    2. Path to ``config.json`` in HuggingFace format, or
                                    3. The name of a pre-defined model architecture.

                                    A ``config.json`` file in HuggingFace format defines the model architecture, including the vocabulary
                                    size, the number of layers, the hidden size, number of attention heads, etc.
                                    Example: https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json.

                                    A HuggingFace directory often contains a ``config.json`` which defines the model architecture,
                                    the non-quantized model weights in PyTorch or SafeTensor format, tokenizer configurations,
                                    as well as an optional ``generation_config.json`` provides additional default configuration for
                                    text generation.
                                    Example: https://huggingface.co/codellama/CodeLlama-7b-hf/tree/main.

                                    For existing pre-defined model architecture, see ``MODEL_PRESETS``
                                    `here <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_chat/compiler/model/model.py>`_.

--quantization QUANTIZATION_MODE    The quantization mode we use to compile.

                                    See :ref:`quantization_mode` for more information.
                                    Available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and
                                    ``q4f16_awq``.

                                    We encourage you to use 4-bit quantization, as the text generated by 3-bit
                                    quantized models may have bad quality depending on the model.

--model-type MODEL_TYPE             Model architecture such as "llama". If not set, it is inferred from ``config.json``.

--device DEVICE                     The device used to do quantization such as "cuda" or "cuda:0". Will detect from
                                    local available GPUs if not specified.

--source SOURCE                     The path to original model weight, infer from ``config`` if missing.

--source-format SOURCE_FORMAT       The format of source model weight, infer from ``config`` if missing.

--output OUTPUT                     The output directory to save the quantized model weight.
                                    Will create ``params_shard_*.bin`` and ```ndarray-cache.json``` in this directory.

2. Generate MLC Chat Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to compile a model, we first need to generate the ``mlc-chat-config.json``. This file contains specifications
like ``context-window-size`` and ``sliding-window-size``, among others that can alter the model compiled. We also process
tokenizers in this step.

Config generation command follows the pattern below:

.. code:: text

    mlc_chat gen_config \
        CONFIG \
        --quantization QUANTIZATION_MODE \
        [--model-type MODEL_TYPE] \
        --conv-template CONV_TEMPLATE \
        [--context-window-size CONTEXT_WINDOW_SIZE] \
        [--sliding-window-size SLIDING_WINDOW_SIZE] \
        [--prefill-chunk-size PREFILL_CHUNK_SIZE] \
        [--tensor-parallel-shard TENSOR_PARALLEL_SHARDS] \
        --output OUTPUT

Note that ``CONFIG`` is a positional argument. Arguments wrapped with ``[ ]`` are optional.

--CONFIG                                        It can be one of the following:

                                                1. Path to a HuggingFace model directory that contains a ``config.json`` or
                                                2. Path to ``config.json`` in HuggingFace format, or
                                                3. The name of a pre-defined model architecture.

                                                A ``config.json`` file in HuggingFace format defines the model architecture, including the vocabulary
                                                size, the number of layers, the hidden size, number of attention heads, etc.
                                                Example: https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json.

                                                A HuggingFace directory often contains a ``config.json`` which defines the model architecture,
                                                the non-quantized model weights in PyTorch or SafeTensor format, tokenizer configurations,
                                                as well as an optional ``generation_config.json`` provides additional default configuration for
                                                text generation.
                                                Example: https://huggingface.co/codellama/CodeLlama-7b-hf/tree/main.

                                                For existing pre-defined model architecture, see ``MODEL_PRESETS``
                                                `here <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_chat/compiler/model/model.py>`_.

--quantization QUANTIZATION_MODE                The quantization mode we use to compile.

                                                See :ref:`quantization_mode` for more information.
                                                Available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and
                                                ``q4f16_awq``.

                                                We encourage you to use 4-bit quantization, as the text generated by 3-bit
                                                quantized models may have bad quality depending on the model.

--model-type MODEL_TYPE                         Model architecture such as "llama". If not set, it is inferred from ``config.json``.

--conv-template CONV_TEMPLATE                   Conversation template. It depends on how the model is tuned. Use "LM" for vanilla base model
                                                For existing pre-defined templates, see ``CONV_TEMPLATES``
                                                `here <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_chat/compiler/model/model.py>`_.

--context-window-size CONTEXT_WINDOW_SIZE       Option to provide the maximum sequence length supported by the model.
                                                This is usually explicitly shown as context length or context window in the model card.
                                                If this option is not set explicitly, by default, 
                                                it will be determined by ``context_window_size`` or ``max_position_embeddings`` in ``config.json``,
                                                and the latter is usually inaccurate for some models.

--sliding-window-size SLIDING_WINDOW            (Experimental) The sliding window size in sliding window attention (SWA).
                                                This optional field overrides the ``sliding_window`` in ``config.json`` for
                                                those models that use SWA. Currently only useful when compiling mistral-based models.
                                                This flag subjects to future refactoring.

--prefill-chunk-size PREFILL_CHUNK_SIZE         (Experimental) The chunk size during prefilling. By default,
                                                the chunk size is the same as ``context_window_size`` or ``sliding_window_size``.
                                                This flag subjects to future refactoring.

--tensor-parallel-shard TENSOR_PARALLEL_SHARDS  Number of shards to split the model into in tensor parallelism multi-gpu inference.

--output OUTPUT                                 The output directory for generated configurations, including `mlc-chat-config.json` and tokenizer configuration.

3. Compile Model Library
^^^^^^^^^^^^^^^^^^^^^^^^

After generating ``mlc-chat-config.json``, we can compile the model into a model library (files ending in ``.so``, ``.tar``, etc. that contains
the inference logic of a model).

Model compilation command follows the pattern below:

.. code:: text

    mlc_chat compile \
        MODEL \
        [--quantization QUANTIZATION_MODE] \
        [--model-type MODEL_TYPE] \
        [--device DEVICE] \
        [--host HOST] \
        [--opt OPT] \
        [--system-lib-prefix SYSTEM_LIB_PREFIX] \
        --output OUTPUT \
        [--overrides OVERRIDES]

Note that ``MODEL`` is a positional argument. Arguments wrapped with ``[ ]`` are optional.

--MODEL                                     A path to ``mlc-chat-config.json``, or an MLC model directory that contains ``mlc-chat-config.json``.

--quantization QUANTIZATION_MODE            The quantization mode we use to compile. If unprovided, will infer from ``MODEL``.

                                            See :ref:`quantization_mode` for more information.
                                            Available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and
                                            ``q4f16_awq``.

                                            We encourage you to use 4-bit quantization, as the text generated by 3-bit
                                            quantized models may have bad quality depending on the model.

--model-type MODEL_TYPE                     Model architecture such as "llama". If not set, it is inferred from ``mlc-chat-config.json``.

--device DEVICE                             The GPU device to compile the model to. If not set, it is inferred from GPUs available locally.

--host HOST                                 The host LLVM triple to compile the model to. If not set, it is inferred from the local CPU and OS.
                                            Examples of the LLVM triple:

                                            1) iPhones: arm64-apple-ios;
                                            2) ARM64 Android phones: aarch64-linux-android;
                                            3) WebAssembly: wasm32-unknown-unknown-wasm;
                                            4) Windows: x86_64-pc-windows-msvc;
                                            5) ARM macOS: arm64-apple-darwin.

--opt OPT                                   Optimization flags. MLC LLM maintains a predefined set of optimization flags,
                                            denoted as ``O0``, ``O1``, ``O2``, ``O3``, where ``O0`` means no optimization, ``O2``
                                            means majority of them, and ``O3`` represents extreme optimization that could
                                            potentially break the system.
                                            
                                            Meanwhile, optimization flags could be explicitly specified via details knobs, e.g.
                                            ``--opt="cutlass_attn=1;cutlass_norm=0;cublas_gemm=0;cudagraph=0"``.

--system-lib-prefix SYSTEM_LIB_PREFIX       Adding a prefix to all symbols exported. Similar to ``objcopy --prefix-symbols``.
                                            This is useful when compiling multiple models into a single library to avoid symbol
                                            conflicts. Different from objcopy, this takes no effect for shared library.


--output OUTPUT                             The path to the output file. The suffix determines if the output file is a shared library or
                                            objects. Available suffixes:

                                            1) Linux: .so (shared), .tar (objects);
                                            2) macOS: .dylib (shared), .tar (objects);
                                            3) Windows: .dll (shared), .tar (objects);
                                            4) Android, iOS: .tar (objects);
                                            5) Web: .wasm (web assembly).

--overrides OVERRIDES                       Model configuration override. Configurations to override ``mlc-chat-config.json``. Supports
                                            ``context_window_size``, ``prefill_chunk_size``, ``sliding_window``, ``max_batch_size`` and
                                            ``tensor_parallel_shards``. Meanwhile, model config could be explicitly specified via details
                                            knobs, e.g. ``--overrides "context_window_size=1024;prefill_chunk_size=128"``.
