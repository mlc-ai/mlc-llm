Compile Models via MLC
======================

This page describes how to compile a model with MLC LLM. Model compilation takes model inputs, produces quantized model weights,
and optimized model lib for a given platform. It enables users to bring their own new model weights, try different quantization modes,
and customize the overall model optimization flow.


.. note::
    Before you proceed, please make sure that you have :ref:`install-tvm-unity` correctly installed on your machine.
    TVM-Unity is the necessary foundation for us to compile models with MLC LLM.
    If you want to build webgpu, please also complete :ref:`install-web-build`.
    Please also follow the instruction in :ref:`deploy-cli` to obtain the CLI app that can be used to chat with the compiled model.
    Finally, we strongly recommend you read :ref:`project-overview` first to get familiarized with the high-level terminologies.

.. contents:: Table of Contents
    :depth: 1
    :local:


Get Started
-----------

This section provides a step by step instructions to guide you through the compilation process of one specific model.
We take the RedPajama-v1-3B as an example.
You can select the platform where you want to **run** your model from the tabs below and run the corresponding command.
We strongly recommend you **start with Metal/CUDA/Vulkan** as it is easier to validate the compilation result on
your personal computer.

.. tabs::

    .. group-tab:: Metal

        On Apple Silicon powered Mac, compile for Apple Silicon Mac:

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target metal --quantization q4f16_0

        On Apple Silicon powered Mac, compile for x86 Mac:

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target metal_x86_64 --quantization q4f16_0

    .. group-tab:: Linux - CUDA

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target cuda --quantization q4f16_0

    .. group-tab:: Vulkan

        On Linux, compile for Linux:

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target vulkan --quantization q4f16_0

        On Linux, compile for Windows: please first install the `LLVM-MinGW <https://github.com/mstorsjo/llvm-mingw>`_ toolchain, and substitute the ``path/to/llvm-mingw`` in the command with your LLVM-MinGW installation path.

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target vulkan --quantization q4f16_0 --llvm-mingw path/to/llvm-mingw

    .. group-tab:: iOS/iPadOS

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target iphone --max-seq-len 768 --quantization q4f16_0

        .. note::
            If it runs into error

            .. code:: text

                Compilation error:
                xcrun: error: unable to find utility "metal", not a developer tool or in PATH
                xcrun: error: unable to find utility "metallib", not a developer tool or in PATH

            , please check and make sure you have Command Line Tools for Xcode installed correctly.
            You can use ``xcrun metal`` to validate: when it prints ``metal: error: no input files``, it means the Command Line Tools for Xcode is installed and can be found, and you can proceed the model compiling.

    .. group-tab:: Android

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target android --max-seq-len 768 --quantization q4f16_0

    .. group-tab:: WebGPU

        .. code:: shell

            python3 build.py --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target webgpu --quantization q4f16_0

By executing the compile command above, we generate the model weights, model lib, and a chat config.
We can check the output with the commands below:

.. tabs::

    .. group-tab:: Metal

        .. code:: shell

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0
              RedPajama-INCITE-Chat-3B-v1-q4f16_0-metal.so     # ===> the model library
              mod_cache_before_build_metal.pkl                 # ===> a cached file for future builds
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We now chat with the model using the command line interface (CLI) app.

        .. code:: shell

            # Run CLI
            mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_0

       The CLI will use the config file ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params/mlc-chat-config.json``
       and model library ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/RedPajama-INCITE-Chat-3B-v1-q4f16_0-metal.so``.

    .. group-tab:: Linux - CUDA

        .. code:: shell

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0
              RedPajama-INCITE-Chat-3B-v1-q4f16_0-cuda.so      # ===> the model library
              mod_cache_before_build_cuda.pkl                  # ===> a cached file for future builds
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We now chat with the model using the command line interface (CLI) app.
        Follow the build from source instruction

        .. code:: shell

            # Run CLI
            mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_0

        The CLI app using config file ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params/mlc-chat-config.json``
        and model library ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/RedPajama-INCITE-Chat-3B-v1-q4f16_0-cuda.so``.

    .. group-tab:: Vulkan

        .. code:: shell

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0
              RedPajama-INCITE-Chat-3B-v1-q4f16_0-vulkan.so    # ===> the model library (will be .dll when built for Windows)
              mod_cache_before_build_vulkan.pkl                # ===> a cached file for future builds
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We can further quickly run and validate the model compilation using the command line interface (CLI) app.

        .. code:: shell

            # Run CLI
            mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_0

        CLI app will use config file ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params/mlc-chat-config.json``
        and model library ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/RedPajama-INCITE-Chat-3B-v1-q4f16_0-vulkan.so`` (or ``.dll``).

    .. group-tab:: iOS/iPadOS

        .. code:: shell

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0
              RedPajama-INCITE-Chat-3B-v1-q4f16_0-iphone.tar   # ===> the model library
              mod_cache_before_build_iphone.pkl                # ===> a cached file for future builds
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        The model lib ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/RedPajama-INCITE-Chat-3B-v1-q4f16_0-iphone.tar``
        will be packaged as a static library into the iOS app. Checkout :ref:`deploy-ios` for more details.

    .. group-tab:: Android

        .. code:: shell

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0
              RedPajama-INCITE-Chat-3B-v1-q4f16_0-android.tar  # ===> the model library
              mod_cache_before_build_android.pkl               # ===> a cached file for future builds
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        The model lib ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/RedPajama-INCITE-Chat-3B-v1-q4f16_0-android.tar``
        will be packaged as a static library into the android app. Checkout :ref:`deploy-android` for more details.

    .. group-tab:: WebGPU

        .. code:: shell

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0
              RedPajama-INCITE-Chat-3B-v1-q4f16_0-webgpu.wasm  # ===> the model library
              mod_cache_before_build_webgpu.pkl                # ===> a cached file for future builds
              params                                           # ===> containing the model weights, tokenizer and chat config

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/params
              mlc-chat-config.json                             # ===> the chat config
              ndarray-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        The model lib ``dist/RedPajama-INCITE-Chat-3B-v1-q4f16_0/RedPajama-INCITE-Chat-3B-v1-q4f16_0-webgpu.wasm``
        can be uploaded to internet. You can pass a ``model_lib_map`` field to WebLLM app config to use this library.


Each compilation target produces a specific model library for the given platform. The model weight is shared across
different targets. If you are interested in distributing the model besides local execution, please checkout :ref:`distribute-compiled-models`.
You are also more than welcome to read the following sections for more details about the compilation.

Compile Command Specification
-----------------------------

This section describes the list of options that can be used during compilation.
Generally, the model compile command is specified by a sequence of arguments and in the following pattern:

.. code:: shell

    python3 build.py \
        --model MODEL_NAME_OR_PATH \
        [--hf-path HUGGINGFACE_NAME] \
        --target TARGET_NAME \
        --quantization QUANTIZATION_MODE \
        [--max-seq-len MAX_ALLOWED_SEQUENCE_LENGTH] \
        [--reuse-lib LIB_NAME] \
        [--use-cache=0] \
        [--debug-dump]

This command first goes with ``--model`` or ``--hf-path``.
**Only one of them needs to be specified**: when the model is publicly available on Hugging Face, you can use ``--hf-path`` to specify the model.
In other cases you need to specify the model via ``--model``.

--model MODEL_NAME_OR_PATH  The name or local path of the model to compile.
                            We will search for the model on your disk in the following two candidates:

                            - ``dist/models/MODEL_NAME_OR_PATH`` (e.g., ``--model vicuna-v1-7b``),
                            - ``MODEL_NAME_OR_PATH`` (e.g., ``--model /my-model/vicuna-v1-7b``).

                            When running the compile command using ``--model``, please make sure you have placed the model to compile under ``dist/models/`` or other location on the disk.

--hf-path HUGGINGFACE_NAME  The name of the model's Hugging Face repository.
                            We will download the model to ``dist/models/HUGGINGFACE_NAME`` and load the model from this directory.

                            For example, by specifying ``--hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1``, it will download the model from ``https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1`` to ``dist/models/``.

Another two necessary arguments for the compile command are the target and the quantization mode:

--target TARGET_NAME                The target platform to compile the model for.
                                    The default target is ``auto``, using which we will detect from ``cuda``, ``metal``, ``vulkan`` and ``opencl``.
                                    Besides ``auto``, other available options are: ``metal`` (for M1/M2), ``metal_x86_64`` (for Intel CPU), ``iphone``,
                                    ``vulkan``, ``cuda``, ``webgpu``, ``android``, and ``opencl``.
--quantization QUANTIZATION_MODE    The quantization mode we use to compile.
                                    The format of the code is ``qAfB(_0)``, where ``A`` represents the number of bits for storing weights and ``B`` represents the number of bits for storing activations.
                                    Available options are: ``q3f16_0``, ``q4f16_0``, ``q4f32_0``, ``q0f32``, ``q0f16``, and ``q8f16_0``.
                                    We encourage you to use 4-bit quantization, as the text generated by 3-bit quantized models may have bad quality depending on the model.

The following arguments are optional:

--max-seq-len MAX_ALLOWED_SEQUENCE_LENGTH   The maximum allowed sequence length for the model.
                                            When it is not specified,
                                            we will use the maximum sequence length from the ``config.json`` in the model directory.
--reuse-lib LIB_NAME                        Specifies the previously generated library to reuse.
                                            This is useful when building the same model architecture with different weights.
                                            You can refer to the :ref:`model distribution <distribute-model-step3-specify-model-lib>` page for detail of this argument.
--use-cache                                 When ``--use-cache=0`` is specified,
                                            the model compilation will not use cached file from previous builds,
                                            and will compile the model from the very start.
                                            Using cache can help reduce the time needed to compile.
--debug-dump                                Specifies whether to dump debugging files during compilation.


More Model Compile Commands
---------------------------

This section lists compile commands for more models that you can try out.

.. tabs::

    .. tab:: Model: vicuna-v1-7b

        Please check this page on :doc:`how to get the Vicuna model weights </compilation/get-vicuna-weight>`.

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target cuda --quantization q3f16_0

            .. tab:: Metal

                On Apple Silicon powered Mac, compile for Apple Silicon Mac:

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target metal --quantization q3f16_0

                On Apple Silicon powered Mac, compile for x86 Mac:

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target metal_x86_64 --quantization q3f16_0

            .. tab:: Vulkan

                On Linux, compile for Linux:

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target vulkan --quantization q3f16_0

                On Linux, compile for Windows: please first install the `LLVM-MinGW <https://github.com/mstorsjo/llvm-mingw>`_ toolchain, and substitute the ``path/to/llvm-mingw`` in the command with your LLVM-MinGW installation path.

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target vulkan --quantization q3f16_0 --llvm-mingw path/to/llvm-mingw

            .. tab:: WebGPU

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target llvm --quantization q4f32_0

            .. tab:: iPhone/iPad

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target iphone --max-seq-len 768 --quantization q3f16_0

            .. tab:: Android

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target android --max-seq-len 768 --quantization q4f16_0

    .. tab:: RedPajama-v1-3B

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target cuda --quantization q4f16_0

            .. tab:: Metal

                On Apple Silicon powered Mac, compile for Apple Silicon Mac:

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target metal --quantization q4f16_0

                On Apple Silicon powered Mac, compile for x86 Mac:

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target metal_x86_64 --quantization q4f16_0

            .. tab:: Vulkan

                On Linux, compile for Linux:

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target vulkan --quantization q4f16_0

                On Linux, compile for Windows: please first install the `LLVM-MinGW <https://github.com/mstorsjo/llvm-mingw>`_ toolchain, and substitute the ``path/to/llvm-mingw`` in the command with your LLVM-MinGW installation path.

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target vulkan --quantization q4f16_0 --llvm-mingw path/to/llvm-mingw

            .. tab:: WebGPU

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target llvm --quantization q4f16_0

            .. tab:: iPhone/iPad

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target iphone --max-seq-len 768 --quantization q4f16_0

            .. tab:: Android

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target android --max-seq-len 768 --quantization q4f16_0

    .. tab:: rwkv-raven-1b5/3b/7b

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target cuda --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target cuda --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target cuda --quantization q8f16_0

            .. tab:: Metal

                On Apple Silicon powered Mac, compile for Apple Silicon Mac:

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target metal --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target metal --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target metal --quantization q8f16_0

                On Apple Silicon powered Mac, compile for x86 Mac:

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target metal_x86_64 --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target metal_x86_64 --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target metal_x86_64 --quantization q8f16_0

            .. tab:: Vulkan

                On Linux, compile for Linux:

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target vulkan --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target vulkan --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target vulkan --quantization q8f16_0

                On Linux, compile for Windows: please first install the `LLVM-MinGW <https://github.com/mstorsjo/llvm-mingw>`_ toolchain, and substitute the ``path/to/llvm-mingw`` in the command with your LLVM-MinGW installation path.

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target vulkan --quantization q8f16_0 --llvm-mingw path/to/llvm-mingw
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target vulkan --quantization q8f16_0 --llvm-mingw path/to/llvm-mingw
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target vulkan --quantization q8f16_0 --llvm-mingw path/to/llvm-mingw

            .. tab:: iPhone/iPad

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target iphone --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target iphone --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target iphone --quantization q8f16_0

    .. tab:: Other models

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target cuda --quantization q4f16_0

            .. tab:: Metal

                On Apple Silicon powered Mac, compile for Apple Silicon Mac:

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target metal --quantization q4f16_0

                On Apple Silicon powered Mac, compile for x86 Mac:

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target metal_x86_64 --quantization q4f16_0

            .. tab:: Vulkan

                On Linux, compile for Linux:

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target vulkan --quantization q4f16_0

                On Linux, compile for Windows: please first install the `LLVM-MinGW <https://github.com/mstorsjo/llvm-mingw>`_ toolchain, and substitute the ``path/to/llvm-mingw`` in the command with your LLVM-MinGW installation path.

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target vulkan --quantization q4f16_0 --llvm-mingw path/to/llvm-mingw

            .. tab:: WebGPU

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target llvm --quantization q4f32_0

            .. tab:: iPhone/iPad

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target iphone --max-seq-len 768 --quantization q4f16_0

            .. tab:: Android

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target android --max-seq-len 768 --quantization q4f16_0


For each model and each backend, the above only provides the most recommended build command (which is the most optimized). You can also try with different argument values (e.g., different quantization modes), whose build results may not run as fast and robustly as the provided one when running the model.

.. note::
    Uing 3-bit quantization usually can be overly aggressive and only works for limited settings.
    If you encounter issues where the compiled model does not perform as expected,
    consider utilizing a higher number of bits for quantization (e.g., 4-bit quantization).

If you are interested in distributing the model besides local execution, please checkout :ref:`distribute-compiled-models`.