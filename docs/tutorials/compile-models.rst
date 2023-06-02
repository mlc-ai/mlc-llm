.. _How to Compile Models:

How to Compile Models
=====================

In this tutorial, we will guide you on how to build an LLM with architectures that are already supported by MLC LLM for different backends. Before proceeding with this tutorial, make sure you have completed the :doc:`/install/index` tutorial. In the following content, we assume that you have already installed the necessary components and will not cover the installation steps.

We have provided a list of off-the-shelf prebuilt models (refer to the :ref:`off-the-shelf-models` section) that you can directly use without the need for building. To learn how to deploy these models, refer to the tutorial :doc:deploy-models.

If your model is not included in the list of off-the-shelf models, but its architecture falls within the supported model architectures (see the :ref:`supported-model-architectures` section), you can follow this tutorial to build your model for different backends.

In the event that your model architecture is not supported, you can refer to the tutorial :doc:`bring-your-own-models` to learn how to introduce new model architectures.

This tutorial contains the following sections in order:

.. contents:: Table of Contents
    :depth: 1
    :local:

.. _compile-model-dependencies:

Dependencies
------------

`TVM-Unity <https://discuss.tvm.apache.org/t/establish-tvm-unity-connection-a-technical-strategy/13344>`__ is required to compile models, please follow the instructions in :ref:`tvm-unity-install` to install the
TVM-Unity package before proceeding with this tutorial.

.. _compile-model-prepare-model-weight:

Prepare Model Weight
--------------------

This section briefly introduces how to prepare the weight of the model we want to build.

.. _compile-models-with-full-weight:

Models with Full Weight
~~~~~~~~~~~~~~~~~~~~~~~

For models whose full weight is directly available (e.g., `RedPajama-v1-3B on Hugging Face <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`_), we need to put the model to path ``dist/models/MODEL_NAME`` under the path of your MLC LLM.

- If the model weight is hosted on `Hugging Face <https://huggingface.co>`_, we can download the model via git clone. For example, the commands below download the RedPajama-v1-3B weight to ``dist/models/RedPajama-INCITE-Chat-3B-v1``.

    .. code:: shell

        git lfs install
        git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1 dist/models/RedPajama-INCITE-Chat-3B-v1

    .. note::
        Depending on the model size and the Hugging Face repository, the git clone operation might take a while (for example, cloning the RedPajama-v1-3B model takes about 10 min on one of our dev workstation).

- If you have your own fine-tuned model, directly copy your model to ``dist/models/MODEL_NAME``.

    .. note::
        An alternative approach (which saves disk space) is to use symbolic link to link your model to ``dist/models/MODEL_NAME``:

        .. code:: shell

            ln -s path/to/your/model dist/models/MODEL_NAME

.. _compile-models-with-base-weight-and-delta-weight:

Models with Base Weight and Delta Weight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For models whose base weight and delta weight are available (e.g., Vicuna-v1-7B only releases the `delta weight <https://huggingface.co/lmsys/vicuna-7b-delta-v1.1>`_ on Hugging Face), we need to apply the delta weights to the base weight. You can refer to the instructions of `getting Vicuna model weight <https://github.com/lm-sys/FastChat#vicuna-weights>`_ as a reference. After getting the full weight of the model, copy or symbolic link the model to ``dist/models/MODEL_NAME``.

.. _compile-models-run-build-script:

Run Build Script
----------------

We can now run the build script ``build.py`` under the path of MLC LLM. The command is usually in the following pattern:

.. code:: shell

    python3 build.py --model MODEL_NAME_OR_PATH --target TARGET_NAME --quantization QUANTIZATION_NAME [--max-seq-len MAX_ALLOWED_SEQUENCE_LENGTH] [--debug-dump] [--use-cache=0]

In the command above, ``--model`` specifies the name of the model to build, ``--target`` specifies the backend we build the model to, ``--quantization`` specifies the quantization mode we use for build. You can find the build command according to the combination of different models and targets. For full explanation and usage of each argument, please check out the `API reference <http://127.0.0.1>`_.

.. tabs::

    .. tab:: Model: vicuna-v1-7b

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target cuda --quantization q3f16_0

            .. tab:: Metal

                On Apple Silicon powered Mac, build for Apple Silicon Mac:

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target metal --quantization q3f16_0

                On Apple Silicon powered Mac, build for x86 Mac:

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target metal_x86_64 --quantization q3f16_0

            .. tab:: Vulkan

                On Linux, build for Linux:

                .. code:: shell

                    python3 build.py --model vicuna-v1-7b --target vulkan --quantization q3f16_0

                On Linux, build for Windows:

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

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target cuda --quantization q3f16_0

            .. tab:: Metal

                On Apple Silicon powered Mac, build for Apple Silicon Mac:

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target metal --quantization q3f16_0

                On Apple Silicon powered Mac, build for x86 Mac:

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target metal_x86_64 --quantization q3f16_0

            .. tab:: Vulkan

                On Linux, build for Linux:

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target vulkan --quantization q3f16_0

                On Linux, build for Windows:

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target vulkan --quantization q3f16_0 --llvm-mingw path/to/llvm-mingw

            .. tab:: WebGPU

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target llvm --quantization q4f32_0

            .. tab:: iPhone/iPad

                .. code:: shell

                    python3 build.py --model RedPajama-INCITE-Chat-3B-v1 --target iphone --max-seq-len 768 --quantization q3f16_0

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

                On Apple Silicon powered Mac, build for Apple Silicon Mac:

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target metal --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target metal --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target metal --quantization q8f16_0

                On Apple Silicon powered Mac, build for x86 Mac:

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target metal_x86_64 --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target metal_x86_64 --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target metal_x86_64 --quantization q8f16_0

            .. tab:: Vulkan

                On Linux, build for Linux:

                .. code:: shell

                    # For 1.5B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-1b5 --target vulkan --quantization q8f16_0
                    # For 3B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-3b --target vulkan --quantization q8f16_0
                    # For 7B model
                    python3 build.py --hf-path=RWKV/rwkv-raven-7b --target vulkan --quantization q8f16_0

                On Linux, build for Windows:

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
                    python3 build.py --model MODEL_NAME --target cuda --quantization q3f16_0

            .. tab:: Metal

                On Apple Silicon powered Mac, build for Apple Silicon Mac:

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target metal --quantization q3f16_0

                On Apple Silicon powered Mac, build for x86 Mac:

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target metal_x86_64 --quantization q3f16_0

            .. tab:: Vulkan

                On Linux, build for Linux:

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target vulkan --quantization q3f16_0

                On Linux, build for Windows:

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target vulkan --quantization q3f16_0 --llvm-mingw path/to/llvm-mingw

            .. tab:: WebGPU

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target llvm --quantization q4f32_0

            .. tab:: iPhone/iPad

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target iphone --max-seq-len 768 --quantization q3f16_0

            .. tab:: Android

                .. code:: shell

                    # Download and put the model to `dist/models/MODEL_NAME`, and then run
                    python3 build.py --model MODEL_NAME --target android --max-seq-len 768 --quantization q4f16_0

Here are some notes on the build commands above:

- For each model and each backend, we only provide the most recommended build command (which is the most optimized). You can also try with different argument values (e.g., different quantization modes), whose build results do not run as fast and robustly as the provided one in deployment.
- After a successful build, the build script outputs some cache files for quicker future builds. If you want to ignore the cached files and want to build from the very beginning, please append ``--use-cache=0`` to the end of the build command.
- You can add ``--debug-dump`` to the build command to  optionally specifies if we will write some dump files for debugging.

After running the build script successfully, you can proceed to the next tutorial on `how to deploy models to different backends <http:127.0.0.1>`_.

.. warning::
    In certain cases, using 3-bit quantization for compiling can be overly aggressive and may result in the compiled model generating meaningless text. If you encounter issues where the compiled model does not perform as expected, consider utilizing a higher number of bits for quantization (e.g., 4-bit quantization).

.. _compile-models-why-need-build:

Why Need Build?
---------------

As supplementary, this section explains what the **build** means in MLC LLM. Compared with PyTorch that runs every model in *eager mode*, the overall workflow of MLC LLM separates model execution into two major stages: **build** and **deployment**.
This separation enables us to build LLM to different backends using a single common flow and also supports us to optimize the LLM execution towards better runtime performance (less run time).

- In the build stage, MLC LLM takes the model, the target backend, and other configurable arguments as input, applies optimizations and transformations that accelerate the execution of the model on the target backend, and generates a set of output for the deployment stage. The set of output includes a binary library file for the model specific to the target backend, the quantized model weights, the tokenizer files specific to the model, and a config JSON file that contains some model basic information as well as the configurable parameters for deployment (such as the chat temperature). The output (and only the output) generated by the build stage will be consumed by the deployment stage.
- The deployment stage runs on the target backend (e.g., web browser, mobile phones, etc.). It takes the output of the build stage as input and provides an interface for people to interact with the model we build. The interface can be a command line if the model is deployed to the native desktop/laptop environment or a chat box if the model is deployed to web browser and mobile phones.

.. image:: https://mlc.ai/blog/img/redpajama/customization.svg
   :alt: compilation workflow
   :align: center

.. _compile-models-troubleshooting:

Troubleshooting FAQ
-------------------

(draft)

.. collapse:: Q: I encountered the ``Unable to parse TuningRecord`` error immediately when I run the build script.

    Please update your MLC LLM codebase to the latest by git.

.. collapse:: Q: I encountered error when building the Moss model.

    Moss support is still ongoing and we are now working on it. Please try other models first.


- LLVM error (https://github.com/mlc-ai/mlc-llm/issues/182)
- Windows unresolved external symbols (https://github.com/mlc-ai/mlc-llm/issues/194)
