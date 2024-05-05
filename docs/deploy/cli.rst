.. _deploy-cli:

CLI
===============

MLCChat CLI is the command line tool to run MLC-compiled LLMs out of the box.

.. contents:: Table of Contents
  :local:
  :depth: 2

Option 1. Conda Prebuilt
~~~~~~~~~~~~~~~~~~~~~~~~

The prebuilt package supports Metal on macOS and Vulkan on Linux and Windows, and can be installed via Conda one-liner.

To use other GPU runtimes, e.g. CUDA, please instead :ref:`build it from source <mlcchat_build_from_source>`.

.. code:: shell

    conda activate your-environment
    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
    mlc_llm chat -h

.. note::
    The prebuilt package supports **Metal** on macOS and **Vulkan** on Linux and Windows. It is possible to use other GPU runtimes such as **CUDA** by compiling MLCChat CLI from the source.


Option 2. Build MLC Runtime from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide options to build mlc runtime libraries and ``mlc_llm`` from source.
This step is useful if the prebuilt is unavailable on your platform, or if you would like to build a runtime
that supports other GPU runtime than the prebuilt version. We can build a customized version
of mlc chat runtime. You only need to do this if you choose not to use the prebuilt.

First, make sure you install TVM unity (following the instruction in :ref:`install-tvm-unity`).
Then please follow the instructions in :ref:`mlcchat_build_from_source` to build the necessary libraries.

.. `|` adds a blank line

|

Run Models through MLCChat CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ``mlc_llm`` is installed, you are able to run any MLC-compiled model on the command line.

To run a model with MLC LLM in any platform, you can either:

- Use off-the-shelf model prebuilts from the MLC Huggingface repo (see :ref:`Model Prebuilts` for details).
- Use locally compiled model weights and libraries following :doc:`the model compilation page </compilation/compile_models>`.

**Option 1: Use model prebuilts**

To run ``mlc_llm``, you can specify the Huggingface MLC prebuilt model repo path with the prefix ``HF://``.
For example, to run the MLC Llama 3 8B Q4F16_1 model (`Repo link <https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC>`_),
simply use ``HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC``. The model weights and library will be downloaded
automatically from Huggingface.

.. code:: shell

  mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC --device "cuda:0" --overrides context_window_size=1024

.. code::

  You can use the following special commands:
    /help               print the special commands
    /exit               quit the cli
    /stats              print out the latest stats (token/sec)
    /reset              restart a fresh chat
    /set [overrides]    override settings in the generation config. For example,
                        `/set temperature=0.5;max_gen_len=100;stop=end,stop`
                        Note: Separate stop words in the `stop` option with commas (,).
    Multi-line input: Use escape+enter to start a new line.

  user: What's the meaning of life
  assistant:
  What a profound and intriguing question! While there's no one definitive answer, I'd be happy to help you explore some perspectives on the meaning of life.

  The concept of the meaning of life has been debated and...


**Option 2: Use locally compiled model weights and libraries**

For models other than the prebuilt ones we provided:

1. If the model is a variant to an existing model library (e.g. ``WizardMathV1.1`` and ``OpenHermes`` are variants of ``Mistral``),
   follow :ref:`convert-weights-via-MLC` to convert the weights and reuse existing model libraries.
2. Otherwise, follow :ref:`compile-model-libraries` to compile both the model library and weights.

Once you have the model locally compiled with a model library and model weights, to run ``mlc_llm``, simply

- Specify the path to ``mlc-chat-config.json`` and the converted model weights to ``--model``
- Specify the path to the compiled model library (e.g. a .so file) to ``--model-lib``

.. code:: shell

  mlc_llm chat dist/Llama-2-7b-chat-hf-q4f16_1-MLC \
               --device "cuda:0" --overrides context_window_size=1024 \
               --model-lib dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-vulkan.so
               # CUDA on Linux: dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so
               # Metal on macOS: dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-metal.so
               # Same rule applies for other platforms
