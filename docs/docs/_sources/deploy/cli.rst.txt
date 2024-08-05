.. _deploy-cli:

CLI
===============

MLC Chat CLI is the command line tool to run MLC-compiled LLMs out of the box interactively.

.. contents:: Table of Contents
  :local:
  :depth: 2

Install MLC-LLM Package
------------------------

Chat CLI is a part of the MLC-LLM package.
To use the chat CLI, first install MLC LLM by following the instructions :ref:`here <install-mlc-packages>`.
Once you have install the MLC-LLM package, you can run the following command to check if the installation was successful:

.. code:: bash

   mlc_llm chat --help

You should see serve help message if the installation was successful.

Quick Start
------------

This section provides a quick start guide to work with MLC-LLM chat CLI.
To launch the CLI session, run the following command:

.. code:: bash

   mlc_llm chat MODEL [--model-lib PATH-TO-MODEL-LIB]

where ``MODEL`` is the model folder after compiling with :ref:`MLC-LLM build process <compile-model-libraries>`. Information about other arguments can be found in the next section.

Once the chat CLI is ready, you can enter the prompt to interact with the model.

.. code::

  You can use the following special commands:
    /help               print the special commands
    /exit               quit the cli
    /stats              print out stats of last request (token/sec)
    /metrics            print out full engine metrics
    /reset              restart a fresh chat
    /set [overrides]    override settings in the generation config. For example,
                        `/set temperature=0.5;top_p=0.8;seed=23;max_tokens=100;stop=str1,str2`
                        Note: Separate stop words in the `stop` option with commas (,).
    Multi-line input: Use escape+enter to start a new line.

  >>> What's the meaning of life?
  The meaning of life is a philosophical and metaphysical question related to the purpose or significance of life or existence in general...

Run CLI with Multi-GPU
----------------------

If you want to enable tensor parallelism to run LLMs on multiple GPUs, please specify argument ``--overrides "tensor_parallel_shards=$NGPU"``. For example,

.. code:: shell

  mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC --overrides "tensor_parallel_shards=2"


The ``mlc_llm chat`` Command
----------------------------

We provide the list of chat CLI interface for reference.

.. code:: bash

   mlc_llm chat MODEL [--model-lib PATH-TO-MODEL-LIB] [--device DEVICE] [--overrides OVERRIDES]


MODEL                  The model folder after compiling with MLC-LLM build process. The parameter
                       can either be the model name with its quantization scheme
                       (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
                       folder. In the former case, we will use the provided name to search
                       for the model folder over possible paths.

--model-lib            A field to specify the full path to the model library file to use (e.g. a ``.so`` file).
--device               The description of the device to run on. User should provide a string in the
                       form of ``device_name:device_id`` or ``device_name``, where ``device_name`` is one of
                       ``cuda``, ``metal``, ``vulkan``, ``rocm``, ``opencl``, ``auto`` (automatically detect the
                       local device), and ``device_id`` is the device id to run on. The default value is ``auto``,
                       with the device id set to 0 for default.
--overrides            Model configuration override. Supports overriding
                       ``context_window_size``, ``prefill_chunk_size``, ``sliding_window_size``, ``attention_sink_size``,
                       and ``tensor_parallel_shards``. The overrides could be explicitly
                       specified via details knobs, e.g. --overrides ``context_window_size=1024;prefill_chunk_size=128``.
