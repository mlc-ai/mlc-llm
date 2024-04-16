.. _project-overview:

Project Overview
================

This page introduces high-level project concepts to help us use and customize MLC LLM.
The MLC-LLM project consists of three distinct submodules: model definition, model compilation, and runtimes.

.. figure:: /_static/img/project-structure.svg
   :width: 600
   :align: center
   :alt: Project Structure

   Three independent submodules in MLC LLM

**➀ Model definition in Python.** MLC offers a variety of pre-defined architectures, such as Llama (e.g., Llama2, Vicuna, OpenLlama, Wizard), GPT-NeoX (e.g., RedPajama, Dolly), RNNs (e.g., RWKV), and GPT-J (e.g., MOSS). Model developers could solely define the model in pure Python, without having to touch code generation and runtime.

**➁ Model compilation in Python.** Models are compiled by :doc:`TVM Unity </install/tvm>` compiler, where the compilation is configured in pure Python. MLC LLM quantizes and exports the Python-based model to a model library and quantized model weights. Quantization and optimization algorithms can be developed in pure Python to compress and accelerate LLMs for specific usecases.

**➂ Platform-native runtimes.** Variants of MLCChat are provided on each platform: **C++** for command line, **Javascript** for web, **Swift** for iOS, and **Java** for Android, configurable with a JSON chat config. App developers only need to familiarize with the platform-naive runtimes to integrate MLC-compiled LLMs into their projects.

.. _terminologies:

Terminologies
-------------

It is helpful for us to familiarize the basic terminologies used in the MLC chat applications. Below are the
three things you need to run a model with MLC.

- **model lib**: The model library refers to the executable libraries that enable
  the execution of a specific model architecture. On Linux and M-chip macOS, these libraries have the suffix
  ``.so``; on intel macOS, the suffix is ``.dylib``; on Windows, the library file ends with ``.dll``;
  on web browser, the library suffix is ``.wasm``. (see `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__).

- **model weights**: The model weight is a folder that contains the quantized neural network weights
  of the language models as well as the tokenizer configurations. (e.g. `Llama-2-7b-chat-hf-q4f16_1-MLC <https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC>`__)

- **chat config**: The chat configuration includes settings that allow customization of parameters such as temperature and system prompt.
  The default chat config usually resides in the same directory as model weights. (e.g. see ``Llama-2-7b-chat-hf-q4f16_1``'s
  `mlc-chat-config.json <https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC/blob/main/mlc-chat-config.json>`__)

Model Preparation
-----------------


There are several ways to prepare the model weights and model lib.

- :ref:`Model Prebuilts` contains models that can be directly used.
- You can also :doc:`run model compilation </compilation/compile_models>` for model weight variants for given supported architectures.
- Finally, you can incorporate a new model architecture/inference logic following :doc:`Define New Models </compilation/define_new_models>`.

A default chat config usually comes with the model weight directory. You can further customize
the system prompt, temperature, and other options by modifying the JSON file.
MLC chat runtimes also provide API to override these options during model reload.
Please refer to :doc:`/get_started/mlc_chat_config` for more details.


Runtime Flow Overview
---------------------

Once the model weights, model library, and chat configuration are prepared, an MLC chat runtime can consume them as an engine to drive a chat application.
The diagram below shows a typical workflow for a MLC chat application.

.. image:: https://raw.githubusercontent.com/mlc-ai/web-data/a05d4598bae6eb5a3133652d5cc0323ced3b0e17/images/mlc-llm/tutorials/mlc-llm-flow-slm.svg
  :width: 90%
  :align: center

On the right side of the figure, you can see pseudo-code illustrating the structure of an MLC chat API during the execution of a chat app.
Typically, there is a ``ChatModule`` that manages the model. We instantiate the chat app with two files: the model weights (which include an ``mlc-chat-config.json``)
and the model library. We also have an optional chat configuration, which allows for overriding settings such as the system prompt and temperature.

All MLC runtimes, including iOS, Web, CLI, and others, use these three elements.
All the runtime can read the same model weight folder. The packaging of the model libraries may vary depending on the runtime.
For the CLI, the model libraries are stored in a DLL directory.
iOS and Android include pre-packaged model libraries within the app due to dynamic loading restrictions.
WebLLM utilizes URLs of local or Internet-hosted WebAssembly (Wasm) files.

What to Do Next
---------------

Thank you for reading and learning the high-level concepts.
Moving next, feel free to check out documents on the left navigation panel and
learn about topics you are interested in.

- :doc:`/get_started/mlc_chat_config` shows how to configure specific chat behavior.
- Build and Deploy App section contains guides to build apps
  and platform-specific MLC chat runtimes.
- Compile models section provides guidelines to convert model weights and produce model libs.
