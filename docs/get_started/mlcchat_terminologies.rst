🚧 Terminologies
================

Welcome to mlc-llm! This page provides an overview of the basic concepts of MLC LLM, including how to run a model, share the models you build, and utilize MLC build to create new model libraries that can be run on various environments.

Firstly, let's review the fundamental components required for running an MLC LLM Chat application. When running a chat app, three key elements come into play:

.. _model_weights:

Model Weights
-------------

The models weights are sharded and stored in a list of binary files with name: ``params_shard_ID.bin`` where ``ID`` is th shard index.

.. _model_lib:

Model Library
-------------


The model weights are divided into shards and stored as a list of binary files named ``params_shard_ID.bin``, where ``ID`` represents the shard index.

The model library refers to the executable libraries that enable the execution of a specific model architecture. On Linux, these libraries have the suffix ``.so``, on macOS, the suffix is ``.dylib``, and on Windows, the library file ends with ``.dll``.


.. _chat_config:

Chat Config
-----------
The chat configuration includes settings that allow customization of parameters such as temperature and system prompt. For detailed instructions on how to customize conversations in the chat app, please refer to our documentation: :doc:`/tutorials/customize-conversation`.

Additionally, the chat configuration contains metadata that is essential for the application to locate and execute the model and tokenizers:

``local_id``
  The key uniquely identifies the model within an app.
``model_lib``
  This key specifies which model library to use.
``tokenizer_files``
  This field accepts a list of filenames (under current directory) that specifies how to load tokeniziers,

  .. collapse:: Details

    - For SentencePiece tokenizer, the tokenizer files would be a single ``tokenizer.model`` file.
    - For HuggingFace-style tokenizer, the tokenizer files would be a single ``tokenizer.json`` file.
    - For Byte-Level BPE tokenizer, the tokenizer files would be a ``vocab.json`` file, a ``merges.txt`` and a ``added_tokens.json`` file. 

Runtime Workflow
----------------

.. image:: https://raw.githubusercontent.com/mlc-ai/web-data/de9a5e5b424f36119bd464ddf5a3ddb4c58cc85e/images/mlc-llm/tutorials/mlc-llm-flow.svg

On the right side of the figure, you can see pseudo code illustrating the structure of an MLC chat API during the execution of a chat app. Typically, there is a ChatModule that manages the model. The chat app includes a reload function that takes a ``local_id`` as well as an optional chat configuration override, which allows for overriding settings such as the system prompt and temperature. The MLC Chat runtime utilizes the ``local_id`` and ``model_lib`` to locate the model and initialize its internal state.

All MLC Chat runtimes, including iOS, Web, CLI, and others, make use of these key elements. They are capable of reading the same model weights, although the packaging of the model libraries may vary. For the CLI, the model libraries are stored in a DLL directory. iOS and Android include pre-packaged model libraries within the app itself due to restrictions on dynamic loading. WebLLM, on the other hand, utilizes a ``model_lib_map`` that maps the library name to URLs of WebAssembly (Wasm) files. Thanks to the shared model weights, we can create the weights once and run them across different platforms.

