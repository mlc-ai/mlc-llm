.. _configure-mlc-chat-json:

Configure MLCChat in JSON
=========================

``mlc-chat-config.json`` is required for both compile-time and runtime, hence serving two purposes:

1. Specify how we compile a model (shown in :ref:`compile-model-libraries`), and
2. Specify conversation behavior in runtime.

**This page focuses on the second purpose.** We explain the components of a chat
configuration and how to customize them by modifying the file. Additionally,
the runtimes also provide APIs to optionally override some of the configurations.

In runtime, this file is stored under the directory of each compiled model
(e.g. `RedPajama chat config <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1/blob/main/mlc-chat-config.json>`__).


.. _struct-mlc-chat-conv:

Structure of MLCChat Configuration
----------------------------------

Below is the ``mlc-chat-config.json`` file corresponding to Llama2 model:

.. code:: json

  // mlc-chat-config.json
  {
    // 1. Metadata used to specify how to compile a model
    "model_type": "llama",
    "quantization": "q4f16_1",
    "version": "0.1.0",
    "model_config": {
      "hidden_size": 4096,
      "intermediate_size": 11008,
      // more fields here...
    },
    "vocab_size": 32000,
    "context_window_size": 4096,
    "sliding_window_size": -1,
    "prefill_chunk_size": 4096,
    "tensor_parallel_shards": 1,

    // 2. Tokenizer-related fields
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tokenizer_files": [
      "tokenizer.model",
      "tokenizer.json",
      "tokenizer_config.json"
    ]

    // 3. Chat related fields that affect runtime behavior
    "mean_gen_len": 128,
    "max_gen_len": 512,
    "shift_fill_factor": 0.3,
    "temperature": 0.6,
    "repetition_penalty": 1.0,
    "top_p": 0.9,
    "conv_template": "llama-2",
  }

.. note:: 
  Fields in the first part of ``mlc-chat-config.json`` (e.g. ``context-window-size``)
  is only for compile-time. Changing them during runtime may lead to unexpected behavior.

**As shown above, the file is divided into three parts. We focus on the third part, which
can be customized to change the behavior of the model.**

``conv_template``
  The name of the conversation template that this chat uses. For more information, please refer to :ref:`conversation structure <struct-conv>`.

``temperature``
  The temperature applied to logits before sampling. The default value is ``0.7``. A higher temperature encourages more diverse outputs, while a lower temperature produces more deterministic outputs.

``repetition_penalty``
  The repetition penalty controls the likelihood of the model generating repeated texts. The default value is set to ``1.0``, indicating that no repetition penalty is applied. Increasing the value reduces the likelihood of repeat text generation. However, setting a high ``repetition_penalty`` may result in the model generating meaningless texts. The ideal choice of repetition penalty may vary among models.

  For more details on how repetition penalty controls text generation, please check out the `CTRL paper <https://arxiv.org/pdf/1909.05858.pdf>`_.

``top_p``
  This parameter determines the set of tokens from which we sample during decoding. The default value is set to ``0.95``. At each step, we select tokens from the minimal set that has a cumulative probability exceeding the ``top_p`` parameter.

  For additional information on top-p sampling, please refer to this `blog post <https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling>`_.

``mean_gen_len``
  The approximated average number of generated tokens in each round. Used to determine whether the maximum window size would be exceeded.

``max_gen_len``
  This parameter determines the maximum length of the generated text. If it is not set, the model will generate text until it encounters a stop token.

``shift_fill_factor``
  The fraction of maximum window size to shift when it is exceeded.

.. _struct-conv:

Conversation Structure
^^^^^^^^^^^^^^^^^^^^^^

There are three options of loading conversation configurations:

1. Load from pre-defined conversation templates.
2. Load from JSON format conversation configuration.
3. First load from pre-defined conversation templates, then override some fields with JSON format conversation configuration.

.. _load-predefined-conv-template:

Load from Pre-defined Conversation Templates
--------------------------------------------

MLC-LLM provided a set of pre-defined conversation templates, which you can directly use by specifying the template name in ``conv_template`` field in the ``mlc-chat-config.json``, below is a list (not complete) of supported conversation templates:

- ``llama-2``
- ``vicuna_v1.1``
- ``redpajama_chat``
- ``rwkv``
- ``dolly``
- ...

Please refer to `conv_template.cc <https://github.com/mlc-ai/mlc-llm/blob/main/cpp/conv_templates.cc>`_ for the full list of supported templates and their implementations.

.. _load-json-conv-config:

Load from JSON Conversation Configuration
-----------------------------------------

Below is a generic structure of a JSON conversation configuration (we use vicuna as an example):

.. code:: json

  // mlc-chat-config.json
  {
    // ...
    "conv_config": {
      "seps": [
        " ",
        "<\/s>"
      ],
      "stop_tokens": [
        2
      ],
      "offset": 0,
      "separator_style": 0,
      "messages": [],
      "stop_str": "<\/s>",
      "roles": [
        "USER",
        "ASSISTANT"
      ],
      "role_msg_sep": ": ",
      "role_empty_sep": ": ",
      "system": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
      "add_bos": true,
      "name": "vicuna_v1.1"
    }
  }

``roles``
  An array that describes the role names of the user and the model. These names are specific to the model being used.
``system``
  The prompt encoded before starting the chat. It can be customized to a user-defined prompt.
``add_bos``
  Determines whether a beginning-of-string (bos) token should be added before the input tokens.
``stop_str``
  When the ``stop_str`` is encountered, the model will stop generating output.
``stop_tokens``
  A list of token IDs that act as stop tokens.
``seps``
  An array of strings indicating the separators to be used after a user message and a model message respectively.
``messages``
  The chat history represented as an array of string pairs in the following format: ``[[role_0, msg_0], [role_1, msg_1], ...]``
``offset``
  The offset used to begin the chat from the chat history. When ``offset`` is not ``0``, ``messages[0:offset-1]`` will be encoded.
``separator_style``
  Specifies whether we are in chat-bot mode (``0``) or pure LM prompt mode (``1``).
``role_msg_sep``
  A string indicating the separator between a role and a message.
``role_empty_sep``
  A string indicating the separator to append to a role when there is no message yet.


When the value of ``separator_style`` is set to 0 (or ``kSepRoleMsg``), each round of conversation follows the format:

.. code:: text

  {role[0]}{separator_style}{user_input}{sep[0]}
  {role[1]}{separator_style}{model_output}{sep[1]}

Here, ``{user_input}`` represents the input provided by the user, and ``{model_output}`` represents the output generated by the model.

On the other hand, if the value of ``separator_style`` is set to 1 (or ``kLM``), the model is not aware of the chat history and generates the response immediately after the user input prompt:


.. code:: text

  {user_prompt}{model_output}


.. _customize-conv-template:

Customize Conversation Template
-------------------------------

In the ``mlc-chat-config.json`` file, you have the option to specify both ``conv_template`` and ``conv_config``. MLC-LLM will first load the predefined template with the name specified in ``conv_template`` and then override some of the configurations specified in ``conv_config``. It's important to note that the configurations in ``conv_config`` don't need to be complete, allowing for partial updates.

.. _example_replace_system_prompt:

Example 1: Replace System Prompt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're tired of the default system prompt, here's an example of how you can replace it:

.. code:: json

  // mlc-chat-config.json
  {
    // ...
    "conv_template": "vicuna_v1.1",
    "conv_config": {
      "system": "You are not Vicuna, your name is Guanaco, now let's chat!"
    }
  }


The next time you run ``mlc_chat_cli``, you will start a chat with Vicuna using a new system prompt.

.. _example_resume_chat_history:

Example 2: Resume from Chat History
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example demonstrates how to chat with Vicuna and resume from a chat history:

.. code:: json

  // mlc-chat-config.json
  {
    // ...
    "conv_template": "vicuna_v1.1",
    "conv_config": {
      "messages": [
        ["USER", "Suppose we already have projects llama, alpaca and vicuna, what do you think would be a great name for the next project?"],
        ["ASSISTANT", "Based on the previous projects, a possible name for the next project could be \"cervidae\" which is the scientific name for deer family. This name reflects the collaboration and teamwork involved in the development of the project, and also nods to the previous projects that have been developed by the team."],
        ["USER", "I like cervidae, but the name is too long!"],
        ["ASSISTANT", "In that case, a shorter and catchier name for the next project could be \"DeerRun\" which plays on the idea of the project being fast and efficient, just like a deer running through the woods. This name is memorable and easy to pronounce, making it a good choice for a project name."]
      ],
      "offset": 4
    }
  }


The next time you start ``mlc_chat_cli``, or use Python API, you will initiate a chat with Vicuna and resume from the provided chat history.
