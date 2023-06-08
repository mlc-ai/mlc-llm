How to Customize Conversation
=============================

This tutorials explains the components of a chat configuration and how to customize them for your own purposes.

There is a ``mlc-chat-config.json`` file under the directory of each compiled model (e.g. 
`RedPajama chat config <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0/blob/main/mlc-chat-config.json>`__ ) which contains the chat configuration. You can customize the chat configuration by modifying this file.

.. _struct-mlc-chat-conv:

Structure of MLC-Chat Configuration
-----------------------------------

Below is the ``mlc-chat-config.json`` file corresponding to Vicuna model:

.. code:: json

  // mlc-chat-config.json
  {
    "model_lib": "vicuna-v1-7b-q4f32_0",
    "local_id": "vicuna-v1-7b-q4f32_0",
    "conv_template": "vicuna_v1.1",
    "temperature": 0.7,
    "repetition_penalty": 1.0,
    "top_p": 0.95,
    "mean_gen_len": 128,
    "shift_fill_factor": 0.3,
    "tokenizer_files": [
        "tokenizer.model"
    ]
  }


The following parameters can be customized to change the behavior of the model:

``conv_template``
  The name of the conversation template that this chat uses. For more information, refer to the section on conversation structure (:ref:`struct-conv`).

``temperature``
  The temperature applied to logits before sampling. The default value is ``0.7``. A higher temperature encourages more diverse outputs, while a lower temperature produces more deterministic outputs.

``repetition_penalty``
  This parameter controls the likelihood of the model generating repeated texts. The default value is set to ``1.0``, indicating that no repetition penalty is applied. Increasing the value reduces the likelihood of repeat text generation. However, setting a high ``repetition_penalty`` may result in the model generating meaningless text. The ideal choice of repetition penalty depends on the specific model.

  For more details on how repetition penalty controls text generation, please consult the `CTRL paper <https://arxiv.org/pdf/1909.05858.pdf>__`.

``top_p``
  This parameter determines the set of tokens from which we sample during decoding. The default value is set to 0.95. At each step, we select tokens from the minimal set that has a cumulative probability exceeding the ``top_p`` parameter.

  For additional information on top-p sampling, please refer to this `blog post <https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling>__`.


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

- ``vicuna_v1.1``
- ``redpajama_chat``
- ``rwkv``
- ``dolly``
- ...

Please refer to `conv_template.cc <https://github.com/mlc-ai/mlc-llm/blob/main/cpp/conv_templates.cc>`__ for the full list of supported templates and their implementations.

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
  An array describes the role names of the user and the model, this names are model-dependent.
``system``
  The prompt before we start chat, can be customized to user-defined prompt.
``add_bos``
  Whether to add a bos token before the input tokens.
``stop_str``
  When the ``stop_str`` is encountered, the model will stop generation.
``stop_tokens``
  The list of stop token ids.
``seps``
  An array of string, indicating the separators after user message and model message, correspondingly.
``messages``
  The chat history in an array of string pairs, in the format of following:
  ``[[role_0, msg_0], [role_1, msg_1], ...}`` where ``role_i`` and ``msg_i`` are role strings and message strings.
``offset``
  The offset indicating the point we start from examples.
``separator_style``
  Whether we are in chat-bot mode (0) or pure LM prompt mode (1).
``role_msg_sep``
  A string indicating the separator between role and message, please check :conv-format: section for more details.
``role_empty_sep``
  A string indicating the separator to append to role when there is no message yet.


When ``separator_style`` is 0 (or ``kSepRoleMsg``), one round of conversation has the following format:

.. code:: text

  {role[0]}{separator_style}{user_input}{sep[0]}
  {role[1]}{separator_style}{model_output}{sep[1]}

where ``{user_input}`` and ``{model_output}`` are the user input and model outputs.

If the ``separator_style`` is 1 (or ``kLM``), the model will not be aware of chat history, and the model will generate immediately after the user input prompt:

.. code:: text

  {user_prompt}{model_output}

.. _customize-conv-template:

Customize Conversation Template
-------------------------------

You can specify both ``conv_template`` and ``conv_config`` in the ``mlc-chat-config.json`` file, and MLC-LLM would first load the pre-defined template with name specified in ``conv_template``, then override the some of the configurations specified in ``conv_config`` (note that the config don't need to be complete and we can perform partial update).

.. _example_replace_system_prompt:

Example 1: Replace System Prompt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might be tired of the default system prompt, here is an example of how to replace it:

.. code:: json

  // mlc-chat-config.json
  {
    // ...
    "conv_template": "vicuna_v1.1",
    "conv_config": {
      "system": "You are not Vicuna, your name is Guanaco, now let's chat!"
    }
  }


Then next time you start ``mlc_chat_cli``, you will chat with vicuna with new system prompt.

.. _example_add_messages:

Example 2: Start Chat from history
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose you want to chat with model from a chat history:

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

Then next time you start ``mlc_chat_cli``, you will chat with vicuna and start from the provided chat history.
