How to Customize Conversation
=============================

There is a ``mlc-chat-config.json`` file under the director of each compiled model, which user can customize for their own purposes.

In this tutorial we will explain the structure of the configuration file and the meaning of each item.

.. _struct-mlc-chat-conv

Structure of MLC-Chat Configuration
-----------------------------------

Below is the ``mlc-chat-config.json`` file corresponding to Vicuna model:

.. code:: json

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


There are some parameters you can customize to change the model behavior.

``conv_template``
  The conversation template this chat uses. Please check the :ref:`struct-conv-template` section for details.

``temperature``
  The temperature applied to logits before sampling, defaults to ``0.7``. High temperature encourages more diverse outputs while low temperature produce more deterministic outputs.

``repetition_penalty``
  A parameter controling how likely the model will generate repeat texts, defaults to ``1.0`` (which means repetition penalty is not applied). The higher this value is, the less possible it's for the model to generate repeat text. However, high repetition_penalty might cause the model genereate meaningless text. The best choice of repetition penalty is model dependent.

  Please check the `CTRL paper<https://arxiv.org/pdf/1909.05858.pdf>`__ for details on how repetition penalty controls text generation.

``top_p``
  This parameters controls the set of tokens we sample from during decoding, defaults to ``0.95``. For each step, we choose from the minimal set of tokens with a cumulative probility exceeds the ``top_p`` parameter.

  Please check `this blog <https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling>`__ on details of top-p sampling.


.. _struct-conv-template

Structure of Conversation Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

User can customize conversation template by specifying a template name (in case that template is supported in MLC-LLM, see `conv_template.cc <https://github.com/mlc-ai/mlc-llm/blob/main/cpp/conv_templates.cc>`__ for supported templates and their short codes) in ``conv_template`` field in the ``mlc-chat-config.json``, or providing a brand new conversation configuration in ``conv_config`` field in JSON format. Below is a generic structure of a conversation configuration (we use vicuna as an example):

.. code:: json

  {  // mlc-chat-config.json
    ...
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
    ...
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
  An array of string, indicating the separators after user message and model message, correspondingly. Please check :ref:`conv-format` section on how they are used in a conversation.
``messages``
  The chat history in an array of string pairs, in the format of following:
  ``{{role_0, msg_0}, {role_1, msg_1}, ...}``.
``offset``
  The offset indicating the point we start from examples.
``separator_style``
  Whether we are in chat-bot mode (0) or pure LM prompt mode (1), see :ref:`conv-format` section for more detailts.
``role_msg_sep``
  A string indicating the separator between role and message, please check :conv-format: section for more details.
``role_empty_sep``
  A string indicating the separator to append to role when there is no message yet.


.. _conv-format

Conversation Format
-------------------

When ``separator_style`` is 0 (or ``kSepRoleMsg``), one round of conversation has the following format:

.. code::
  {role[0]}{separator_style}{user_input}{sep[0]}
  {role[1]}{separator_style}{model_output}{sep[1]}

where ``{user_input}`` and ``{model_output}`` are the user input and model outputs.

If the ``separator_style`` is 1 (or ``kLM``), the model will not be aware of chat history, and the model will generate immediately after the user input prompt:

.. code::
  {user_prompt}{model_output}

.. _customize-conv-template

Customize Conversation Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


