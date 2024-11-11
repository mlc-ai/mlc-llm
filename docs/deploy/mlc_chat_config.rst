.. _configure-mlc-chat-json:

Customize MLC Chat Config
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

    // 3. Conversation template related fields
    "conv_template": {
      "name": "llama-2",
      "system_template": "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n ",
      "system_message": "You are a helpful, respectful and honest assistant.",
      // more fields here...
    },

    // 4. Chat related fields that affect runtime behavior
    "temperature": 0.6,
    "repetition_penalty": 1.0,
    "top_p": 0.9
  }

.. note::
  Fields in the first part of ``mlc-chat-config.json`` (e.g. ``context-window-size``)
  is only for compile-time. Changing them during runtime may lead to unexpected behavior.

**As shown above, the file is divided into three parts. We focus on the third part, which
can be customized to change the behavior of the model.**

``conv_template``
  .. note::
    Legacy ``mlc-chat-config.json`` may specify a string for this field to look up a registered conversation
    template. It will be deprecated in the future. Re-generate config using the latest version of mlc_llm
    to make sure this field is a complete JSON object.

  The conversation template that this chat uses. For more information, please refer to :ref:`conversation structure <struct-conv>`.

``temperature``
  The temperature applied to logits before sampling. The default value is ``0.7``. A higher temperature encourages more diverse outputs, while a lower temperature produces more deterministic outputs.

``repetition_penalty``
  The repetition penalty controls the likelihood of the model generating repeated texts. The default value is set to ``1.0``, indicating that no repetition penalty is applied. Increasing the value reduces the likelihood of repeat text generation. However, setting a high ``repetition_penalty`` may result in the model generating meaningless texts. The ideal choice of repetition penalty may vary among models.

  For more details on how repetition penalty controls text generation, please check out the `CTRL paper <https://arxiv.org/pdf/1909.05858.pdf>`_.

``top_p``
  This parameter determines the set of tokens from which we sample during decoding. The default value is set to ``0.95``. At each step, we select tokens from the minimal set that has a cumulative probability exceeding the ``top_p`` parameter.

  For additional information on top-p sampling, please refer to this `blog post <https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling>`_.


.. _struct-conv:

Conversation Structure
^^^^^^^^^^^^^^^^^^^^^^

MLC-LLM provided a set of pre-defined conversation templates, which you can directly use by
specifying ``--conv-template [name]`` when generating config. Below is a list (not complete) of
supported conversation templates:

- ``llama-2``
- ``mistral_default``
- ``chatml``
- ``phi-2``
- ...

Please refer to `conversation_template <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/conversation_template>`_ directory for the full list of supported templates and their implementations.

Below is a generic structure of a JSON conversation configuration (we use vicuna as an example):

.. code:: json

  // mlc-chat-config.json
  {
    // ...
    "conv_template": {
      "name": "llama-2",
      "system_template": "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n ",
      "system_message": "You are a helpful, respectful and honest assistant.",
      "roles": {
        "user": "[INST]",
        "assistant": "[/INST]",
        "tool": "[INST]"
      },
      "role_templates": {
        "user": "{user_message}",
        "assistant": "{assistant_message}",
        "tool": "{tool_message}"
      },
      "messages": [],
      "seps": [
        " "
      ],
      "role_content_sep": " ",
      "role_empty_sep": " ",
      "stop_str": [
        "[INST]"
      ],
      "stop_token_ids": [
        2
      ],
      "function_string": "",
      "use_function_calling": false
    }
  }

``name``
    Name of the conversation.
``system_template``
    The system prompt template, it optionally contains the system
    message placeholder, and the placeholder will be replaced with
    the system message below.
``system_message``
    The content of the system prompt (without the template format).
``system_prefix_token_ids``
    The system token ids to be prepended at the beginning of tokenized
    generated prompt.
``roles``
    The conversation roles
``role_templates``
    The roles prompt template, it optionally contains the defaults
    message placeholders and will be replaced by actual content
``messages``
    The conversation history messages.
    Each message is a pair of strings, denoting "(role, content)".
    The content can be None.
``seps``
    An array of strings indicating the separators to be used after a user
    message and a model message respectively.
``role_content_sep``
    The separator between the role and the content in a message.
``role_empty_sep``
    The separator between the role and empty contents.
``stop_str``
    When the ``stop_str`` is encountered, the model will stop generating output.
``stop_token_ids``
    A list of token IDs that act as stop tokens.
``function_string``
    The function calling string.
``use_function_calling``
    Whether using function calling or not, helps check for output message format in API call.


Given a conversation template, the corresponding prompt generated out
from it is in the following format:

.. code:: text

  <<system>><<messages[0][0]>><<role_content_sep>><<messages[0][1]>><<seps[0]>>
            <<messages[1][0]>><<role_content_sep>><<messages[1][1]>><<seps[1]>>
            ...
            <<messages[2][0]>><<role_content_sep>><<messages[2][1]>><<seps[0]>>
            <<roles[1]>><<role_empty_sep>>
