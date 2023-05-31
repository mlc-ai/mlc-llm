.. _How to Customize Chat:

Customize Chat
==============

MLC-LLM allows user to customize parameters and .

.. _chat-config-format:

Chat Configuration Format
^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. _customize-conversation:

Customize Conversation
^^^^^^^^^^^^^^^^^^^^^^