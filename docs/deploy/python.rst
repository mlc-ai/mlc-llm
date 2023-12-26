.. _deploy-python:

Python API
==========

.. contents:: Table of Contents
   :local:
   :depth: 2

We expose Python API for the MLC-Chat for easy integration into other Python projects.

The Python API is a part of the MLC-Chat package, which we have prepared pre-built pip wheels via
the :doc:`installation page <../install/mlc_llm>`.

Instead of following this page, you could also checkout the following tutorials in
Python notebook (all runnable in Colab):

- `Getting Started with MLC-LLM <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb>`_:
  how to quickly download prebuilt models and chat with it
- `Raw Text Generation with MLC-LLM <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_raw_text_generation.ipynb>`_:
  how to perform raw text generation with MLC-LLM in Python

.. These notebooks are not up-to-date with SLM yet
.. - `Compiling Llama-2 with MLC-LLM <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_compile_llama2_with_mlc_llm.ipynb>`_:
..   how to use Python APIs to compile models with the MLC-LLM workflow
.. - `Extensions to More Model Variants <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_extensions_to_more_model_variants.ipynb>`_:
..   how to use Python APIs to compile and chat with any model variant you'd like


Verify Installation
-------------------

.. code:: bash

   python -c "from mlc_chat import ChatModule; print(ChatModule)"

You are expected to see the information about the :class:`mlc_chat.ChatModule` class.

If the command above results in error, follow :ref:`install-mlc-packages` (either install the prebuilt pip wheels
or :ref:`mlcchat_build_from_source`).

Run MLC Models w/ Python
------------------------

To run a model with MLC LLM in any platform/runtime, you need:

1. **Model weights** converted to MLC format (e.g. `RedPajama-INCITE-Chat-3B-v1-MLC 
   <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-MLC/tree/main>`_.)
2. **Model library** that comprises the inference logic (see repo `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__).

There are two ways to obtain the model weights and libraries:

1. Compile your own model weights and libraries following :doc:`the model compilation page </compilation/compile_models>`.
2. Use off-the-shelf `prebuilt models weights <https://huggingface.co/mlc-ai>`__ and
   `prebuilt model libraries <https://github.com/mlc-ai/binary-mlc-llm-libs>`__ (see :ref:`Model Prebuilts` for details).

We use off-the-shelf prebuilt models in this page. However, same steps apply if you want to run
the models you compiled yourself.

**Step 1: Download prebuilt model weights and libraries**

Skip this step if you have already obtained the model weights and libraries.

.. code:: shell

  # Download pre-conveted weights
  git lfs install && mkdir dist/
  git clone https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC \
                                     dist/Llama-2-7b-chat-hf-q4f16_1-MLC

  # Download pre-compiled model library
  git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs


**Step 2: Run the model in Python**

Use the conda environment you used to install ``mlc_chat``.
From the ``mlc-llm`` directory, you can create a Python
file ``sample_mlc_chat.py`` and paste the following lines:

.. code:: python

   from mlc_chat import ChatModule
   from mlc_chat.callback import StreamToStdout

   # Create a ChatModule instance
   cm = ChatModule(
      model="dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
      model_lib_path="dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
      # Vulkan on Linux: Llama-2-7b-chat-hf-q4f16_1-vulkan.so
      # Metal on macOS: Llama-2-7b-chat-hf-q4f16_1-metal.so
      # Other platforms: Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}
   )

   # You can change to other models that you downloaded
   # Model variants of the same architecture can reuse the same model library
   # Here WizardMath reuses Mistral's model library
   # cm = ChatModule(
   #     model="dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",  # or "dist/WizardMath-7B-V1.1-q4f16_1-MLC"
   #     model_lib_path="dist/prebuilt_libs/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-cuda.so"
   # )

   # Generate a response for a given prompt
   output = cm.generate(
      prompt="What is the meaning of life?",
      progress_callback=StreamToStdout(callback_interval=2),
   )

   # Print prefill and decode performance statistics
   print(f"Statistics: {cm.stats()}\n")

   output = cm.generate(
      prompt="How many points did you list out?",
      progress_callback=StreamToStdout(callback_interval=2),
   )

   # Reset the chat module by
   # cm.reset_chat()


Now run the Python file to start the chat

.. code:: bash

   python sample_mlc_chat.py


.. collapse:: See output

   .. code::

      Using model folder: ./dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
      Using mlc chat config: ./dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1/mlc-chat-config.json
      Using library model: ./dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so

      Thank you for your question! The meaning of life is a complex and subjective topic that has been debated by philosophers, theologians, scientists, and many others for centuries. There is no one definitive answer to this question, as it can vary depending on a person's beliefs, values, experiences, and perspectives.

      However, here are some possible ways to approach the question:

      1. Religious or spiritual beliefs: Many people believe that the meaning of life is to fulfill a divine or spiritual purpose, whether that be to follow a set of moral guidelines, to achieve spiritual enlightenment, or to fulfill a particular destiny.
      2. Personal growth and development: Some people believe that the meaning of life is to learn, grow, and evolve as individuals, to develop one's talents and abilities, and to become the best version of oneself.
      3. Relationships and connections: Others believe that the meaning of life is to form meaningful connections and relationships with others, to love and be loved, and to build a supportive and fulfilling social network.
      4. Contribution and impact: Some people believe that the meaning of life is to make a positive impact on the world, to contribute to society in a meaningful way, and to leave a lasting legacy.
      5. Simple pleasures and enjoyment: Finally, some people believe that the meaning of life is to simply enjoy the present moment, to find pleasure and happiness in the simple things in life, and to appreciate the beauty and wonder of the world around us.

      Ultimately, the meaning of life is a deeply personal and subjective question, and each person must find their own answer based on their own beliefs, values, and experiences.

      Statistics: prefill: 3477.5 tok/s, decode: 153.6 tok/s

      I listed out 5 possible ways to approach the question of the meaning of life.

|

**Running other models**

Checkout the :doc:`/prebuilt_models` page to run other pre-compiled models.

For models other than the prebuilt ones we provided:

1. If the model is a variant to an existing model library (e.g. ``WizardMathV1.1`` and ``OpenHermes`` are variants of ``Mistral`` as
   shown in the code snippet), follow :ref:`convert-weights-via-MLC` to convert the weights and reuse existing model libraries.
2. Otherwise, follow :ref:`compile-model-libraries` to compile both the model library and weights.


Configure MLCChat in Python
---------------------------
If you have checked out :ref:`Configure MLCChat in JSON<configure-mlc-chat-json>`, you would know
that you could configure MLCChat through various fields such as ``temperature``. We provide the
option of overriding any field you'd like in Python, so that you do not need to manually edit
``mlc-chat-config.json``.

Since there are two concepts -- `MLCChat Configuration` and `Conversation Configuration` -- we correspondingly
provide two dataclasses :class:`mlc_chat.ChatConfig` and :class:`mlc_chat.ConvConfig`.

We provide an example below.

.. code:: python

   from mlc_chat import ChatModule, ChatConfig, ConvConfig
   from mlc_chat.callback import StreamToStdout

   # Using a `ConvConfig`, we modify `system`, a field in the conversation template
   # `system` refers to the prompt encoded before starting the chat
   conv_config = ConvConfig(system='Please show as much happiness as you can when talking to me.')

   # We then include the `ConvConfig` instance in `ChatConfig` while overriding `max_gen_len`
   # Note that `conv_config` is an optional subfield of `chat_config`
   chat_config = ChatConfig(max_gen_len=256, conv_config=conv_config)

   # Using the `chat_config` we created, instantiate a `ChatModule`
   cm = ChatModule(
      chat_config=chat_config,
      model="dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
      model_lib_path="dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
      # Vulkan on Linux: Llama-2-7b-chat-hf-q4f16_1-vulkan.so
      # Metal on macOS: Llama-2-7b-chat-hf-q4f16_1-metal.so
      # Other platforms: Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}
   )

   output = cm.generate(
      prompt="What is one plus one?",
      progress_callback=StreamToStdout(callback_interval=2),
   )

   # You could also pass in a `ConvConfig` instance to `reset_chat()`
   conv_config = ConvConfig(system='Please show as much sadness as you can when talking to me.')
   chat_config = ChatConfig(max_gen_len=128, conv_config=conv_config)
   cm.reset_chat(chat_config)

   output = cm.generate(
      prompt="What is one plus one?",
      progress_callback=StreamToStdout(callback_interval=2),
   )


.. collapse:: See output

   .. code::

      Using model folder: ./dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
      Using mlc chat config: ./dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1/mlc-chat-config.json
      Using library model: ./dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so

      Oh, wow, *excitedly* one plus one? *grinning* Well, let me see... *counting on fingers* One plus one is... *eureka* Two!
      ...

      *Sobs* Oh, the tragedy of it all... *sobs* One plus one... *chokes back tears* It's... *gulps* it's... *breaks down in tears* TWO!
      ...

|

.. note:: 
   You do not need to specify the entire ``ChatConfig`` or ``ConvConfig``. Instead, we will first
   load all the fields defined in ``mlc-chat-config.json``, a file required when instantiating
   a :class:`mlc_chat.ChatModule`. Then, we will load in the optional ``ChatConfig`` you provide, overriding the
   fields specified.
   
   It is also worth noting that ``ConvConfig`` itself is overriding the original conversation template
   specified by the field ``conv_template`` in the chat configuration. Learn more about it in
   :ref:`Configure MLCChat in JSON<configure-mlc-chat-json>`.

Raw Text Generation in Python
-----------------------------

Raw text generation allows the user to have more flexibility over his prompts, 
without being forced to create a new conversational template, making prompt customization easier.
This serves other demands for APIs to handle LLM generation without the usual system prompts and other items.

We provide an example below.

.. code:: python

   from mlc_chat import ChatModule, ChatConfig, ConvConfig
   from mlc_chat.callback import StreamToStdout

   # Use a `ConvConfig` to define the generation settings
   # Since the "LM" template only supports raw text generation,
   # System prompts will not be executed even if provided
   conv_config = ConvConfig(stop_tokens=[2,], add_bos=True, stop_str="[INST]")

   # Note that `conv_config` is an optional subfield of `chat_config`
   # The "LM" template serves the basic purposes of raw text generation
   chat_config = ChatConfig(conv_config=conv_config, conv_template="LM")

   # Using the `chat_config` we created, instantiate a `ChatModule`
   cm = ChatModule(
      chat_config=chat_config,
      model="dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
      model_lib_path="dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
      # Vulkan on Linux: Llama-2-7b-chat-hf-q4f16_1-vulkan.so
      # Metal on macOS: Llama-2-7b-chat-hf-q4f16_1-metal.so
      # Other platforms: Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}
   )
   # To make the model follow conversations a chat structure should be provided
   # This allows users to build their own prompts without building a new template
   system_prompt = "<<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n"
   inst_prompt = "What is mother nature?"

   # Concatenate system and instruction prompts, and add instruction tags
   output = cm.generate(
      prompt=f"[INST] {system_prompt+inst_prompt} [/INST]",
      progress_callback=StreamToStdout(callback_interval=2),
   )

   # The LM template has no memory, so it will be reset every single generation
   # In this case the model will just follow normal text completion
   # because there isn't a chat structure
   output = cm.generate(
      prompt="Life is a quality that distinguishes",
      progress_callback=StreamToStdout(callback_interval=2),
   )

.. note:: 
   The ``LM`` is a template without memory, which means that every execution will be cleared.
   Additionally, system prompts will not be run when instantiating a `mlc_chat.ChatModule`,
   unless explicitly given inside the prompt.

Stream Iterator in Python
-------------------------

Stream Iterator gives users an option to stream generated text to the function that the API is called from,
instead of streaming to stdout, which could be a necessity when building services on top of MLC Chat.

We provide an example below.

.. code:: python

   from mlc_chat import ChatModule
   from mlc_chat.callback import StreamIterator

   # Create a ChatModule instance
   cm = ChatModule(
      model="dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
      model_lib_path="dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
      # Vulkan on Linux: Llama-2-7b-chat-hf-q4f16_1-vulkan.so
      # Metal on macOS: Llama-2-7b-chat-hf-q4f16_1-metal.so
      # Other platforms: Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}
   )

   # Stream to an Iterator
   from threading import Thread

   stream = StreamIterator(callback_interval=2)
   generation_thread = Thread(
      target=cm.generate,
      kwargs={"prompt": "What is the meaning of life?", "progress_callback": stream},
   )
   generation_thread.start()

   output = ""
   for delta_message in stream:
      output += delta_message

   generation_thread.join()


API Reference
-------------

User can initiate a chat module by creating :class:`mlc_chat.ChatModule` class, which is a wrapper of the MLC-Chat model.
The :class:`mlc_chat.ChatModule` class provides the following methods:

.. currentmodule:: mlc_chat

.. autoclass:: ChatModule
   :members:
   :exclude-members: evaluate
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

.. autoclass:: ChatConfig
   :members:

.. autoclass:: ConvConfig
   :members:

.. autoclass:: GenerationConfig
   :members:
