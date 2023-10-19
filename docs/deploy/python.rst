Python API and Gradio Frontend
==============================

.. contents:: Table of Contents
   :local:
   :depth: 2

We expose Python API for the MLC-Chat for easy integration into other Python projects.
We also provide a web demo based on `gradio <https://gradio.app/>`_ as an example of using Python API to interact with MLC-Chat.

Python API
----------

The Python API is a part of the MLC-Chat package, which we have prepared pre-built pip wheels via the :doc:`installation page <../install/mlc_llm>`.

Verify Installation
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   python -c "from mlc_chat import ChatModule; print(ChatModule)"

You are expected to see the information about the :class:`mlc_chat.ChatModule` class.

If the prebuilt is unavailable on your platform, or you would like to build a runtime
that supports other GPU runtime than the prebuilt version. Please refer our :ref:`Build MLC-Chat Package From Source<mlcchat_package_build_from_source>` tutorial.

Get Started
^^^^^^^^^^^
After confirming that the package ``mlc_chat`` is installed, we can follow the steps
below to chat with a MLC-compiled model in Python.

First, let us make sure that the MLC-compiled ``model`` we want to chat with already exists.

.. note::
   ``model`` has the format ``f"{model_name}-{quantize_mode}"``. For instance, if
   you used ``q4f16_1`` as the ``quantize_mode`` to compile ``Llama-2-7b-chat-hf``, you 
   would have ``model`` being ``Llama-2-7b-chat-hf-q4f16_1``.

If you do not have the MLC-compiled ``model`` ready:

- Checkout :ref:`Try out MLC Chat<get_started>` to download prebuilt models for simplicity, or
- Checkout :ref:`Compile Models via MLC<compile-models-via-MLC>` to compile models with ``mlc_llm`` (another package) yourself

.. tabs ::

   .. tab :: Check prebuilt models

      If you downloaded prebuilt models from MLC LLM, by default:

      - Model lib should be placed at ``./dist/prebuilt/lib/$(model)-$(arch).$(suffix)``.
      - Model weights and chat config are located under ``./dist/prebuilt/mlc-chat-$(model)/``.

      .. collapse:: Example

         .. code:: shell

            >>> ls -l ./dist/prebuilt/lib
            Llama-2-7b-chat-hf-q4f16_1-metal.so  # Format: $(model)-$(arch).$(suffix)
            Llama-2-7b-chat-hf-q4f16_1-vulkan.so
            ...
            >>> ls -l ./dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1  # Format: ./dist/prebuilt/mlc-chat-$(model)/
            # chat config:
            mlc-chat-config.json
            # model weights:
            ndarray-cache.json
            params_shard_*.bin
            ...

   .. tab :: Check compiled models

      If you have compiled models using MLC LLM, by default:

      - Model libraries should be placed at ``./dist/$(model)/$(model)-$(arch).$(suffix)``.
      - Model weights and chat config are located under ``./dist/$(model)/params/``.

      .. collapse:: Example

         .. code:: shell

            >>> ls -l ./dist/Llama-2-7b-chat-hf-q4f16_1/ # Format: ./dist/$(model)/
            Llama-2-7b-chat-hf-q4f16_1-metal.so  # Format: $(model)-$(arch).$(suffix)
            ...
            >>> ls -l ./dist/Llama-2-7b-chat-hf-q4f16_1/params  # Format: ``./dist/$(model)/params/``
            # chat config:
            mlc-chat-config.json
            # model weights:
            ndarray-cache.json
            params_shard_*.bin
            ...

After making sure that the files exist, using the conda environment you used
to install ``mlc_chat``, from the ``mlc-llm`` directory, you can create a Python
file ``sample_mlc_chat.py`` and paste the following lines:

.. code:: python

   from mlc_chat import ChatModule
   from mlc_chat.callback import StreamToStdout

   # From the mlc-llm directory, run
   # $ python sample_mlc_chat.py

   # Create a ChatModule instance
   cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")
   # You can change to other models that you downloaded, for example,
   # cm = ChatModule(model="Llama-2-13b-chat-hf-q4f16_1")  # Llama2 13b model

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

You can also checkout the :doc:`/prebuilt_models` page to run other models.

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

.. note:: 
   You could also specify the address of ``model`` and ``lib_path`` explicitly. If
   you only specify ``model`` as ``model_name`` and ``quantize_mode``, we will
   do a search for you. See more in the documentation of :meth:`mlc_chat.ChatModule.__init__`.

Tutorial with Python Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that you have tried out how to chat with the model in Python, we would
recommend you to checkout the following tutorials in Python notebook (all runnable in Colab):

- `Getting Started with MLC-LLM <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb>`_:
  how to quickly download prebuilt models and chat with it
- `Compiling Llama-2 with MLC-LLM <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_compile_llama2_with_mlc_llm.ipynb>`_:
  how to use Python APIs to compile models with the MLC-LLM workflow
- `Extensions to More Model Variants <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_extensions_to_more_model_variants.ipynb>`_:
  how to use Python APIs to compile and chat with any model variant you'd like
- `Raw Text Generation with MLC-LLM <https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_raw_text_generation.ipynb>`_:
  how to perform raw text generation with MLC-LLM in Python

Configure MLCChat in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
   cm = mlc_chat.ChatModule('Llama-2-7b-chat-hf-q4f16_1', chat_config=chat_config)

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
   specified by the field ``conv_template`` in chat configuration. Learn more about it in
   :ref:`Configure MLCChat in JSON<configure-mlc-chat-json>`.

Raw Text Generation in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Raw text generation allows the user to have more flexibility over his prompts, 
without being forced to create a new conversational template, making prompt customization easier.
This serves other demands for APIs to handle LLM generation without the usual system prompts and other items.

We provide an example below.

.. code:: python

   from mlc_chat import ChatModule, ChatConfig, ConvConfig
   from mlc_chat.callback import StreamToStdout

   # Use a `ConvConfig` to define the generation settings
   # Since the "LM" template only supports raw text generation,
   # system prompts will not be executed even if provided
   conv_config = ConvConfig(stop_tokens=[2,], add_bos=True, stop_str="[INST]")

   # Note that `conv_config` is an optional subfield of `chat_config`
   # The "LM" template serves the basic purposes of raw text generation
   chat_config = ChatConfig(conv_config=conv_config, conv_template="LM")

   # Using the `chat_config` we created, instantiate a `ChatModule`
   cm = ChatModule('Llama-2-7b-chat-hf-q4f16_1', chat_config=chat_config)

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

Gradio Frontend
---------------

The gradio frontend provides a web interface for the MLC-Chat model, which allows user to interact with the model in a more user-friendly way and switch between different models to compare performance.
To use gradio frontend, you need to install gradio first:

.. code-block:: bash

   pip install gradio

Then you can run the following code to start the interface:

.. code:: bash

   python -m mlc_chat.gradio --artifact-path ARTIFACT_PATH [--device DEVICE] [--port PORT_NUMBER] [--share]

--artifact-path        Please provide a path containing all the model folders you wish to use. The default value is ``dist``.
--device               The description of the device to run on. User should provide a string in the form of 'device_name:device_id' or 'device_name', where 'device_name' is one of 'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the local device), and 'device_id' is the device id to run on. If no 'device_id' is provided, it will be set to 0. The default value is ``auto``.
--port                 The port number to run gradio. The default value is ``7860``.   
--share                Whether to create a publicly shareable link for the interface.

After setting up properly, you are expected to see the following interface in your browser:

.. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/gradio-interface.png
   :width: 100%
   :align: center
