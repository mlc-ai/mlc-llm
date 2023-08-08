Python API and Gradio Frontend
==============================

.. contents:: Table of Contents
   :local:
   :depth: 2

We expose Python API for the MLC-Chat for easy integration into other Python projects.
We also provide a web demo based on `gradio <https://gradio.app/>`_ as an example of using Python API to interact with MLC-Chat.

Python API
----------

The Python API is a part of the MLC-Chat package, which we have prepared pre-built pip wheels and you can install it by
following the instructions in `<https://mlc.ai/package/>`_.

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

After making sure that the files exist, from the ``mlc-llm`` directory, we can
run the following code with the conda environment you used to install ``mlc_chat``:

.. code:: python

   from mlc_chat import ChatModule
   cm = ChatModule(model='Llama-2-7b-chat-hf-q4f16_1')  # Create a ChatModule instance
   output = cm.generate(prompt="What is the meaning of life?") # Generate a response for a given prompt
   print(output)
   runtime_stats = cm.stats()  # Print some runtime statistics for the generation
   print(f"{runtime_stats=}")
   cm.reset_chat()  # Reset the chat module


.. collapse:: See output

   .. code::

      Using model folder: ./dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
      Using mlc chat config: ./dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1/mlc-chat-config.json
      Using library model: ./dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so

      Hello! I'm here to help you with your question. However, I must point out
      ...

      runtime_stats='prefill: 336.0 tok/s, decode: 118.5 tok/s'

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

   # Using a `ConvConfig`, we modify `system`, a field in the conversation template
   # `system` refers to the prompt encoded before starting the chat
   conv_config = ConvConfig(system='Please show as much happiness as you can when talking to me.')

   # We then include the `ConvConfig` instance in `ChatConfig` while overriding `max_gen_len`
   # Note that `conv_config` is an optional subfield of `chat_config`
   chat_config = ChatConfig(max_gen_len=256, conv_config=conv_config)

   # Using the `chat_config` we created, instantiate a `ChatModule`
   cm = mlc_chat.ChatModule('Llama-2-7b-chat-hf-q4f16_1', chat_config=chat_config)
   print(cm.generate(prompt='What is one plus one?'))

   # You could also pass in a `ConvConfig` instance to `reset_chat()`
   conv_config = ConvConfig(system='Please show as much sadness as you can when talking to me.')
   chat_config = ChatConfig(max_gen_len=128, conv_config=conv_config)
   cm.reset_chat(chat_config)
   print(cm.generate(prompt='What is one plus one?'))


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

Gradio Frontend
---------------

The gradio frontend provides a web interface for the MLC-Chat model, which allows user to interact with the model in a more user-friendly way.
To use gradio frontend, you need to install gradio first:

.. code-block:: bash

   pip install gradio

Then you can run the following code to start the interface:

.. code:: bash

   python -m mlc_chat.gradio --artifact-path ARTIFACT_PATH --device-name DEVICE_NAME --device-id DEVICE_ID [--port PORT_NUMBER] [--share]

--artifact-path        The path to the artifact folder where models are stored. The default value is ``dist``.
--device-name          The device name to run the model. Available options are:
                       ``metal``, ``cuda``, ``vulkan``, ``cpu``. The default value is ``cuda``.
--device-id            The device id to run the model. The default value is ``0``.
--port                 The port number to run gradio. The default value is ``7860``.   
--share                Whether to create a publicly shareable link for the interface.

After setting up properly, you are expected to see the following interface in your browser:

.. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/gradio-interface.png
   :width: 100%
   :align: center
