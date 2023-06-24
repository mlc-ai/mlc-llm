Python API and Gradio Frontend
==============================

.. contents:: Table of Contents
   :local:
   :depth: 2

We expose Python API for the MLC-Chat for easy integration into other Python projects,
we also provide a web demo based on `gradio <https://gradio.app/>`_ as an example of using Python API to interact with MLC-Chat.

Python API
----------

The Python API is a part of the MLC-Chat package, which we have prepared pre-built pip wheels and you can install it by
following the instructions in `<https://mlc.ai/package/>`_.

Verify Installation
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   python -c "from mlc_chat import ChatModule; print(ChatModule)"

You are expected to see the information about the :class:`ChatModule` class.

If the prebuild is unavailable on your platform, or you would like to build a runtime
that supports other GPU runtime than the prebuilt version. Please refer our :ref:`Build MLC-Chat Package From Source<mlcchat_build_from_source>` tutorial.

API Reference
-------------

User can initiate a chat module by creating :class:`ChatModule` class, which is a wrapper of the MLC-Chat model.
The :class:`ChatModule` class provides the following methods:

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
