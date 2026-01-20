.. _quick-start:

Quick Start
===========

Examples
--------

To begin with, try out MLC LLM support for int4-quantized Llama3 8B.
It is recommended to have at least 6GB free VRAM to run it.

.. tabs::

  .. tab:: Python

    **Install MLC LLM**. :ref:`MLC LLM <install-mlc-packages>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    **Run chat completion in Python.** The following Python script showcases the Python API of MLC LLM:

    .. code:: python

      from mlc_llm import MLCEngine

      # Create engine
      model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
      engine = MLCEngine(model)

      # Run chat completion in OpenAI API.
      for response in engine.chat.completions.create(
          messages=[{"role": "user", "content": "What is the meaning of life?"}],
          model=model,
          stream=True,
      ):
          for choice in response.choices:
              print(choice.delta.content, end="", flush=True)
      print("\n")

      engine.terminate()

    .. Todo: link the colab notebook when ready:

    **Documentation and tutorial.** Python API reference and its tutorials are :ref:`available online <deploy-python-engine>`.

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/python-engine-api.jpg
      :width: 600
      :align: center

      MLC LLM Python API

  .. tab:: REST Server

    **Install MLC LLM**. :ref:`MLC LLM <install-mlc-packages>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    **Launch a REST server.** Run the following command from command line to launch a REST server at ``http://127.0.0.1:8000``.

    .. code:: shell

      mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC

    **Send requests to server.** When the server is ready (showing ``INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)``),
    open a new shell and send a request via the following command:

    .. code:: shell

      curl -X POST \
        -H "Content-Type: application/json" \
        -d '{
              "model": "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
              "messages": [
                  {"role": "user", "content": "Hello! Our project is MLC LLM. What is the name of our project?"}
              ]
        }' \
        http://127.0.0.1:8000/v1/chat/completions

    **Documentation and tutorial.** Check out :ref:`deploy-rest-api` for the REST API reference and tutorial.
    Our REST API has complete OpenAI API support.

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/python-serve-request.jpg
      :width: 600
      :align: center

      Send HTTP request to REST server in MLC LLM

  .. tab:: Command Line

    **Install MLC LLM**. :ref:`MLC LLM <install-mlc-packages>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    For Windows/Linux users, make sure to have latest :ref:`Vulkan driver <vulkan_driver>` installed.

    **Run in command line**.

    .. code:: bash

      mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC


    If you are using windows/linux/steamdeck and would like to use vulkan,
    we recommend installing necessary vulkan loader dependency via conda
    to avoid vulkan not found issues.

    .. code:: bash

      conda install -c conda-forge gcc libvulkan-loader


  .. tab:: Web Browser

    `WebLLM <https://webllm.mlc.ai/#chat-demo>`__. MLC LLM generates performant code for WebGPU and WebAssembly,
    so that LLMs can be run locally in a web browser without server resources.

    **Download pre-quantized weights**. This step is self-contained in WebLLM.

    **Download pre-compiled model library**. WebLLM automatically downloads WebGPU code to execute.

    **Check browser compatibility**. The latest Google Chrome provides WebGPU runtime and `WebGPU Report <https://webgpureport.org/>`__ as a useful tool to verify WebGPU capabilities of your browser.

    .. figure:: https://blog.mlc.ai/img/redpajama/web.gif
      :width: 300
      :align: center

      MLC LLM on Web

  .. tab:: iOS

    **Install MLC Chat iOS**. It is available on AppStore:

    .. image:: https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg
      :width: 135
      :target: https://apps.apple.com/us/app/mlc-chat/id6448482937

    |

    **Note**. The larger model might take more VRAM, try start with smaller models first.

    **Tutorial and source code**. The source code of the iOS app is fully `open source <https://github.com/mlc-ai/mlc-llm/tree/main/ios>`__,
    and a :ref:`tutorial <deploy-ios>` is included in documentation.

    .. figure:: https://blog.mlc.ai/img/redpajama/ios.gif
      :width: 300
      :align: center

      MLC Chat on iOS

  .. tab:: Android

    **Install MLC Chat Android**. A prebuilt is available as an APK:

    .. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
      :width: 135
      :target: https://github.com/mlc-ai/binary-mlc-llm-libs/releases/download/Android-09262024/mlc-chat.apk

    |

    **Note**. The larger model might take more VRAM, try start with smaller models first.
    The demo is tested on

    - Samsung S23 with Snapdragon 8 Gen 2 chip
    - Redmi Note 12 Pro with Snapdragon 685
    - Google Pixel phones

    **Tutorial and source code**. The source code of the android app is fully `open source <https://github.com/mlc-ai/mlc-llm/tree/main/android>`__,
    and a :ref:`tutorial <deploy-android>` is included in documentation.

    .. figure:: https://blog.mlc.ai/img/android/android-recording.gif
      :width: 300
      :align: center

      MLC LLM on Android


What to Do Next
---------------

- Check out :ref:`introduction-to-mlc-llm` for the introduction of a complete workflow in MLC LLM.
- Depending on your use case, check out our API documentation and tutorial pages:

  - :ref:`webllm-runtime`
  - :ref:`deploy-rest-api`
  - :ref:`deploy-cli`
  - :ref:`deploy-python-engine`
  - :ref:`deploy-ios`
  - :ref:`deploy-android`
  - :ref:`deploy-ide-integration`

- :ref:`convert-weights-via-MLC`, if you want to run your own models.
- :ref:`compile-model-libraries`, if you want to deploy to web/iOS/Android or control the model optimizations.
- Report any problem or ask any question: open new issues in our `GitHub repo <https://github.com/mlc-ai/mlc-llm/issues>`_.
