👋 Welcome to MLC LLM
=====================

`Discord <https://discord.gg/9Xpy2HGBuD>`_ | `GitHub <https://github.com/mlc-ai/mlc-llm>`_

Machine Learning Compilation for Large Language Models (MLC LLM) is a high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration. The mission of this project is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices with ML compilation techniques.

.. _get_started:

Getting Started
---------------

To begin with, try out MLC LLM support for int4-quantized Llama2 7B.
It is recommended to have at least 6GB free VRAM to run it.

.. tabs::

  .. tab:: Python

    **Install MLC LLM**. :doc:`MLC LLM <install/mlc_llm>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    **Run chat completion in Python.** The following Python script showcases the Python API of MLC LLM:

    .. code:: python

      from mlc_llm import Engine

      # Create engine
      model = "HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC"
      engine = Engine(model)

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

    **Documentation and tutorial.** Python API reference and its tutorials are :doc:`available online <deploy/python_engine>`.

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/python-engine-api.jpg
      :width: 600
      :align: center

      MLC LLM Python API

  .. tab:: REST Server

    **Install MLC LLM**. :doc:`MLC LLM <install/mlc_llm>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    **Launch a REST server.** Run the following command from command line to launch a REST server at ``http://127.0.0.1:8000``.

    .. code:: shell

      mlc_llm serve HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC

    **Send requests to server.** When the server is ready (showing ``INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)``),
    open a new shell and send a request via the following command:

    .. code:: shell

      curl -X POST \
        -H "Content-Type: application/json" \
        -d '{
              "model": "HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC",
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

    **Install MLC LLM**. :doc:`MLC LLM <install/mlc_llm>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    For Windows/Linux users, make sure to have latest :ref:`Vulkan driver <vulkan_driver>` installed.

    **Run in command line**.

    .. code:: bash

      mlc_llm chat HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC


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

    **Requirement**. Llama2-7B model needs an iOS device with a minimum of 6GB RAM, whereas the RedPajama-3B model runs with at least 4GB RAM.

    **Tutorial and source code**. The source code of the iOS app is fully `open source <https://github.com/mlc-ai/mlc-llm/tree/main/ios>`__,
    and a :doc:`tutorial <deploy/ios>` is included in documentation.

    .. figure:: https://blog.mlc.ai/img/redpajama/ios.gif
      :width: 300
      :align: center

      MLC Chat on iOS

  .. tab:: Android

    **Install MLC Chat Android**. A prebuilt is available as an APK:

    .. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
      :width: 135
      :target: https://github.com/mlc-ai/binary-mlc-llm-libs/releases/download/Android/mlc-chat.apk

    |

    **Requirement**. Llama2-7B model needs a device with a minimum of 6GB RAM, whereas the RedPajama-3B model runs with at least 4GB RAM.
    The demo is tested on

    - Samsung S23 with Snapdragon 8 Gen 2 chip
    - Redmi Note 12 Pro with Snapdragon 685
    - Google Pixel phones

    **Tutorial and source code**. The source code of the android app is fully `open source <https://github.com/mlc-ai/mlc-llm/tree/main/android>`__,
    and a :doc:`tutorial <deploy/android>` is included in documentation.

    .. figure:: https://blog.mlc.ai/img/android/android-recording.gif
      :width: 300
      :align: center

      MLC LLM on Android


What to Do Next
---------------

- Depending on your use case, check out our API documentation and tutorial pages:

  - :ref:`webllm-runtime`
  - :ref:`deploy-rest-api`
  - :ref:`deploy-cli`
  - :ref:`deploy-python-engine`
  - :ref:`deploy-ios`
  - :ref:`deploy-android`
  - :ref:`deploy-ide-integration`

- Deploy your local model: check out :ref:`convert-weights-via-MLC` to convert your model weights to MLC format.
- Deploy models to Web or build iOS/Android apps on your own: check out :ref:`compile-model-libraries` to compile the models into binary libraries.
- Customize model optimizations: check out :ref:`compile-model-libraries`.
- Report any problem or ask any question: open new issues in our `GitHub repo <https://github.com/mlc-ai/mlc-llm/issues>`_.


.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:

   get_started/project_overview.rst
   get_started/mlc_chat_config.rst

.. toctree::
   :maxdepth: 1
   :caption: Build and Deploy Apps
   :hidden:

   deploy/javascript.rst
   deploy/rest.rst
   deploy/cli.rst
   deploy/python_engine.rst
   deploy/ios.rst
   deploy/android.rst
   deploy/ide_integration.rst

.. toctree::
   :maxdepth: 1
   :caption: Compile Models
   :hidden:

   compilation/convert_weights.rst
   compilation/compile_models.rst
   compilation/define_new_models.rst
   compilation/configure_quantization.rst

.. toctree::
   :maxdepth: 1
   :caption: Model Prebuilts
   :hidden:

   prebuilt_models.rst
   prebuilt_models_deprecated.rst

.. toctree::
   :maxdepth: 1
   :caption: Dependency Installation
   :hidden:

   install/tvm.rst
   install/mlc_llm.rst
   install/conda.rst
   install/gpu.rst
   install/emcc.rst

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   community/guideline.rst
   community/faq.rst


.. toctree::
   :maxdepth: 1
   :caption: Privacy
   :hidden:

   privacy.rst
