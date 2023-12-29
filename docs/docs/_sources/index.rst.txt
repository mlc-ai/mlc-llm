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

    **Install MLC Chat Python**. :doc:`MLC LLM <install/mlc_llm>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    **Download pre-quantized weights**. The commands below download the int4-quantized Llama2-7B from HuggingFace:

    .. code:: bash

      git lfs install && mkdir dist/
      git clone https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC \
                                        dist/Llama-2-7b-chat-hf-q4f16_1-MLC

    **Download pre-compiled model library**. The pre-compiled model library is available as below:

    .. code:: bash

      git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs

    **Run in Python.** The following Python script showcases the Python API of MLC LLM and its stream capability:

    .. code:: python

      from mlc_chat import ChatModule
      from mlc_chat.callback import StreamToStdout

      cm = ChatModule(
          model="dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
          model_lib_path="dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
          # Vulkan on Linux: Llama-2-7b-chat-hf-q4f16_1-vulkan.so
          # Metal on macOS: Llama-2-7b-chat-hf-q4f16_1-metal.so
          # Other platforms: Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}
      )
      cm.generate(prompt="What is the meaning of life?", progress_callback=StreamToStdout(callback_interval=2))

    **Colab walkthrough.**  A Jupyter notebook on `Colab <https://colab.research.google.com/github/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb>`_
    is provided with detailed walkthrough of the Python API.

    **Documentation and tutorial.** Python API reference and its tutorials are `available online <https://llm.mlc.ai/docs/deploy/python.html#api-reference>`_.

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/python-api.jpg
      :width: 600
      :align: center

      MLC LLM Python API

  .. tab:: Command Line

    **Install MLC Chat CLI.** MLC Chat CLI is available via conda using the command below.
    It is always recommended to install it in an isolated conda virtual environment.
    For Windows/Linux users, make sure to have latest :ref:`Vulkan driver <vulkan_driver>` installed.

    .. code:: bash

      conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-cli-nightly
      conda activate mlc-chat-venv

    **Download pre-quantized weights**. The comamnds below download the int4-quantized Llama2-7B from HuggingFace:

    .. code:: bash

      git lfs install && mkdir dist/
      git clone https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC \
                                        dist/Llama-2-7b-chat-hf-q4f16_1-MLC

    **Download pre-compiled model library**. The pre-compiled model library is available as below:

    .. code:: bash

      git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs

    **Run in command line**.

    .. code:: bash

      mlc_chat_cli --model dist/Llama-2-7b-chat-hf-q4f16_1-MLC \
                   --model-lib-path dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-vulkan.so
                   # Metal on macOS: dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-metal.so

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/Llama2-macOS.gif
      :width: 500
      :align: center

      MLC LLM on CLI

    .. note::
      The MLC Chat CLI package is only built with Vulkan (Windows/Linux) and Metal (macOS).
      To use other GPU backends such as CUDA and ROCm, please use the prebuilt Python package or build from source.

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
      :target: https://github.com/mlc-ai/binary-mlc-llm-libs/raw/main/mlc-chat.apk

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
   deploy/python.rst
   deploy/ios.rst
   deploy/android.rst

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
