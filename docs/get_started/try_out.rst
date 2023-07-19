.. _get_started:

Try out MLC Chat
================

Welcome to MLC LLM. To get started, we have prepared prebuilt packages
for you to try out MLC Chat app built with MLC LLM,
and you can try out prebuilt models on the following platforms:

.. tabs::

  .. tab:: CLI (Linux/MacOS/Windows)

    **MLC LLM now supports 7B/13B Llama-2.** *Follow the instructions below and try out the CLI!*

    To run the models on your PC, you can try out the CLI version of MLC LLM.

    We have prepared Conda packages for MLC Chat CLI. If you haven't installed Conda yet,
    please refer to :doc:`this tutorial </install/conda>` to install Conda.

    .. code:: bash

      # Create a new conda environment, install CLI app, and activate the environment.
      conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-cli-nightly
      conda activate mlc-chat-venv

      # Install Git and Git-LFS if you haven't already.
      # They are used for downloading the model weights from HuggingFace.
      conda install git git-lfs
      git lfs install

      # Create a directory, download the model weights from HuggingFace, and download the binary libraries
      # from GitHub.
      mkdir -p dist/prebuilt
      git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib

      # Download prebuilt weights of Llama-2-7B or Llama-2-13B
      cd dist/prebuilt
      git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
      # or the 13B model
      # git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f16_1
      cd ../..
      mlc_chat_cli --local-id Llama-2-7b-chat-hf-q4f16_1
      # or the 13B model
      # mlc_chat_cli --local-id Llama-2-13b-chat-hf-q4f16_1

      # You can try more models, for example:
      # download prebuilt weights of RedPajama-3B
      cd dist/prebuilt
      git clone https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0
      cd ../..
      mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_0

    .. note::
      If you are using Windows or Linux. Make sure you have the latest Vulkan driver installed.
      Please follow the instructions in :doc:`/install/gpu` tutorial to prepare the environment.

    You can also checkout the :doc:`/prebuilt_models` page to run other models.

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/Llama2-macOS.gif
      :width: 500
      :align: center

      MLC LLM on CLI

  .. tab:: Web Browser

    With the advancements of WebGPU, we can now run LLM completely in the web browser environment.
    You can try out the web version of MLC LLM in `WebLLM <https://webllm.mlc.ai/#chat-demo>`__.

    In WebLLM, once the model weights are fetched and stored in the local cache in the first run, you can start to interact with the model without Internet connection.

    A WebGPU-compatible browser and a local GPU are needed to run WebLLM. You can download the latest Google Chrome and use `WebGPU Report <https://webgpureport.org/>`__ to verify the functionality of WebGPU on your browser.

    .. figure:: https://mlc.ai/blog/img/redpajama/web.gif
      :width: 300
      :align: center

      MLC LLM on Web

  .. tab:: iOS

    The MLC Chat app is now available in App Store at no cost. You can download and explore it by simply clicking the button below:

    .. image:: https://linkmaker.itunes.apple.com/assets/shared/badges/en-us/appstore-lrg.svg
      :width: 135
      :target: https://apps.apple.com/us/app/mlc-chat/id6448482937

    |

    **MLC LLM now supports Llama-2 via the test link below** *

    .. note::
      You can also try out the beta version of MLC-Chat on
      `TestFlight <https://testflight.apple.com/join/57zd7oxa>`__.

    Once the app is installed, you can download the models and then engage in chat with the model without requiring an internet connection.

    Memory requirements vary across different models. The Vicuna-7B model necessitates an iPhone device with a minimum of 6GB RAM, whereas the RedPajama-3B model can run on an iPhone with at least 4GB RAM.

    .. figure:: https://mlc.ai/blog/img/redpajama/ios.gif
      :width: 300
      :align: center

      MLC Chat on iOS

  .. tab:: Android

    The MLC Chat Android app is free and available for download, and you can try out by simply clicking the button below:

    .. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
      :width: 135
      :target: https://github.com/mlc-ai/binary-mlc-llm-libs/raw/main/mlc-chat.apk

    |

    Once the app is installed, you can engage in a chat with the model without the need for an internet connection:

    Memory requirements vary across different models. The Vicuna-7B model necessitates an Android device with a minimum of 6GB RAM, whereas the RedPajama-3B model can run on an Android device with at least 4GB RAM.

    .. figure:: https://mlc.ai/blog/img/android/android-recording.gif
      :width: 300
      :align: center

      MLC LLM on Android
