.. _get_started:

Try out MLC LLM on your device
==============================

We have prepared packages for you to try out MLC LLM locally, and you can try out prebuilt models on your device:

.. tabs::

  .. tab:: iOS

    The MLC LLM app is now accessible on the App Store at no cost. You can download and explore it by simply clicking the button below:

    .. image:: https://linkmaker.itunes.apple.com/assets/shared/badges/en-us/appstore-lrg.svg
      :width: 135
      :target: https://apps.apple.com/us/app/mlc-chat/id6448482937

    Once the app is installed, you can download the models and then engage in chat with the model without requiring an internet connection.

    Memory requirements vary across different models. The Vicuna-7B model necessitates an iPhone device with a minimum of 6GB RAM, whereas the RedPajama-3B model can run on an iPhone with at least 4GB RAM.

    .. figure:: https://mlc.ai/blog/img/redpajama/ios.gif
      :width: 300
      :align: center

      MLC LLM on iOS

  .. tab:: Android

    The MLC LLM Android app is free and available for download and can be tried out by simply clicking the button below:

    .. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
      :width: 135
      :target: https://github.com/mlc-ai/binary-mlc-llm-libs/raw/main/mlc-chat.apk

    Once the app is installed, you can engage in a chat with the model without the need for an internet connection:

    Memory requirements vary across different models. The Vicuna-7B model necessitates an Android device with a minimum of 6GB RAM, whereas the RedPajama-3B model can run on an Android device with at least 4GB RAM.

    .. figure:: https://mlc.ai/blog/img/android/android-recording.gif
      :width: 300
      :align: center

      MLC LLM on Android

  .. tab:: CLI (Linux/MacOS/Windows)

    To utilize the models on your PC, we highly recommend trying out the CLI version of MLC LLM.

    We have prepared Conda packages for MLC Chat CLI. If you haven't installed Conda yet, please refer to :doc:`this tutorial </install/conda>` to install Conda.
    To install MLC Chat CLI, simply run the following command:

    .. code:: bash

      conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-nightly
      conda activate mlc-chat-venv

    If you are using Windows or Linux and want to use your native GPU, please follow the instructions in :doc:`/install/gpu` tutorial to prepare the environment.

    .. figure:: https://mlc.ai/blog/img/redpajama/cli.gif
      :width: 300
      :align: center

      MLC LLM on CLI

  .. tab:: Web Browser

    With the advancements of WebGPU, we can now run LLM directly on web browsers. You have the opportunity to experience the web version of MLC LLM through `WebLLM <https://mlc.ai/webllm>`__.

    Once the parameters have been fetched and stored in the local cache, you can begin interacting with the model without the need for an internet connection.

    You can use `WebGPU Report <https://webgpureport.org/>`__ to verify the functionality of WebGPU on your browser.

    .. figure:: https://mlc.ai/blog/img/redpajama/web.gif
      :width: 300
      :align: center

      MLC LLM on Web


Customize MLC-Chat Configuration
--------------------------------

The behavior of the chat can be customized by modifying the chat configuration file. To learn more about customizing the chat configuration JSON, you can refer to the following tutorials which provide a detailed walkthrough:

- :doc:`/get_started/mlc_chat_config`

Model Prebuilts
---------------

To use different pre-built models, you can refer to the following tutorials:

- :doc:`/tutorials/prebuilts/prebuilt_models`
