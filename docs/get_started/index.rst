Get Started with MLC LLM
========================

.. contents:: Table of Contents
   :local:
   :depth: 2

Welcome to the MLC LLM! Here, you will find everything you need to begin exploring and utilizing our powerful framework that deploys large language models on any device. Whether you're an experienced developer or just starting out, this documentation is designed to provide a seamless learning experience. Discover how to try out MLC LLM on your own device, familiarize yourself with key terminologies, delve into the runtime workflow, and gain an overview of the compilation process and runtime functionality. Our aim is to help you quickly grasp the core ideas and easily navigate through the tutorial, enabling you to leverage the full potential of MLC LLM. Let's get started!

Try out MLC LLM on your device
------------------------------

We have prepared packages for you to try out MLC LLM locally, and you can try out :doc:`a list of prebuilt models </tutorials/prebuilts/prebuilt_models>` on your device:

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

  .. tab:: PC(CLI)

    To utilize the models on your PC, we highly recommend trying out the CLI version of MLC LLM. Installing the CLI version of MLC LLM is made easy by following the tutorial in the :doc:`/install/cli` documentation.

    .. figure:: https://mlc.ai/blog/img/redpajama/cli.gif
      :width: 300
      :align: center
      
      MLC LLM on CLI

  .. tab:: PC(Web Browser)
    
    With the advancements of WebGPU, we can now run LLM directly on web browsers. You have the opportunity to experience the web version of MLC LLM through `WebLLM <https://mlc.ai/webllm>`__.

    Once the parameters have been fetched and stored in the local cache, you can begin interacting with the model without the need for an internet connection.

    You can use `WebGPU Report <https://webgpureport.org/>`__ to verify the functionality of WebGPU on your browser.

    .. figure:: https://mlc.ai/blog/img/redpajama/web.gif
      :width: 300
      :align: center
      
      MLC LLM on Web

.. _terminologies:

Terminologies
-------------

To gain a comprehensive understanding of MLC LLM, it is essential to familiarize ourselves with key terminologies that represent fundamental components within the MLC LLM Chat application. When operating a chat application, three pivotal elements are involved:

.. _model_weights:

Model Weights
^^^^^^^^^^^^^

The models weights include 

- **Neural Network Weights**: which are sharded and stored in a list of binary files with name: ``params_shard_ID.bin`` where ``ID`` is the shard index.
- **Tokenizer files**: which are stored in a list of files under the current directory. The number of files depends on the tokenizer type.

   - For SentencePiece tokenizer, the tokenizer files would be a single ``tokenizer.model`` file.
   - For HuggingFace-style tokenizer, the tokenizer files would be a single ``tokenizer.json`` file.
   - For Byte-Level BPE tokenizer, the tokenizer files would be a ``vocab.json`` file, a ``merges.txt`` and a ``added_tokens.json`` file. 

.. _model_lib:

Model Library
^^^^^^^^^^^^^

The model library refers to the executable libraries that enable the execution of a specific model architecture. On Linux, these libraries have the suffix ``.so``, on macOS, the suffix is ``.dylib``, and on Windows, the library file ends with ``.dll``.

.. _chat_config:

Chat Config
^^^^^^^^^^^

The chat configuration includes settings that allow customization of parameters such as temperature and system prompt. For detailed instructions on how to customize conversations in the chat app, please refer to our documentation: :doc:`/tutorials/runtime/mlc_chat_config`.

Additionally, the chat configuration contains metadata that is essential for the application to locate and execute the model and tokenizers:

``local_id``
  The key uniquely identifies the model within an app.
``model_lib``
  This key specifies which model library to use.
``tokenizer_files``
  This field specifies the list of tokenizer files.

Runtime Workflow
----------------

Once the model weights, model library, and chat configuration are prepared, the MLC-LLM can be employed as an engine to drive a chat application. The diagram below depicts a typical workflow for an application that utilizes the MLC-LLM's capabilities.

.. image:: https://raw.githubusercontent.com/mlc-ai/web-data/de9a5e5b424f36119bd464ddf5a3ddb4c58cc85e/images/mlc-llm/tutorials/mlc-llm-flow.svg
  :width: 100%
  :align: center

On the right side of the figure, you can see pseudo code illustrating the structure of an MLC chat API during the execution of a chat app. Typically, there is a ``ChatModule`` that manages the model. The chat app includes a reload function that takes a ``local_id`` as well as an optional chat configuration override, which allows for overriding settings such as the system prompt and temperature. The MLC Chat runtime utilizes the ``local_id`` and ``model_lib`` to locate the model and initialize its internal state.

All MLC Chat runtimes, including iOS, Web, CLI, and others, make use of these key elements. They are capable of reading the same model weights, although the packaging of the model libraries may vary. For the CLI, the model libraries are stored in a DLL directory. iOS and Android include pre-packaged model libraries within the app itself due to restrictions on dynamic loading. WebLLM, on the other hand, utilizes a ``model_lib_map`` that maps the library name to URLs of WebAssembly (Wasm) files. Thanks to the shared model weights, we can create the weights once and run them across different platforms.

Overview of MLC LLM
-------------------

To prepare model weights, libraries, and chat configurations, we can utilize the compiler component of MLC LLM. The project consists of three distinct submodules: model definition, model compilation, and runtimes.

.. figure:: /_static/img/project-structure.svg
   :width: 600
   :align: center
   :alt: Project Structure

   Three independent submodules in MLC LLM

.. ➀➁➂➃➄➅➆➇➈➉
.. ➊➋➌➍➎➏➐➑➒➓

**➀ Model definition in Python.** MLC offers a variety of pre-defined architectures, such as Llama (e.g., Vicuna, OpenLlama, Llama, Wizard), GPT-NeoX (e.g., RedPajama, Dolly), RNNs (e.g., RWKV), and GPT-J (e.g., MOSS). Model developers could solely define the model in pure Python, without having to touch code generation and runtime.

**➁ Model compilation in Python.** :doc:`TVM Unity </install/tvm>` compiler are configured in pure python, and it quantizes and exports the Python-based model to :ref:`model lib <model_lib>` and quantized :ref:`model weights <model_weights>`. Quantization and optimization algorithms can be developed in pure Python to compress and accelerate LLMs for specific usecases.

**➂ Platform-native runtimes.** Variants of MLCChat are provided on each platform: **C++** for command line, **Javascript** for web, **Swift** for iOS, and **Java** for Android, configurable with a JSON :ref:`chat config <chat_config>`. App developers only need to familiarize with the platform-naive runtimes to integrate MLC-compiled LLMs into their projects.


Further Exploration
-------------------

In addition to the information provided in this page, we offer detailed tutorials on specific topics to help you dive deeper into MLC LLM. If you're interested in trying out advanced features, such as compile your own models, we have dedicated tutorials to guide you through the process. To learn more about each topic mentioned in this documentation, simply click on the corresponding links below. We encourage you to explore these tutorials to expand your knowledge and make the most of MLC LLM. Happy learning!

- :doc:`/tutorials/prebuilts/prebuilt_models`: Check out the prebuilt models available with MLC LLM.
- :ref:`runtime_apis`: Explore how to use MLC-LLM APIs in your own projects.
- :ref:`compile_models`: Learn how to compile your own language models using MLC LLM.
- :ref:`define_new_models`: Learn how to incorporate new model architectures.
- Blog Posts on bringing LLM to the edge
  - `Bringing Hardware Accelerated Language Models to Consumer Devices <https://mlc.ai/blog/2023/05/01/bringing-accelerated-llm-to-consumer-hardware>`_
  - `Bringing Hardware Accelerated Language Models to Android Devices <https://mlc.ai/blog/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices>`_
