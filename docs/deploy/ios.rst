.. _deploy-ios:

iOS App and Swift API
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

The MLC LLM iOS app can be installed in two ways: through the pre-built package or by building from the source.
If you are an iOS user looking to try out the models, the pre-built package is recommended. If you are a
developer seeking to integrate new features into the package, building the iOS package from the source is required.

Use Pre-built iOS App
---------------------
The MLC Chat app is now available in App Store at no cost. You can download and explore it by simply clicking the button below:

    .. image:: https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg
      :width: 135
      :target: https://apps.apple.com/us/app/mlc-chat/id6448482937


Build iOS App from Source
-------------------------

This section shows how we can build the app from the source.

Step 1. Install Build Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First and foremost, please clone the `MLC LLM GitHub repository <https://github.com/mlc-ai/mlc-llm>`_.

Please follow :doc:`/install/tvm` to install TVM Unity.
Note that we **do not** have to run `build.py` since we can use prebuilt weights.
We only need TVM Unity's utility to combine the libraries (`local-id-iphone.tar`) into a single library.

We also need to have the following build dependencies:

* CMake >= 3.24,
* Git and Git-LFS,
* `Rust and Cargo <https://www.rust-lang.org/tools/install>`_, which are required by Hugging Face's tokenizer.


Step 2. Download Prebuilt Weights and Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You also need to obtain a copy of the MLC-LLM source code
by cloning the `MLC LLM GitHub repository <https://github.com/mlc-ai/mlc-llm>`_.
To simplify the build, we will use prebuilt model
weights and libraries here. Run the following command
in the root directory of the MLC-LLM.

.. code:: bash

   mkdir -p dist/prebuilt
   git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib

   cd dist/prebuilt
   git lfs install
   git clone https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
   cd ../..

Validate that the files and directories exist:

.. code:: bash

   >>> ls -l ./dist/prebuilt/lib/*/*-iphone.tar
   ./dist/prebuilt/lib/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar
   ./dist/prebuilt/lib/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q3f16_1-iphone.tar
   ...

   >>> ls -l ./dist/prebuilt/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
   # chat config:
   mlc-chat-config.json
   # model weights:
   ndarray-cache.json
   params_shard_*.bin
   ...


Step 3. Build Auxiliary Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tokenizer and runtime**

In addition to the model itself, a lightweight runtime and tokenizer are
required to actually run the LLM. You can build and organize these
components by following these steps:

.. code:: bash

   git submodule update --init --recursive
   cd ./ios
   ./prepare_libs.sh

This will create a ``./build`` folder that contains the following files.
Please make sure all the following files exist in ``./build/``.

.. code:: bash

   >>> ls ./build/lib/
   libmlc_llm.a         # A lightweight interface to interact with LLM, tokenizer, and TVM Unity runtime
   libmodel_iphone.a    # The compiled model lib
   libsentencepiece.a   # SentencePiece tokenizer
   libtokenizers_cpp.a  # Huggingface tokenizer
   libtvm_runtime.a     # TVM Unity runtime

**Add prepackage model**

We can also *optionally* add prepackage weights into the app,
run the following command under the ``./ios`` directory:

.. code:: bash

   cd ./ios
   open ./prepare_params.sh # make sure builtin_list only contains "RedPajama-INCITE-Chat-3B-v1-q4f16_1"
   ./prepare_params.sh

The outcome should be as follows:

.. code:: bash

   >>> ls ./dist/
   RedPajama-INCITE-Chat-3B-v1-q4f16_1

Step 4. Build iOS App
^^^^^^^^^^^^^^^^^^^^^

Open ``./ios/MLCChat.xcodeproj`` using Xcode. Note that you will need an
Apple Developer Account to use Xcode, and you may be prompted to use
your own developer team credential and product bundle identifier.

Ensure that all the necessary dependencies and configurations are
correctly set up in the Xcode project.

Once you have made the necessary changes, build the iOS app using Xcode.
If you have an Apple Silicon Mac, you can select target "My Mac (designed for iPad)"
to run on your Mac. You can also directly run it on your iPad or iPhone.

.. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/xcode-build.jpg
   :align: center
   :width: 60%

|

Customize the App
-----------------

We can customize the iOS app in several ways.
`MLCChat/app-config.json <https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCChat/app-config.json>`_
controls the list of local and remote models to be packaged into the app, given a local path or a URL respectively. Only models in ``model_list`` will have their libraries brought into the app when running `./prepare_libs` to package them into ``libmodel_iphone.a``. Each model defined in `app-config.json` contain the following fields:

``model_path``
   (Required if local model) Name of the local folder containing the weights.

``model_url``
   (Required if remote model) URL to the repo containing the weights.

``model_id``
  (Required) Unique local identifier to identify the model.

``model_lib``
   (Required) Matches the system-lib-prefix, generally set during ``mlc_chat compile`` which can be specified using 
   ``--system-lib-prefix`` argument. By default, it is set to ``"${model_type}_${quantization}"`` e.g. ``gpt_neox_q4f16_1`` 
   for the RedPajama-INCITE-Chat-3B-v1 model. If the ``--system-lib-prefix`` argument is manually specified during 
   ``mlc_chat compile``, the ``model_lib`` field should be updated accordingly.

``model_lib_path``
   (Required) Path to the model library (``.tar`` file) in the repo.

``required_vram_bytes``
   (Required) Estimated requirements of VRAM to run the model.

````

Additionally, the app prepackages the models under ``./ios/dist``.
This built-in list can be controlled by editing ``prepare_params.sh``.
You can package new prebuilt models or compiled models by changing the above fields and then repeating the steps above.


Build Apps with MLC Swift API
-----------------------------

We also provide a Swift package that you can use to build
your own app. The package is located under `ios/MLCSwift`.

- First make sure you have run the same steps listed
  in the previous section. This will give us the necessary libraries
  under ``/path/to/ios/build/lib``.
- Then you can add ``ios/MLCSwift`` package to your app in Xcode.
  Under "Frameworks, Libraries, and Embedded Content", click add package dependencies
  and add local package that points to ``ios/MLCSwift``.
- Finally, we need to add the libraries dependencies. Under build settings:

  - Add library search path ``/path/to/ios/build/lib``.
  - Add the following items to "other linker flags".

   .. code::

      -Wl,-all_load
      -lmodel_iphone
      -lmlc_llm -ltvm_runtime
      -ltokenizers_cpp
      -lsentencepiece
      -ltokenizers_c


You can then import the `MLCSwift` package into your app.
The following code shows an illustrative example of how to use the chat module.

.. code:: swift

   import MLCSwift

   let threadWorker = ThreadWorker()
   let chat = ChatModule()

   threadWorker.push {
      let modelLib = "model-lib-name"
      let modelPath = "/path/to/model/weights"
      let input = "What is the capital of Canada?"
      chat.reload(modelLib, modelPath: modelPath)

      chat.prefill(input)
      while (!chat.stopped()) {
         displayReply(chat.getMessage())
         chat.decode()
      }
   }

.. note::

   Because the chat module makes heavy use of GPU and thread-local
   resources, it needs to run on a dedicated background thread.
   Therefore, **avoid using** `DispatchQueue`, which can cause context switching to
   different threads and segfaults due to thread-safety issues.
   Use the `ThreadWorker` class to launch all the jobs related
   to the chat module. You can check out the source code of
   the MLCChat app for a complete example.
