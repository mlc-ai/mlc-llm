iOS App and Swift API
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

The MLC LLM iOS app can be installed in two ways: through the pre-built package or by building it from source. If you're an iOS user looking to try out the models, the pre-built package is recommended. However, if you're a developer seeking to integrate new features into the package, building the iOS package from source is required.

Use Pre-built iOS App
---------------------
The MLC LLM app is accessible on the App Store at no cost. You can download and explore it by simply clicking the button below:

    .. image:: https://linkmaker.itunes.apple.com/assets/shared/badges/en-us/appstore-lrg.svg
      :width: 135
      :target: https://apps.apple.com/us/app/mlc-chat/id6448482937

Build iOS Package from Source
-----------------------------

We will use ``vicuna-v1-7b`` as an example to demonstrate the
quantization and cross-compilation workflow for LLM on iOS devices. The
same procedure can be applied to other models as well, such as Dolly and
RedPajama.

Step 1. Install TVM Unity
^^^^^^^^^^^^^^^^^^^^^^^^^

Please follow :doc:`/install/tvm` to install TVM Unity.

Step 2. Download LLM weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our tutorial :doc:`/compilation/get-vicuna-weight` to get Vicuna weights. The contents of the folder should include the following files:

.. code:: bash

   >>> ls ./dist/models/vicuna-v1-7b
   config.json
   pytorch_model.bin.index.json
   pytorch_model-00001-of-00002.bin
   pytorch_model-00002-of-00002.bin
   tokenizer.model
   tokenizer_config.json
   special_tokens_map.json

Step 3. Quantize and cross-compile LLMs with TVM Unity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To quantize and cross-compile the LLMs to iPhone/iPad targets, use the
provided ``build.py`` script:

.. code:: bash


   python3 build.py                    \
       --model /path/to/vicuna-v1-7b   \
       --quantization q3f16_0          \
       --target iphone                 \
       --max-seq-len 768

   # If the model path is `./dist/models/vicuna-v1-7b`,
   # we can simplify the build command to
   python3 build.py            \
       --model vicuna-v1-7b    \
       --quantization q3f16_0  \
       --target iphone         \
       --max-seq-len 768

By default, the compiled artifact will be located under
``./dist/vicuna-v1-7b-q3f16_0``. The folder includes: - The static
library for LLM computation: ``vicuna-v1-7b-q3f16_0-iphone.a`` - Assets,
including weight shards under ``params/`` - Tokenizer metadata

Step 4. Build auxiliary components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tokenizer and runtime**

In addition to the model itself, a lightweight runtime and tokenizer are
required to actually run the LLM. You can build and organize these
components by following these steps:

.. code:: bash

   cd ./ios
   vim ./prepare_libs.sh # Update MODEL, QUANTIZATION, and other variables
   ./prepare_libs.sh

This will create a ``./build`` folder that contains the following files:

.. code:: bash

   >>> ls ./build/lib/
   libmlc_llm.a         # The lightweight interface to interact with LLM, tokenizer, and TVM Unity runtime
   libmodel_iphone.a    # The compiled LLM
   libsentencepiece.a   # SentencePiece tokenizer
   libtokenizers_cpp.a  # Huggingface tokenizer
   libtvm_runtime.a     # TVM Unity runtime

**Collect assets**

To organize the assets, execute the following script within the
``./ios`` directory:

.. code:: bash

   cd ./ios
   vim ./prepare_params.sh # Update MODEL, QUANTIZATION, TOKENIZER, and other variables
   ./prepare_params.sh

The outcome should be as follows:

.. code:: bash

   >>> ls ./dist/
   params/              # Parameter shards
   tokenizer.json       # Tokenizer metadata

Step 5. Build iOS App
^^^^^^^^^^^^^^^^^^^^^

Open ``./ios/MLCChat.xcodeproj`` using Xcode. Note that you will need an
Apple Developer Account to use Xcode, and you may be prompted to use
your own developer team credential and product bundle identifier.

Ensure that all the necessary dependencies and configurations are
correctly set up in the Xcode project.

Once you have made the necessary changes, build the iOS app using Xcode.
Make sure to select a target device or simulator for the build.

After a successful build, you can run the iOS app on your device or
simulator to use the LLM model for text generation and processing.


Build your own App with MLC Swift API
-------------------------------------

We also provide an swift package that you can use to build
your own app. The package is located under `ios/MLCSwift`.

- First make sure you have run the same steps listed
  this this document. This will give us the necessary libraries
  under `/path/to/ios/build/lib`.
- Then you can add `ios/MLCSwift` package to your app in xcode.
  Under frameworks libraries embedded content, click add package dependencies
  and add local package that points to ios/MLCSwift
- Finally, we need to add the libraries dependencies. Under build settings:

  - Add library search path `/path/to/ios/build/lib`
  - Add the following items to "other linker flags"

   .. code::

      -Wl,-all_load
      -lmodel_iphone
      -lmlc_llm -ltvm_runtime
      -Wl,-noall_load
      -ltokenizers_cpp
      -lsentencepiece
      -ltokenizers_c


You can then can import the `MLCSwift` package in your app.
The following code shows an illustrative example about how to use the chat module.

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

Because the chat module makes heavy use of GPU and thread-local
resources, it needs to run on a dedicated background thread.
Do not use DispatchQueue, as that can cause context switching to
different threads and segfaults due to thread-safety issue.
Use the ThreadWorker class to launch all the jobs related
to the chat module. You can checkot the source code of
the MLCChat app for a complete example.
