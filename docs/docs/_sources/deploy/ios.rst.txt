.. _deploy-ios:

iOS Swift SDK
=============

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
After cloning, go to the ``ios/`` directory.

.. code:: bash

   git clone https://github.com/mlc-ai/mlc-llm.git
   cd mlc-llm
   git submodule update --init --recursive
   cd ./ios


Please follow :doc:`/install/mlc_llm` to obtain a binary build of mlc_llm package. Note that this
is independent from the above source code that we use for iOS package build.
You do not need to build mlc_llm for your host and we can use the prebuilt package for that purpose.

We also need to have the following build dependencies:

* CMake >= 3.24,
* Git and Git-LFS,
* `Rust and Cargo <https://www.rust-lang.org/tools/install>`_, which are required by Hugging Face's tokenizer.

.. _ios-build-runtime-and-model-libraries:

Step 2. Build Runtime and Model Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The models to be built for the iOS app are specified in ``MLCChat/mlc-package-config.json``:
in the ``model_list``,

* ``model`` points to the Hugging Face repository which contains the pre-converted model weights. The iOS app will download model weights from the Hugging Face URL.
* ``model_id`` is a unique model identifier.
* ``estimated_vram_bytes`` is an estimation of the vRAM the model takes at runtime.
* ``"bundle_weight": true`` means the model weights of the model will be bundled into the app when building.
* ``overrides`` specifies some model config parameter overrides.


We have a one-line command to build and prepare all the model libraries:

.. code:: bash

   cd /path/to/MLCChat  # e.g., "ios/MLCChat"
   export MLC_LLM_SOURCE_DIR=/path/to/mlc-llm  # e.g., "../.."
   mlc_llm package

This command mainly executes the following two steps:

1. **Compile models.** We compile each model in ``model_list`` of ``MLCChat/mlc-package-config.json`` into a binary model library.
2. **Build runtime and tokenizer.** In addition to the model itself, a lightweight runtime and tokenizer are required to actually run the LLM.

The command creates a ``./dist/`` directory that contains the runtime and model build output.
Please make sure ``dist/`` follows the structure below, except the optional model weights.

.. code::

   dist
   ├── bundle                   # The directory for mlc-app-config.json (and optionally model weights)
   │   │                        # that will be bundled into the iOS app.
   │   ├── mlc-app-config.json  # The app config JSON file.
   │   └── [optional model weights]
   └── lib
      ├── libmlc_llm.a          # A lightweight interface to interact with LLM, tokenizer, and TVM Unity runtime.
      ├── libmodel_iphone.a     # The compiled model lib.
      ├── libsentencepiece.a    # SentencePiece tokenizer
      ├── libtokenizers_cpp.a   # Huggingface tokenizer.
      └── libtvm_runtime.a      # TVM Unity runtime.


.. note::

   We leverage a local JIT cache to avoid repetitive compilation of the same input.
   However, sometimes it is helpful to force rebuild when we have a new compiler update
   or when something goes wrong with the cached library.
   You can do so by setting the environment variable ``MLC_JIT_POLICY=REDO``

   .. code:: bash

      MLC_JIT_POLICY=REDO mlc_llm package

.. _ios-bundle-model-weights:

Step 3. (Optional) Bundle model weights into the app
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, we download the model weights from Hugging Face when running the app.
**As an option,**, we bundle model weights into the app:
set the field ``"bundle_weight": true`` for any model you want to bundle weights
in ``MLCChat/mlc-package-config.json``, and run ``mlc_llm package`` again.
Below is an example:

.. code:: json

   {
      "device": "iphone",
      "model_list": [
         {
            "model": "HF://mlc-ai/gemma-2b-it-q4f16_1-MLC",
            "model_id": "gemma-2b-q4f16_1",
            "estimated_vram_bytes": 3000000000,
            "overrides": {
               "prefill_chunk_size": 128
            },
            "bundle_weight": true
         }
      ]
   }

The outcome of running ``mlc_llm package`` should be as follows:

.. code::

   dist
   ├── bundle
   │   ├── gemma-2b-q4f16_1   # The model weights that will be bundled into the app.
   │   └── mlc-app-config.json
   └── ...

.. _ios-build-app:

Step 4. Build iOS App
^^^^^^^^^^^^^^^^^^^^^

Open ``./ios/MLCChat/MLCChat.xcodeproj`` using Xcode. Note that you will need an
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

We can customize the models built in the iOS app by customizing `MLCChat/mlc-package-config.json <https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCChat/mlc-package-config.json>`_.
We introduce each field of the JSON file here.

Each entry in ``"model_list"`` of the JSON file has the following fields:

``model``
   (Required) The path to the MLC-converted model to be built into the app.

   It can be either a Hugging Face URL (e.g., ``"model": "HF://mlc-ai/phi-2-q4f16_1-MLC"```), or a path to a local model directory which contains converted model weights (e.g., ``"model": "../dist/gemma-2b-q4f16_1"``). Please check out :ref:`convert-weights-via-MLC` if you want to build local model into the app.

   *Note: the local path (if relative) is relative to the* ``ios/`` *directory.*

``model_id``
  (Required) A unique local identifier to identify the model.
  It can be an arbitrary one.

``estimated_vram_bytes``
   (Required) Estimated requirements of vRAM to run the model.

``bundle_weight``
   (Optional) A boolean flag indicating whether to bundle model weights into the app. See :ref:`ios-bundle-model-weights`.

``overrides``
   (Optional) A dictionary to override the default model context window size (to limit the KV cache size) and prefill chunk size (to limit the model temporary execution memory).
   Example:

   .. code:: json

      {
         "device": "iphone",
         "model_list": [
            {
                  "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
                  "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
                  "estimated_vram_bytes": 2960000000,
                  "overrides": {
                     "context_window_size": 512,
                     "prefill_chunk_size": 128
                  }
            }
         ]
      }

``model_lib``
   (Optional) A string specifying the system library prefix to use for the model.
   Usually this is used when you want to build multiple model variants with the same architecture into the app.
   **This field does not affect any app functionality.**
   The ``"model_lib_path_for_prepare_libs"`` introduced below is also related.
   Example:

   .. code:: json

      {
         "device": "iphone",
         "model_list": [
            {
                  "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
                  "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
                  "estimated_vram_bytes": 2960000000,
                  "model_lib": "gpt_neox_q4f16_1"
            }
         ]
      }


Besides ``model_list`` in ``MLCChat/mlc-package-config.json``,
you can also **optionally** specify a dictionary of ``"model_lib_path_for_prepare_libs"``,
**if you want to use model libraries that are manually compiled**.
The keys of this dictionary should be the ``model_lib`` that specified in model list,
and the values of this dictionary are the paths (absolute, or relative) to the manually compiled model libraries.
The model libraries specified in ``"model_lib_path_for_prepare_libs"`` will be built into the app when running ``mlc_llm package``.
Example:

.. code:: json

   {
      "device": "iphone",
      "model_list": [
         {
               "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
               "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
               "estimated_vram_bytes": 2960000000,
               "model_lib": "gpt_neox_q4f16_1"
         }
      ],
      "model_lib_path_for_prepare_libs": {
         "gpt_neox_q4f16_1": "../../dist/lib/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar"
      }
   }


Bring Your Own Model
--------------------

This section introduces how to build your own model into the iOS app.
We use the example of `NeuralHermes <https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B>`_ model, which a variant of Mistral model.

.. note::

  This section largely replicates :ref:`convert-weights-via-MLC`.
  See that page for more details. Note that the weights are shared across
  all platforms in MLC.

**Step 1. Clone from HF and convert_weight**

You can be under the mlc-llm repo, or your own working directory. Note that all platforms
can share the same compiled/quantized weights. See :ref:`compile-command-specification`
for specification of ``convert_weight``.

.. code:: shell

    # Create directory
    mkdir -p dist/models && cd dist/models
    # Clone HF weights
    git lfs install
    git clone https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B
    cd ../..
    # Convert weight
    mlc_llm convert_weight ./dist/models/NeuralHermes-2.5-Mistral-7B/ \
        --quantization q4f16_1 \
        -o dist/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC

**Step 2. Generate MLC Chat Config**

Use ``mlc_llm gen_config`` to generate ``mlc-chat-config.json`` and process tokenizers.
See :ref:`compile-command-specification` for specification of ``gen_config``.

.. code:: shell

    mlc_llm gen_config ./dist/models/NeuralHermes-2.5-Mistral-7B/ \
        --quantization q3f16_1 --conv-template neural_hermes_mistral \
        -o dist/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC

For the ``conv-template``, `conversation_template.py <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/conversation_template.py>`__
contains a full list of conversation templates that MLC provides.

If the model you are adding requires a new conversation template, you would need to add your own.
Follow `this PR <https://github.com/mlc-ai/mlc-llm/pull/2163>`__ as an example.
We look up the template to use with the ``conv_template`` field in ``mlc-chat-config.json``.

For more details, please see :ref:`configure-mlc-chat-json`.

**Step 3. Upload weights to HF**

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-mistral-weight-huggingface-repo
    cd my-mistral-weight-huggingface-repo
    cp path/to/mlc-llm/dist/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC/* .
    git add . && git commit -m "Add mistral model weights"
    git push origin main

After successfully following all steps, you should end up with a Huggingface repo similar to
`NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC <https://huggingface.co/mlc-ai/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC>`__,
which includes the converted/quantized weights, the ``mlc-chat-config.json``, and tokenizer files.


**Step 4. Register in Model List**

Finally, we add the model into the ``model_list`` of
`MLCChat/mlc-package-config.json <https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCChat/mlc-package-config.json>`_ by specifying the Hugging Face link as ``model``:

.. code:: json

   {
      "device": "iphone",
      "model_list": [
         {
               "model": "HF://mlc-ai/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC",
               "model_id": "Mistral-7B-Instruct-v0.2-q3f16_1",
               "estimated_vram_bytes": 3316000000,
         }
      ]
   }


Now, go through :ref:`ios-build-runtime-and-model-libraries` and :ref:`ios-build-app` again.
The app will use the ``NeuralHermes-Mistral`` model you just added.


Build Apps with MLC Swift API
-----------------------------

We also provide a Swift package that you can use to build
your own app. The package is located under ``ios/MLCSwift``.

- First, create ``mlc-package-config.json`` in your project folder.
  You do so by copying the files in MLCChat folder.
  Run ``mlc_llm package``.
  This will give us the necessary libraries under ``/path/to/project/dist``.
- Under "Build phases", add ``/path/to/project/dist/bundle`` this will copying
  this folder into your app to include bundled weights and configs.
- Add ``ios/MLCSwift`` package to your app in Xcode.
  Under "Frameworks, Libraries, and Embedded Content", click add package dependencies
  and add local package that points to ``ios/MLCSwift``.
- Finally, we need to add the libraries dependencies. Under build settings:

  - Add library search path ``/path/to/project/dist/lib``.
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

   func runExample() async {
      let engine = MLCEngine()
      let modelPath = "/path/to/model/weights"
      let modelLib = "model-lib-name"

      await engine.reload(modelPath: modelPath, modelLib: modelLib)

      // run chat completion as in OpenAI API style
      for await res in await engine.chat.completions.create(
            messages: [
               ChatCompletionMessage(
                  role: .user,
                  content: "What is the meaning of life?"
               )
            ]
      ) {
         print(res.choices[0].delta.content!.asText())
      }
   }

Checkout `MLCEngineExample <https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCEngineExample>`_
for a minimal starter example.
