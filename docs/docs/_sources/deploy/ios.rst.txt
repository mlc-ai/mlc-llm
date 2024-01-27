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

``required_vram_bytes``
   (Required) Estimated requirements of VRAM to run the model.

``model_lib_path_for_prepare_libs``
   (Required) List of paths to the model libraries in the app (respective ``.tar`` file in the ``binary-mlc-llm-libs``
   repo, relative path in the ``dist`` artifact folder or full path to the library). Only used while running
   ``prepare_libs.sh`` to determine which model library to use during runtime. Useful when selecting a library with
   different settings (e.g. ``prefill_chunk_size``, ``context_window_size``, and ``sliding_window_size``).

Additionally, the app prepackages the models under ``./ios/dist``.
This built-in list can be controlled by editing ``prepare_params.sh``.
You can package new prebuilt models or compiled models by changing the above fields and then repeating the steps above.


Bring Your Own Model Variant
----------------------------

In cases where the model you are adding is simply a variant of an existing
model, we only need to convert weights and reuse existing model library. For instance:

- Adding ``NeuralHermes`` when MLC already supports the ``Mistral`` architecture


In this section, we walk you through adding ``NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC`` to the MLC iOS app.
According to the model's ``config.json`` on `its Huggingface repo <https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B/blob/main/config.json>`_,
it reuses the Mistral model architecture.

.. note:: 

  This section largely replicates :ref:`convert-weights-via-MLC`.
  See that page for more details. Note that the weights are shared across
  all platforms in MLC.

**Step 1 Clone from HF and convert_weight**

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
    mlc_chat convert_weight ./dist/models/NeuralHermes-2.5-Mistral-7B/ \
        --quantization q4f16_1 \
        -o dist/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC

**Step 2 Generate MLC Chat Config**

Use ``mlc_chat gen_config`` to generate ``mlc-chat-config.json`` and process tokenizers.
See :ref:`compile-command-specification` for specification of ``gen_config``.

.. code:: shell

    mlc_chat gen_config ./dist/models/NeuralHermes-2.5-Mistral-7B/ \
        --quantization q3f16_1 --conv-template neural_hermes_mistral \
        -o dist/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC

For the ``conv-template``, `conv_template.cc <https://github.com/mlc-ai/mlc-llm/blob/main/cpp/conv_templates.cc>`__
contains a full list of conversation templates that MLC provides.

If the model you are adding requires a new conversation template, you would need to add your own. 
Follow `this PR <https://github.com/mlc-ai/mlc-llm/pull/1402>`__ as an example. 
We look up the template to use with the ``conv_template`` field in ``mlc-chat-config.json``.

For more details, please see :ref:`configure-mlc-chat-json`.

**Step 3 Upload weights to HF**

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


**Step 4 Register as a ModelRecord**

Finally, we modify the code snippet for
`app-config.json <https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCChat/app-config.json>`__
pasted above.

We simply specify the Huggingface link as ``model_url``, while reusing the ``model_lib`` for 
``Mistral-7B``.

.. code:: javascript
   
   "model_list": [
      // Other records here omitted...
      {
         // Substitute model_url with the one you created `my-huggingface-account/my-mistral-weight-huggingface-repo`
         "model_url": "https://huggingface.co/mlc-ai/NeuralHermes-2.5-Mistral-7B-q3f16_1-MLC",
         "model_id": "Mistral-7B-Instruct-v0.2-q3f16_1",
         "model_lib": "mistral_q3f16_1",
         "model_lib_path": "lib/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q3f16_1-iphone.tar",
         "estimated_vram_bytes": 3316000000
      }
   ]


Now, the app will use the ``NeuralHermes-Mistral`` model you just added.


Bring Your Own Model Library
----------------------------

A model library is specified by:

 - The model architecture (e.g. ``mistral``, ``phi-msft``)
 - Quantization Scheme (e.g. ``q3f16_1``, ``q0f32``)
 - Metadata (e.g. ``context_window_size``, ``sliding_window_size``, ``prefill_chunk_size``), which affects memory planning
 - Platform (e.g. ``cuda``, ``webgpu``, ``iphone``, ``android``)

In cases where the model you want to run is not compatible with the provided MLC
prebuilt model libraries (e.g. having a different quantization, a different
metadata spec, or even a different model architecture), you need to build your
own model library.

In this section, we walk you through adding ``phi-2`` to the iOS app.

This section largely replicates :ref:`compile-model-libraries`. See that page for
more details, specifically the ``iOS`` option.

**Step 0. Install dependencies**

To compile model libraries for iOS, you need to :ref:`build mlc_chat from source <mlcchat_build_from_source>`.

**Step 1. Clone from HF and convert_weight**

You can be under the mlc-llm repo, or your own working directory. Note that all platforms
can share the same compiled/quantized weights.

.. code:: shell

    # Create directory
    mkdir -p dist/models && cd dist/models
    # Clone HF weights
    git lfs install
    git clone https://huggingface.co/microsoft/phi-2
    cd ../..
    # Convert weight
    mlc_chat convert_weight ./dist/models/phi-2/ \
        --quantization q4f16_1 \
        -o dist/phi-2-q4f16_1-MLC

**Step 2. Generate mlc-chat-config and compile**

A model library is specified by:

 - The model architecture (e.g. ``mistral``, ``phi-msft``)
 - Quantization Scheme (e.g. ``q3f16_1``, ``q0f32``)
 - Metadata (e.g. ``context_window_size``, ``sliding_window_size``, ``prefill_chunk_size``), which affects memory planning
 - Platform (e.g. ``cuda``, ``webgpu``, ``iphone``, ``android``)

All these knobs are specified in ``mlc-chat-config.json`` generated by ``gen_config``.

.. code:: shell

    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
    mlc_chat gen_config ./dist/models/phi-2/ \
        --quantization q4f16_1 --conv-template phi-2 \
        -o dist/phi-2-q4f16_1-MLC/
    # 2. compile: compile model library with specification in mlc-chat-config.json
    mlc_chat compile ./dist/phi-2-q4f16_1-MLC/mlc-chat-config.json \
        --device iphone -o dist/libs/phi-2-q4f16_1-iphone.tar

.. note::
    When compiling larger models like ``Llama-2-7B``, you may want to add a lower chunk size
    while prefilling prompts ``--prefill_chunk_size 128`` or even lower ``context_window_size``\
    to decrease memory usage. Otherwise, during runtime, you may run out of memory.


**Step 3. Distribute model library and model weights**

After following the steps above, you should end up with:

.. code:: shell

    ~/mlc-llm > ls dist/libs
      phi-2-q4f16_1-iphone.tar  # ===> the model library

    ~/mlc-llm > ls dist/phi-2-q4f16_1-MLC
      mlc-chat-config.json                             # ===> the chat config
      ndarray-cache.json                               # ===> the model weight info
      params_shard_0.bin                               # ===> the model weights
      params_shard_1.bin
      ...
      tokenizer.json                                   # ===> the tokenizer files
      tokenizer_config.json

Upload the ``phi-2-q4f16_1-iphone.tar`` to a github repository (for us,
it is in `binary-mlc-llm-libs <https://github.com/mlc-ai/binary-mlc-llm-libs>`__). Then
upload the weights ``phi-2-q4f16_1-MLC`` to a Huggingface repo:

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-phi-weight-huggingface-repo
    cd my-phi-weight-huggingface-repo
    cp path/to/mlc-llm/dist/phi-2-q4f16_1-MLC/* .
    git add . && git commit -m "Add phi-2 model weights"
    git push origin main

This would result in something like `phi-2-q4f16_1-MLC
<https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC/tree/main>`_.


**Step 4. Calculate estimated VRAM usage**

Given the compiled library, it is possible to calculate an upper bound for the VRAM
usage during runtime. This useful to better understand if a model is able to fit particular
hardware. We can calculate this estimate using the following command:

.. code:: shell

    ~/mlc-llm > python -m mlc_chat.cli.model_metadata ./dist/libs/phi-2-q4f16_1-iphone.tar \
      > --memory-only --mlc-chat-config ./dist/phi-2-q4f16_1-MLC/mlc-chat-config.json
      INFO model_metadata.py:90: Total memory usage: 3042.96 MB (Parameters: 1492.45 MB. KVCache: 640.00 MB. Temporary buffer: 910.51 MB)
      INFO model_metadata.py:99: To reduce memory usage, tweak `prefill_chunk_size`, `context_window_size` and `sliding_window_size`


**Step 5. Register as a ModelRecord**

Finally, we update the code snippet for
`app-config.json <https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCChat/app-config.json>`__
pasted above.

We simply specify the Huggingface link as ``model_url``, while using the new ``model_lib`` for 
``phi-2``. Regarding the field ``estimated_vram_bytes``, we can use the output of the last step
rounded up to MB.

.. code:: javascript
   
   "model_list": [
      // Other records here omitted...
      {
         // Substitute model_url with the one you created `my-huggingface-account/my-phi-weight-huggingface-repo`
         "model_url": "https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC",
         "model_id": "phi-2-q4f16_1",
         "model_lib": "phi_msft_q4f16_1",
         "model_lib_path": "lib/phi-2/phi-2-q4f16_1-iphone.tar",
         "estimated_vram_bytes": 3043000000
      }
   ]


Now, the app will use the ``phi-2`` model library you just added.


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
