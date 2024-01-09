.. _deploy-android:

Android App
===========

.. contents:: Table of Contents
   :local:
   :depth: 2

Demo App
--------

The demo APK below is built for Samsung S23 with Snapdragon 8 Gen 2 chip.

.. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
  :width: 135
  :target: https://github.com/mlc-ai/binary-mlc-llm-libs/raw/main/mlc-chat.apk

Prerequisite
------------

**Rust** (`install <https://www.rust-lang.org/tools/install>`__) is needed to cross-compile HuggingFace tokenizers to Android. Make sure rustc, cargo, and rustup are available in ``$PATH``.

**Android Studio** (`install <https://developer.android.com/studio>`__) with NDK and CMake. To install NDK and CMake, in the Android Studio welcome page, click "Projects → SDK Manager → SDK Tools". Set up the following environment variables:

- ``ANDROID_NDK`` so that ``$ANDROID_NDK/build/cmake/android.toolchain.cmake`` is available.
- ``TVM_NDK_CC`` that points to NDK's clang compiler.

.. code-block:: bash

  # Example on macOS
  ANDROID_NDK: $HOME/Library/Android/sdk/ndk/25.2.9519653
  TVM_NDK_CC: $ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang
  # Example on Windows
  ANDROID_NDK: $HOME/Library/Android/sdk/ndk/25.2.9519653
  TVM_NDK_CC: $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang

**JDK**, such as OpenJDK >= 17, to compile Java bindings of TVM Unity runtime. It could be installed via Homebrew on macOS, apt on Ubuntu or other package managers. Set up the following environment variable:

- ``JAVA_HOME`` so that Java is available in ``$JAVA_HOME/bin/java``.

**TVM Unity runtime** is placed under `3rdparty/tvm <https://github.com/mlc-ai/mlc-llm/tree/main/3rdparty>`__ in MLC LLM, so there is no need to install anything extra. Set up the following environment variable:

- ``TVM_HOME`` so that its headers are available under ``$TVM_HOME/include/tvm/runtime``.

(Optional) **TVM Unity compiler** Python package (:ref:`install <tvm-unity-prebuilt-package>` or :ref:`build from source <tvm-unity-build-from-source>`). It is *NOT* required if models are prebuilt, but to compile PyTorch models from HuggingFace in the following section, the compiler is a must-dependency.

.. note::
    ❗ Whenever using Python, it is highly recommended to use **conda** to manage an isolated Python environment to avoid missing dependencies, incompatible versions, and package conflicts.

Check if **environment variable** are properly set as the last check. One way to ensure this is to place them in ``$HOME/.zshrc``, ``$HOME/.bashrc`` or environment management tools.

.. code-block:: bash

  source $HOME/.cargo/env # Rust
  export ANDROID_NDK=...  # Android NDK toolchain
  export TVM_NDK_CC=...   # Android NDK clang
  export JAVA_HOME=...    # Java
  export TVM_HOME=...     # TVM Unity runtime

Compile PyTorch Models from HuggingFace
---------------------------------------

To deploy models on Android with reasonable performance, one has to cross-compile to and fully utilize mobile GPUs using TVM Unity. MLC provides a few pre-compiled models, or one could compile the models on their own.

**Cloning MLC LLM from GitHub**. Download MLC LLM via the following command:

.. code-block:: bash

  git clone --recursive https://github.com/mlc-ai/mlc-llm/
            ^^^^^^^^^^^
  cd ./mlc-llm/

.. note::
    ❗ The ``--recursive`` flag is necessary to download submodules like `3rdparty/tvm <https://github.com/mlc-ai/mlc-llm/tree/main/3rdparty>`__. If you see any file missing during compilation, please double check if git submodules are properly cloned.

**Download the PyTorch model** using Git Large File Storage (LFS), and by default, under ``./dist/models/``:

.. code-block:: bash

  MODEL_NAME=Llama-2-7b-chat-hf
  QUANTIZATION=q4f16_1

  git lfs install
  git clone https://huggingface.co/meta-llama/$MODEL_NAME \
            ./dist/models/

**Compile Android-capable models**. Install TVM Unity compiler as a Python package, and then compile the model for android using the following commands:

.. code-block:: bash

  # convert weights
  mlc_chat convert_weight ./dist/models/$MODEL_NAME/ --quantization $QUANTIZATION -o dist/$MODEL_NAME-$QUANTIZATION-MLC/

  # create mlc-chat-config.json
  mlc_chat gen_config ./dist/models/$MODEL_NAME/ --quantization $QUANTIZATION \
    --conv-template llama-2 --context-window-size 768 -o dist/${MODEL_NAME}-${QUANTIZATION}-MLC/

  # 2. compile: compile model library with specification in mlc-chat-config.json
  mlc_chat compile ./dist/${MODEL_NAME}-${QUANTIZATION}-MLC/mlc-chat-config.json \
      --device android -o ./dist/${MODEL_NAME}-${QUANTIZATION}-MLC/${MODEL_NAME}-${QUANTIZATION}-android.tar

This generates the directory ``./dist/$MODEL_NAME-$QUANTIZATION-MLC`` which contains the necessary components to run the model, as explained below.

**Expected output format**. By default models are placed under ``./dist/${MODEL_NAME}-${QUANTIZATION}-MLC``, and the result consists of 3 major components:

- Runtime configuration: It configures conversation templates including system prompts, repetition penalty, sampling including temperature and top-p probability, maximum sequence length, etc. It is usually named as ``mlc-chat-config.json`` alongside with tokenizer configurations.
- Model lib: The compiled library that uses mobile GPU. It is usually named as ``${MODEL_NAME}-${QUANTIZATION}-android.tar``, for example, ``Llama-2-7b-chat-hf-q4f16_1-android.tar``.
- Model weights: the model weights are sharded as ``params_shard_*.bin`` and the metadata is stored in ``ndarray-cache.json``

Create Android Project using Compiled Models
--------------------------------------------

The source code for MLC LLM is available under ``android/``, including scripts to build dependencies. Enter the directory first:

.. code-block:: bash

  cd ./android/library

**Build necessary dependencies.** Configure the list of models the app comes with using the JSON file ``app-config.json``. The ``model_libs`` field contains the list of model libraries that are bundled with and supported by the apk. The ``model_list`` field contains data for models that are not bundled with the apk, but downloaded from the Internet at run-time. By default, it is configured to use both Llama2-7B and RedPajama-3B models. To change the configuration, edit ``app-config.json``:

.. code-block:: bash

  vim ./src/main/assets/app-config.json

.. note::
    ❗ The compiled library path specificed using ``model_lib_path`` field in ``app-config.json`` expects it to be placed under ``./dist/`` under the project ``HOME`` directory. The ``model_lib`` field for each model is the system-lib-prefix set during ``mlc_chat compile`` which can be specified using ``--system-lib-prefix`` argument. By default, it is set to ``"${model_type}_${quantization}"`` e.g. ``llama_q4f16_1``. If the ``--system-lib-prefix`` argument is manually specified during ``mlc_chat compile``, the ``model_lib`` field in ``app-config.json`` should be updated accordingly.

Then bundle the android library ``${MODEL_NAME}-${QUANTIZATION}-android.tar`` compiled from ``mlc_chat compile`` in the previous steps, with TVM Unity's Java runtime by running the commands below:

.. code-block:: bash

  ./prepare_libs.sh

which generates the two files below:

.. code-block:: bash

  >>> find ./build/output -type f
  ./build/output/arm64-v8a/libtvm4j_runtime_packed.so
  ./build/output/tvm4j_core.jar

The model execution logic in mobile GPUs is incorporated into ``libtvm4j_runtime_packed.so``, while ``tvm4j_core.jar`` is a lightweight (~60 kb) `Java binding <https://tvm.apache.org/docs/reference/api/javadoc/>`_ to it.

**Build the Android app**. Open folder ``./android`` as an Android Studio Project. Connect your Android device to your machine. In the menu bar of Android Studio, click "Build → Make Project". Once the build is finished, click "Run → Run 'app'" and you will see the app launched on your phone.

.. note::
    ❗ This app cannot be run in an emulator and thus a physical phone is required, because MLC LLM needs an actual mobile GPU to meaningfully run at an accelerated speed.

Incorporate Model Weights
-------------------------

Instructions have been provided to build an Android App with MLC LLM in previous sections, but it requires run-time weight downloading from HuggingFace, as configured in `app-config.json` in previous steps under `model_url`. However, it could be desirable to bundle weights together into the app to avoid downloading over the network. In this section, we provide a simple ADB-based walkthrough that hopefully helps with further development.

**Generating APK**. Enter Android Studio, and click "Build → Generate Signed Bundle/APK" to build an APK for release. If it is the first time you generate an APK, you will need to create a key according to `the official guide from Android <https://developer.android.com/studio/publish/app-signing#generate-key>`_. This APK will be placed under ``android/app/release/app-release.apk``.

**Install ADB and USB debugging**. Enable "USB debugging" in the developer mode in your phone settings. In SDK manager, install `Android SDK Platform-Tools <https://developer.android.com/studio/releases/platform-tools>`_. Add the path to platform-tool path to the environment variable ``PATH``. Run the following commands, and if ADB is installed correctly, your phone will appear as a device:

.. code-block:: bash

  adb devices

**Install the APK and weights to your phone**. Run the commands below replacing ``${MODEL_NAME}`` and ``${QUANTIZATION}`` with the actual model name (e.g. Llama-2-7b-chat-hf) and quantization format (e.g. q4f16_1).

.. code-block:: bash

  adb install android/app/release/app-release.apk
  adb push dist/${MODEL_NAME}-${QUANTIZATION}-MLC/params /data/local/tmp/${MODEL_NAME}-${QUANTIZATION}-MLC/
  adb shell "mkdir -p /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/"
  adb shell "mv /data/local/tmp/${MODEL_NAME}-${QUANTIZATION}-MLC /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/"
