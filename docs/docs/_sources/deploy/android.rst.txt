.. _deploy-android:

Android App
===========

.. contents:: Table of Contents
   :local:
   :depth: 2


The MLC LLM Android package can be installed in two ways: either from the pre-built package or by building it from source. If you're an Android user interested in trying out models, the pre-built package is the way to go. On the other hand, if you're a developer aiming to incorporate new features into the package, building the Android package from source is necessary.

Use Pre-built Android Package
-----------------------------

The MLC LLM Android app is free and available for download and can be tried out by simply clicking the button below:

.. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
   :width: 135
   :target: https://github.com/mlc-ai/binary-mlc-llm-libs/raw/main/mlc-chat.apk


Build Android Package from Source
---------------------------------

If you're a developer looking to integrate new functionality or support different model architectures in the Android Package, you may need to build it from source. To do so, please follow the instructions provided below on building the Android package from source. 
Before you begin, clone this repository and initialize the submodules, as well as install and initalize rustup.

App Build Instructions
^^^^^^^^^^^^^^^^^^^^^^

1. Follow the instructions in :doc:`/compilation/compile_models` to
   either build the model using a Hugging Face URL, or a local
   directory. For Vicuna weights, please follow our :doc:`/compilation/get-vicuna-weight` tutorial.

   .. code:: shell

      # From mlc-llm project directory
      python3 build.py --model path/to/vicuna-v1-7b --quantization q4f16_1 --target android --max-seq-len 768

      # If the model path is `dist/models/vicuna-v1-7b`,
      # we can simplify the build command to
      # python build.py --model vicuna-v1-7b --quantization q4f16_1 --target android --max-seq-len 768

2. Configure the ``model_libs`` in ``android/MLCChat/app/src/main/assets/app-config.json``:
   
   If there is a ``local_id`` in ``model_libs`` list, then there should be a ``local_id-target.tar`` in ``dist/local_id`` compiled in Step 1.

   For example, if you have ``vicuna-v1-7b-q4f16_1`` in the ``model_libs``:

   .. code:: bash

      cat android/MLCChat/app/src/main/assets/app-config.json
      # "model_libs": [
      #   ...
      #   "vicuna-v1-7b-q4f16_1",
      #   ...
      # ],
   
   then there should be a ``dist/vicuna-v1-7b-q4f16_1/vicuna-v1-7b-q4f16_1-android.tar`` file:

   .. code:: bash

      ls dist/vicuna-v1-7b-q4f16_1
      # ...
      # vicuna-v1-7b-q4f16_1-android.tar,
      # ...

3. Download `Android Studio <https://developer.android.com/studio>`_, install the ``NDK`` and ``CMake`` via `SDK Manager <https://developer.android.com/studio/projects/install-ndk>`_.

4. Setup ``ANDROID_NDK`` and ``TVM_NDK_CC`` environment variable to the installed NDK compiler path:

   .. code:: bash

      # replace the /path/to/android/ndk to your NDK path
      # e.g. for MacOS: export ANDROID_NDK=/Users/me/Library/Android/sdk/ndk/25.2.9519653
      # e.g. for Linux: export ANDROID_NDK=/home/user/Downloads/android-studio/plugins/android-ndk
      export ANDROID_NDK=/path/to/android/ndk

      # replace the /path/to/android/ndk/clang to your NDK compiler path
      # e.g. for MacOS: export TVM_NDK_CC=/Users/me/Library/Android/sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang
      # e.g. for Linux: export TVM_NDK_CC=/home/user/Downloads/android-studio/plugins/c-clangd/bin/clang/linux/x64/clangd
      export TVM_NDK_CC=/path/to/android/ndk/clang

5. Setup ``JAVA_HOME`` environment variable:

   .. code:: bash

      # replace the /path/to/jdk to your JDK path
      # e.g. for MacOS: export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
      # e.g. for Linux: export JAVA_HOME=/home/user/Downloads/jdk-20
      export JAVA_HOME=/path/to/jdk

6. Build the libs for Android app and then copy the built files to the ``android/MLCChat/app/src/main/src/libs``:

   .. code:: bash

      cd android && ./prepare_libs.sh
      # If building successfully, there should be a `tvm4j_core.jar` and `arm64-v8a/libtvm4j_runtime_packed.so` in `build/output` dir.
      ls ./build/output
      # tvm4j_core.jar
      # arm64-v8a
      ls ./build/output/arm64-v8a
      # libtvm4j_runtime_packed.so
      cp -a build/output/. MLCChat/app/src/main/libs

7.  Open folder ``android/MLCChat`` as the project with Android Studio. And connect your Android device to your machine. In the menu bar of Android Studio, click ``Build - Make Project``.

8. Once the build is finished, click ``Run - Run 'app'``, and you will see the app launched on your phone.

.. image:: https://github.com/mlc-ai/mlc-llm/raw/main/site/img/android/android-studio.png

Use Your Own Model Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^

By following the instructions above, the installed app will download
weights from our pre-uploaded HuggingFace repository. If you do not want
to download the weights from Internet and instead wish to use the
weights you build, please follow the steps below.

-  Step 1 - step 8: same as `section ”App Build
   Instructions” <#app-build-instructions>`__.

-  Step 9. In ``Build - Generate Signed Bundle / APK``, build the
   project to an APK for release. If it is the first time you generate
   an APK, you will need to create a key. Please follow `the official
   guide from
   Android <https://developer.android.com/studio/publish/app-signing#generate-key>`__
   for more instructions on this. After generating the release APK, you
   will get the APK file ``app-release.apk`` under
   ``android/MLCChat/app/release/``.

-  Step 10. Enable “USB debugging” in the developer options your phone
   settings.

-  Step 11. Install `Android SDK
   Platform-Tools <https://developer.android.com/studio/releases/platform-tools>`_
   for ADB (Android Debug Bridge) via `SDK Manager <https://developer.android.com/studio/projects/install-ndk>`_. The platform tools will be already
   available under your Android SDK path if you have installed SDK
   (e.g., at ``/path/to/android-sdk/platform-tools/``). Add the
   platform-tool path to your ``PATH`` environment. Run ``adb devices`` to
   verify that ADB is installed correctly your phone is listed as a
   device.

-  Step 12. In command line, run the following command to install APK to your phone:

  .. code:: bash

     adb install android/MLCChat/app/release/app-release.apk


  .. note::

   If it errors with message

   .. code:: bash

     adb: failed to install android/MLCChat/app/release/app-release.apk: Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE: Existing package ai.mlc.mlcchat signatures do not match newer version; ignoring!]

   please uninstall the existing app and try ``adb install`` again.

-  Step 13. Push the model dir to your phone through
   ADB.

    .. code:: bash

      adb push dist/models/vicuna-v1-7b/ /data/local/tmp/vicuna-v1-7b/
      adb shell "mkdir -p /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/"
      adb shell "mv /data/local/tmp/vicuna-v1-7b /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/vicuna-v1-7b"

-  Step 14. Everything is ready. Launch the MLCChat on your phone and
   you will be able to use the app with your own weights. You will find
   that no weight download is needed.
