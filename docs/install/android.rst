Build Android Package
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

.. image:: https://github.com/mlc-ai/mlc-llm/raw/main/site/gif/android-demo.gif
  :width: 400
  :align: center


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

App Build Instructions
^^^^^^^^^^^^^^^^^^^^^^

1. Install TVM Unity by following :ref:`Build TVM Unity from Source <tvm-unity-build-from-source>` tutorial.

   Note that our pre-built wheels do not support OpenCL, and you need to built TVM-Unity 
   from source and set ``USE_OPENCL`` as ``ON``.

2. Setup ``TVM_NDK_CC`` environment variable to NDK compiler path:
   
   .. code:: bash

      # replace the /path/to/android/ndk/clang to your NDK compiler path
      # e.g. export TVM_NDK_CC=/Users/me/Library/Android/sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang
      export TVM_NDK_CC=/path/to/android/ndk/clang

3. Install `Apache Maven <https://maven.apache.org/download.cgi>`__ for
   our Java dependency management. Run command ``mvn --version`` to
   verify that Maven is correctly installed.

4. Build TVM4j (Java Frontend for TVM Runtime) under the TVM Unity directory.

   .. code:: shell

      cd jvm; mvn install -pl core -DskipTests -Dcheckstyle.skip=true

5. Follow the instructions in :doc:`/tutorials/compilation/model_compilation_walkthrough` to
   either build the model using a Hugging Face URL, or a local
   directory. For Vicuna weights, please follow our :doc:`/tutorials/compilation/get-vicuna-weight` tutorial.

   .. code:: shell

      # From mlc-llm project directory
      python3 build.py --model path/to/vicuna-v1-7b --quantization q4f16_0 --target android --max-seq-len 768

      # If the model path is `dist/models/vicuna-v1-7b`,
      # we can simplify the build command to
      # python build.py --model vicuna-v1-7b --quantization q4f16_0 --target android --max-seq-len 768

6. Build libraries for Android app.

   .. code:: shell

      export ANDROID_NDK=/path/to/android/ndk
      For example
      export ANDROID_NDK=/Users/me/Library/Android/sdk/ndk/25.2.9519653
      cd android && ./prepare_libs.sh

7. Download `Android Studio <https://developer.android.com/studio>`__,
   and use Android Studio to open folder ``android/MLCChat`` as the
   project.

   1. Install Android SDK and NDK either inside Android Studio
      (recommended) or separately.

   2. Specify the Android SDK and NDK path in file
      ``android/MLCChat/local.properties`` (if it does not exist, create
      one):

      .. code:: shell

         sdk.dir=/path/to/android/sdk
         ndk.dir=/path/to/android/ndk

      For example, a good ``local.properties`` can be:

      .. code:: shell

         sdk.dir=/Users/me/Library/Android/sdk
         ndk.dir=/Users/me/Library/Android/sdk/ndk/25.2.9519653

8. Connect your Android device to your machine. In the menu bar of
   Android Studio, click ``Build - Make Project``.

9.  Once the build is finished, click ``Run - Run 'app'``, and you will see the app launched on your phone.

.. image:: https://github.com/mlc-ai/mlc-llm/raw/main/site/img/android/android-studio.png

Use Your Own Model Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^

By following the instructions above, the installed app will download
weights from our pre-uploaded HuggingFace repository. If you do not want
to download the weights from Internet and instead wish to use the
weights you build, please follow the steps below.

-  Step 1 - step 9: same as `section ”App Build
   Instructions” <#app-build-instructions>`__.

-  Step 10. In ``Build - Generate Signed Bundle / APK``, build the
   project to an APK for release. If it is the first time you generate
   an APK, you will need to create a key. Please follow `the official
   guide from
   Android <https://developer.android.com/studio/publish/app-signing#generate-key>`__
   for more instructions on this. After generating the release APK, you
   will get the APK file ``app-release.apk`` under
   ``android/MLCChat/app/release/``.

-  Step 11. Enable “USB debugging” in the developer options your phone
   settings.

-  Step 12. Install `Android SDK
   Platform-Tools <https://developer.android.com/studio/releases/platform-tools>`__
   for ADB (Android Debug Bridge). The platform tools will be already
   available under your Android SDK path if you have installed SDK
   (e.g., at ``/path/to/android-sdk/platform-tools/``). Add the
   platform-tool path to your PATH environment. Run ``adb devices`` to
   verify that ADB is installed correctly your phone is listed as a
   device.

-  Step 13. In command line, run the following command to install APK to your phone:

  .. code:: bash

     adb install android/MLCChat/app/release/app-release.apk


  .. note::

   If it errors with message

   .. code:: bash

     adb: failed to install android/MLCChat/app/release/app-release.apk: Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE: Existing package ai.mlc.mlcchat signatures do not match newer version; ignoring!]

   please uninstall the existing app and try ``adb install`` again.

-  Step 14. Push the tokenizer and model weights to your phone through
   ADB.
   
    .. code:: bash

      adb push dist/models/vicuna-v1-7b/tokenizer.model /data/local/tmp/vicuna-v1-7b/tokenizer.model
      adb push dist/vicuna-v1-7b/float16/params /data/local/tmp/vicuna-v1-7b/params
      adb shell "mkdir -p /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/Download/"
      adb shell "mv /data/local/tmp/vicuna-v1-7b /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/Download/vicuna-v1-7b"

-  Step 15. Everything is ready. Launch the MLCChat on your phone and
   you will be able to use the app with your own weights. You will find
   that no weight download is needed.
