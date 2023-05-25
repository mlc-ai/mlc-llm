# Introduction to MLC-LLM for Android

<p align="center">
  <img src="../site/gif/android-demo.gif" height="700">
</p>

We are excited to share that we have enabled the Android support for MLC-LLM. Checkout [the instruction page](https://mlc.ai/mlc-llm/#android) for instructions to download and install our Android app. Checkout the [announcing blog post](https://mlc.ai/blog/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices) for the technical details throughout our process of making MLC-LLM possible for Android.

## App Build Instructions

1. Install TVM Unity.
    We have some local changes to TVM Unity, so please try out the mlc/relax repo for now. We will migrate change back to TVM Unity soon.

    ```shell
    git clone https://github.com/mlc-ai/relax.git --recursive
    cd relax
    mkdir build
    cp cmake/config.cmake build
    ```

    In build/config.cmake, set `USE_OPENCL` and `USE_LLVM` as ON
    ```shell
    make -j
    export TVM_HOME=$(pwd)
    export PYTHONPATH=$PYTHONPATH:$TVM_HOME/python
    export TVM_NDK_CC=/path/to/android/ndk/clang
    For example
    export TVM_NDK_CC=/Users/me/Library/Android/sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang
    ```

2. Install [Apache Maven](https://maven.apache.org/download.cgi) for our Java dependency management. Run command `mnv --version` to verify that Maven is correctly installed.

3. Build TVM4j (Java Frontend for TVM Runtime).
    ```shell
    cd ${TVM_HOME}/jvm; mvn install -pl core -DskipTests -Dcheckstyle.skip=true
    ```

4. Follow the instructions [here](https://github.com/mlc-ai/mlc-llm#building-from-source) to either build the model using a Hugging Face URL, or a local directory. If opting for a local directory, you can follow the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama) to get the original LLaMA weights in the HuggingFace format, and [here](https://github.com/lm-sys/FastChat#vicuna-weights) to get Vicuna weights.

    ```shell
    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm

    # From local directory
    python3 build.py --model path/to/vicuna-v1-7b --quantization q4f16_0 --target android --max-seq-len 768

    # If the model path is `dist/models/vicuna-v1-7b`,
    # we can simplify the build command to
    # python build.py --model vicuna-v1-7b --quantization q4f16_0 --target android --max-seq-len 768
    ```

5. Build libraries for Android app.
    ```shell
    export ANDROID_NDK=/path/to/android/ndk
    For example
    export ANDROID_NDK=/Users/me/Library/Android/sdk/ndk/25.2.9519653
    cd android
    ./prepare_libs.sh
    ```

6. Download [Android Studio](https://developer.android.com/studio), and use Android Studio to open folder `android/MLCChat` as the project.
    1. Install Android SDK and NDK either inside Android Studio (recommended) or separately.
    2. Specify the Android SDK and NDK path in file `android/MLCChat/local.properties` (if it does not exist, create one):
        ```shell
        sdk.dir=/path/to/android/sdk
        ndk.dir=/path/to/android/ndk
        ```
        For example, a good `local.properties` can be:
        ```shell
        sdk.dir=/Users/me/Library/Android/sdk
        ndk.dir=/Users/me/Library/Android/sdk/ndk/25.2.9519653
        ```

7. Connect your Android device to your machine. In the menu bar of Android Studio, click `Build - Make Project`.

8. Once the build is finished, click `Run - Run 'app'`, and you will see the app launched on your phone.

<p align="center">
  <img src="../site/img/android/android-studio.png">
</p>

## Use Your Own Model Weights

By following the instructions above, the installed app will download weights from our pre-uploaded HuggingFace repository. If you do not want to download the weights from Internet and instead wish to use the weights you build, please follow the steps below.

* Step 1 - step 8: same as [section ”App Build Instructions”](#app-build-instructions).

* Step 9. In `Build - Generate Signed Bundle / APK`, build the project to an APK for release. If it is the first time you generate an APK, you will need to create a key. Please follow [the official guide from Android](https://developer.android.com/studio/publish/app-signing#generate-key) for more instructions on this. After generating the release APK, you will get the APK file `app-release.apk` under `android/MLCChat/app/release/`.

* Step 10. Enable “USB debugging” in the developer options your phone settings.

* Step 11. Install [Android SDK Platform-Tools](https://developer.android.com/studio/releases/platform-tools) for ADB (Android Debug Bridge). The platform tools will be already available under your Android SDK path if you have installed SDK (e.g., at `/path/to/android-sdk/platform-tools/`). Add the platform-tool path to your PATH environment. Run `adb devices` to verify that ADB is installed correctly your phone is listed as a device.

* Step 12. In command line, run
    ```shell
    adb install android/MLCChat/app/release/app-release.apk
    ```
    to install the APK to your phone. If it errors with message `adb: failed to install android/MLCChat/app/release/app-release.apk: Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE: Existing package ai.mlc.mlcchat signatures do not match newer version; ignoring!]`, please uninstall the existing app and try `adb install` again.

* Step 13. Push the tokenizer and model weights to your phone through ADB.
    ```shell
    adb push dist/models/vicuna-v1-7b/tokenizer.model /data/local/tmp/vicuna-v1-7b/tokenizer.model
    adb push dist/vicuna-v1-7b/float16/params /data/local/tmp/vicuna-v1-7b/params
    adb shell "mkdir -p /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/Download/"
    adb shell "mv /data/local/tmp/vicuna-v1-7b /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/Download/vicuna-v1-7b"
    ```

* Step 14. Everything is ready. Launch the MLCChat on your phone and you will be able to use the app with your own weights. You will find that no weight download is needed.
