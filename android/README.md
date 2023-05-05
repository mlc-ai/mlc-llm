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
    in build/config.cmake, set `USE_OPENCL` and `USE_LLVM` as ON
    ```shell
    make -j
    export TVM_HOME=$(pwd)
    export PYTHONPATH=$PYTHONPATH:$TVM_HOME/python
    ```

2. Get Model Weight.

    Currently we support LLaMA and Vicuna.

    1. Get the original LLaMA weights in the HuggingFace format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
    2. Use instructions [here](https://github.com/lm-sys/FastChat#vicuna-weights) to get vicuna weights.
    3. Create a soft link to the model path under dist/models.
        ```shell
        mkdir -p dist/models
        ln -s your_model_path dist/models/model_name
        # For example:
        # ln -s path/to/vicuna-v1-7b dist/models/vicuna-v1-7b

3. Build model to library.
    ```shell
    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm
    python3 build.py --model vicuna-v1-7b --dtype float16 --target android --quantization-mode int4 --quantization-sym --quantization-storage-nbit 32 --max-seq-len 768
    ```

4. Build libraries for Android app.
    ```shell
    cd android
    ./prepare_libs.sh
    ```

5. Download [Android Studio](https://developer.android.com/studio), and install Android APK and NDK either inside Android Studio (recommended) or separately. Connect your Android device to your machine. Use Android Studio to open folder `android/MLCChat` as the project. In the menu bar, click `Build - Make Project`. Once the build is finished, click `Run - Run 'app'`, and you will see the app launched on your phone.

<p align="center">
  <img src="../site/img/android/android-studio.png">
</p>
