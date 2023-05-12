# Build Instructions

Note: You will need Apple Developer Account to build iOS App locally.

1. Install TVM Unity. 
    We have some local changes to TVM Unity, so please try out the mlc/relax repo for now. We will migrate change back to TVM Unity soon.

    ```
    git clone https://github.com/mlc-ai/relax.git --recursive
    cd relax
    mkdir build
    cp cmake/config.cmake build
    ```
    in build/config.cmake, set `USE_METAL` and `USE_LLVM` as ON
    ```
    make -j
    export TVM_HOME=$(pwd)
    export PYTHONPATH=$PYTHONPATH:$TVM_HOME/python
    ```

2. Get Model Weight

    Currently we support LLaMA and Vicuna.

    1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
    2. Use instructions [here](https://github.com/lm-sys/FastChat#vicuna-weights) to get vicuna weights.
    3. Create a soft link to the model path under dist/models
        ```shell
        mkdir -p dist/models
        ln -s your_model_path dist/models/model_name
        # For example:
        # ln -s path/to/vicuna-v1-7b dist/models/vicuna-v1-7b


3. build model to library
    ```
    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm
    python3 build.py --model vicuna-v1-7b --quantization q3f16_0 --target iphone --max-seq-len 768
    ```
4. Prepare lib and params
    ```
    cd ios
    ./prepare_libs.sh
    ./prepare_params.sh
    ```


5. use Xcode to open MLCChat.xcodeproj, click on Automatically manage signing, and then click Product - Run. 
If you find the error "Failed to register bundle identifier", change the Bundle Identifier to any other name.
