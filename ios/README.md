# Build Instructions

Note: You will need Apple Developer Account to build iOS App locally.

1. Install TVM Unity. 
    We have some local changes to TVM Unity, so please try out the mlc/relax repo for now. We will migrate change back to TVM Unity soon.

    ```shell
    git clone https://github.com/mlc-ai/relax.git --recursive
    cd relax
    mkdir build
    cp cmake/config.cmake build
    ```
    
    In build/config.cmake, set `USE_METAL` and `USE_LLVM` as ON
    ```shell
    make -j
    export TVM_HOME=$(pwd)
    export PYTHONPATH=$PYTHONPATH:$TVM_HOME/python
    ```


2. Follow the instructions [here](https://github.com/mlc-ai/mlc-llm#building-from-source) to either build the model using a Hugging Face URL, or a local directory. If opting for a local directory, you can follow the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama) to get the original LLaMA weights in the HuggingFace format, and [here](https://github.com/lm-sys/FastChat#vicuna-weights) to get Vicuna weights.

    ```shell
    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm

    # From Hugging Face URL
    python3 build.py --hf-path databricks/dolly-v2-3b --quantization q3f16_0 --target iphone --max-seq-len 768

    # From local directory
    python3 build.py --model-path path/to/vicuna-v1-7b --quantization q3f16_0 --target iphone --max-seq-len 768
    ```

3. Prepare lib and params
    ```shell
    cd ios
    ./prepare_libs.sh
    ./prepare_params.sh
    ```

4. Use Xcode to open MLCChat.xcodeproj, click on Automatically manage signing, and then click Product - Run. 
If you find the error "Failed to register bundle identifier", change the Bundle Identifier to any other name.
