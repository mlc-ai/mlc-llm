# Build Instructions

1. Follow the instructions [here](https://github.com/mlc-ai/mlc-llm#building-from-source) to either build the model using a Hugging Face URL, or a local directory. If opting for a local directory, you can follow the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama) to get the original LLaMA weights in the HuggingFace format, and [here](https://github.com/lm-sys/FastChat#vicuna-weights) to get Vicuna weights.

    ```shell
    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm

    # From Hugging Face URL
    python3 build.py --hf-path databricks/dolly-v2-3b --quantization q3f16_0 --max-seq-len 768

    # From local directory
    python3 build.py --model path/to/vicuna-v1-7b --quantization q3f16_0 --max-seq-len 768

    # If the model path is in the form of `dist/models/model_name`,
    # we can simplify the build command to
    # python build.py --model model_name --quantization q3f16_0 --max-seq-len 768
    ```

2. Build the CLI.
    ```shell
    # Compile and build
    mkdir -p build
    cd build
    cmake ..
    make
    cd ..

    # Execute the CLI
    ./build/mlc_chat_cli --model vicuna-v1-7b
    ```
