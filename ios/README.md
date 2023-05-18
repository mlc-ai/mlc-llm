# Build Instructions

We will use `vicuna-v1-7b` as an example to demonstrate the quantization and cross-compilation workflow for LLM on iOS devices. The same procedure can be applied to other models as well, such as Dolly and RedPajama.

## Step 1. Install TVM Unity

First, we need to install the TVM Unity compiler for machine learning compilation. You can choose to install it from pip or build it from source.

**Option 1. Install using pip**

You can install a nightly build of TVM Unity for macOS from the following location: [https://mlc.ai/wheels](https://mlc.ai/wheels). Create a fresh python3 environment and run the following command to install it:

```bash
pip3 install mlc-ai-nightly -f https://mlc.ai/wheels
```

Next, determine the installation location of TVM by executing the following command and set an environment variable `TVM_HOME` for MLC LLM to use:

```bash
python3 -c "import tvm, os; print(os.path.dirname(tvm.__file__))"
export TVM_HOME=... # The output from the previous command
```

**Option 2. Build from source**

To build TVM Unity from source, follow these steps:

```bash
# Download TVM Unity from GitHub
git clone https://github.com/mlc-ai/relax.git --recursive tvm-unity && cd tvm-unity
# Configure CMake build
mkdir build && cd build
cp ../cmake/config.cmake .
echo set\(USE_METAL ON\) >> config.cmake
echo set\(USE_LLVM ON\) >> config.cmake
echo set\(CMAKE_BUILD_TYPE Release\) >> config.cmake
echo set\(CMAKE_EXPORT_COMPILE_COMMANDS ON\) >> config.cmake
# Build TVM Unity
cmake .. && make -j$(nproc) && cd ..
```

Once TVM Unity is successfully built, set two environment variables so that the python package of TVM Unity can be found:

```bash
export TVM_HOME=$(pwd) # make sure the file exists: $TVM_HOME/build/libtvm.so
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

## Step 2. Download LLM weights

Download the LLM weights for Vicuna-7b by following the instructions from [LMSys](https://github.com/lm-sys/FastChat#vicuna-weights). Place the downloaded weights under the `./dist/models/vicuna-v1-7b` directory within the `mlc-llm` folder. The contents of the folder should include the following files:

```
>>> ls ./dist/models/vicuna-v1-7b
config.json
pytorch_model.bin.index.json
pytorch_model-00001-of-00002.bin
pytorch_model-00002-of-00002.bin
tokenizer.model
tokenizer_config.json
special_tokens_map.json
```

## Step 3. Quantize and cross-compile LLMs with TVM Unity

To quantize and cross-compile the LLMs to iPhone/iPad targets, use the provided `build.py` script:

```bash

python3 build.py                    \
    --model /path/to/vicuna-v1-7b   \ 
    --quantization q3f16_0          \
    --target iphone                 \
    --max-seq-len 768

# If the model path is `./dist/models/vicuna-v1-7b`,
# we can simplify the build command to
python3 build.py            \
    --model vicuna-v1-7b    \
    --quantization q3f16_0  \
    --target iphone         \
    --max-seq-len 768
```

By default, the compiled artifact will be located under `./dist/vicuna-v1-7b-q3f16_0`. The folder includes:
- The static library for LLM computation: `vicuna-v1-7b-q3f16_0-iphone.a`
- Assets, including weight shards under `params/`
- Tokenizer metadata

## Step 4. Build auxiliary components

**Tokenizer and runtime**

In addition to the model itself, a lightweight runtime and tokenizer are required to actually run the LLM. You can build and organize these components by following these steps:

```bash
cd ./ios
vim ./prepare_libs.sh # Update MODEL, QUANTIZATION, and other variables
./prepare_libs.sh
```

This will create a `./build` folder that contains the following files:

```
>>> ls ./build/lib/
libmlc_llm.a         # The lightweight interface to interact with LLM, tokenizer, and TVM Unity runtime
libmodel_iphone.a    # The compiled LLM
libsentencepiece.a   # SentencePiece tokenizer
libtokenizers_cpp.a  # Huggingface tokenizer
libtvm_runtime.a     # TVM Unity runtime
```

**Collect assets**

To organize the assets, execute the following script within the `./ios` directory:

```bash
cd ./ios
vim ./prepare_params.sh # Update MODEL, QUANTIZATION, TOKENIZER, and other variables
./prepare_params.sh
```

The outcome should be as follows:

```
>>> ls ./dist/
params/              # Parameter shards
tokenizer.json       # Tokenizer metadata
```

## Step 5. Build iOS App

Open `./ios/MLCChat.xcodeproj` using Xcode. Note that you will need an Apple Developer Account to use Xcode, and you may be prompted to use your own developer team credential and product bundle identifier.

To use a specific model, edit `./ios/MLCChat/LLMChat.mm` to configure the following settings properly:
1. Model name using the `model` variable
2. Conversation template using the `conv_template` variable
3. Tokenizer name using the `tokenizer_path` variable
4. Other settings including max sequence length, temperature, etc.

Ensure that all the necessary dependencies and configurations are correctly set up in the Xcode project.

Once you have made the necessary changes, build the iOS app using Xcode. Make sure to select a target device or simulator for the build.

After a successful build, you can run the iOS app on your device or simulator to use the LLM model for text generation and processing.
