---
layout: default
title: Home
notitle: true
---

# MLC LLM

MLC LLM is a universal solution that allows any language model to be deployed natively on a diverse set of hardware backends and native applications, plus a productive framework for everyone to further optimize model performance for their own use cases.
**Everything runs locally  with no server support and accelerated with local GPUs on your phone and laptop**.
Check out our [GitHub repository](https://github.com/mlc-ai/mlc-llm) to see how we did it. You can also read through instructions below for trying out demos.

<p align="center">
<img src="gif/ios-demo.gif" height="700">
</p>

## Try it out

This section contains the instructions to run large-language models and chatbot natively on your environment.

- [iPhone](#iphone)
- [Android](#android)
- [Windows Linux Mac](#windows-linux-mac)
- [Web browser](#web-browser)

### iPhone

Try out this [TestFlight page](https://testflight.apple.com/join/57zd7oxa) (limited to the first 9000 users) to install and use our example iOS chat app built for iPhone.
Vicuna-7B takes 4GB of RAM and RedPajama-3B takes 2.2GB to run. Considering the iOS and other running applications, we will need a recent iPhone with 6GB for Vicuna-7B or 4GB for RedPajama-3B to run the app. The application is only tested on iPhone 14 Pro Max, iPhone 14 Pro and iPhone 12 Pro.

To build the iOS app from source, You can also check out our [GitHub repo](https://github.com/mlc-ai/mlc-llm).

Note: The text generation speed on the iOS app can be unstable from time to time. It might run slow in the beginning and recover to a normal speed then.

### Android

Download the APK file [here](https://github.com/mlc-ai/binary-mlc-llm-libs/raw/main/mlc-chat.apk) and install on your phone. You can then start a chat with LLM. When you first open the app, parameters need to be downloaded and the loading process could be slow. In future run, the parameters will be loaded from cache (which is fast) and you can use the app offline. Our current demo relies on OpenCL support on the phone and takes about 6GB of RAM, if you have a phone with the latest Snapdragon chip, you can try out out demo.

We tested our demo on Samsung Galaxy S23. It does not yet work on Google Pixel due to limited OpenCL support. We will continue to bring support and welcome contributions from the open source community. You can also check out our [GitHub repo](https://github.com/mlc-ai/mlc-llm/tree/main/android) to build the Android app from source.

Check out our [blog post](https://mlc.ai/blog/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices) for the technical details throughout our process of making MLC-LLM possible for Android.

<p align="center">
<img src="gif/android-demo.gif" height="700">
</p>

### Windows Linux Mac

We provide a CLI (command-line interface) app to chat with the bot in your terminal. Before installing
the CLI app, we should install some dependencies first.
1. We use Conda to manage our app, so we need to install a version of conda. We can install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
2. On Windows and Linux, the chatbot application runs on GPU via the Vulkan platform. For Windows and Linux users,
please install the latest [Vulkan driver](https://developer.nvidia.com/vulkan-driver). For NVIDIA GPU users, please make sure to install
Vulkan driver, as the CUDA driver may not be good.

After installing all the dependencies, just follow the instructions below the install the CLI app:

```shell
# Create a new conda environment and activate the environment.
conda create -n mlc-chat
conda activate mlc-chat

# Install Git and Git-LFS if you haven't already.
# They are used for downloading the model weights from HuggingFace.
conda install git git-lfs
git lfs install

# Install the chat CLI app from Conda.
conda install -c mlc-ai -c conda-forge mlc-chat-nightly --force-reinstall

# Create a directory, download the model weights from HuggingFace, and download the binary libraries
# from GitHub. Select one of the following `LOCAL_ID` for a prebuilt LLM.

mkdir -p dist/prebuilt
git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib

# Download prebuilt weights of Vicuna-7B
cd dist/prebuilt
git clone https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0
cd ../..
mlc_chat_cli --local-id vicuna-v1-7b-q3f16_0

# Download prebuilt weights of RedPajama-3B
cd dist/prebuilt
git clone https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0
cd ../..
mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_0

# Download prebuilt weights of RWKV-raven-1.5B/3B/7B
cd dist/prebuilt
git clone https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-1b5-q8f16_0
# or git clone  https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-3b-q8f16_0
# or git clone  https://huggingface.co/mlc-ai/mlc-chat-rwkv-raven-7b-q8f16_0
cd ../..
mlc_chat_cli --local-id rwkv-raven-1b5-q8f16_0 # Replace your local id if you use 3b or 7b model.
```

<p align="center">
<img src="gif/linux-demo.gif" width="80%">
</p>

### Web Browser

Please check out [WebLLM](https://mlc.ai/web-llm/), our companion project that deploys models natively to browsers. Everything here runs inside the browser with no server support and accelerated with WebGPU.

## Links

* Check out our [GitHub repo](https://github.com/mlc-ai/mlc-llm) to see how we build, optimize and deploy the bring large-language models to various devices and backends.
* Check out our companion project [WebLLM](https://mlc.ai/web-llm/) to run the chatbot purely in your browser.
* You might also be interested in [Web Stable Diffusion](https://mlc.ai/web-stable-diffusion/), which runs the stable-diffusion model purely in the browser.
* You might want to check out our online public [Machine Learning Compilation course](https://mlc.ai) for a systematic
walkthrough of our approaches.

## Disclaimer

The pre-packaged demos are for research purposes only, subject to the model License.
