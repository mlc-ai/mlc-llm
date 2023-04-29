---
layout: default
title: Home
notitle: true
---

# MLC LLM

MLC LLM is a universal solution that allows any language model to be deployed natively on a diverse set of hardware backends and native applications, plus a productive framework for everyone to further optimize model performance for their own use cases. Checkout our [GitHub repository](https://github.com/mlc-ai/mlc-llm) to see how we did it. You can also read through instructions below for trying out demos.

<p align="center">
<img src="demo.gif" height="700">
</p>

## Get Started

This section contains the instructions to run large-language models and chatbot natively on your environment.

### Windows, Linux, Mac

We provide a CLI (command-line interface) app to chat with the bot in your terminal. Before installing
the CLI app, we should install some dependencies first.
1. We use Conda to manage our app, so we need to install a version of conda. We can install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
2. On Windows and Linux, the chatbot application runs on GPU via the Vulkan platform. For Windows and Linux users,
please install the latest [Vulkan driver](https://developer.nvidia.com/vulkan-driver).

After installing all the dependencies, just follow the instructions below the install the CLI app:
```shell
# Create a new conda environment and activate the environment.
conda create -n mlc-chat
conda activate mlc-chat

# Install Git and Git-LFS, which is used for downloading the model weights
# from Hugging Face.
conda install git git-lfs

# Install the chat CLI app from Conda.
conda install -c mlc-ai -c conda-forge mlc-chat-nightly

# Create a directory, download the model weights from HuggingFace, and download the binary libraries
# from GitHub.
mkdir -p dist
git lfs install
git clone https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int3 dist/vicuna-v1-7b
git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/lib

# Enter this line and enjoy chatting with the bot running natively on your machine!
mlc_chat_cli
```

### iOS

Please check out this [TestFlight invitation](https://testflight.apple.com/join/57zd7oxa) to install and use
our iOS chat app. This invitation is limited to first 9000 users.

Note: The text generation speed on the iOS app can be unstable from time to time. It might run slow
for a while and recover to a normal speed then.

### Web Browser

Please check out [WebLLM](https://mlc.ai/web-llm/), our work that brings large-language model and
LLM-based chatbot to web browsers. Everything here runs inside the browser with no server support and accelerated with WebGPU.

## Links

You might also be interested in [Web Stable Diffusion](https://mlc.ai/web-stable-diffusion/).

## Disclaimer
The pre-packaged demos are for research purposes only, subject to the model License.
