---
layout: default
title: Home
notitle: true
---

# MLC LLM

MLC LLM is a universal solution that allows any language model to be deployed natively on a diverse set of hardware backends and native applications.

Please visit [Getting Started](https://llm.mlc.ai/docs/get_started/try_out.html) for detailed instructions.

## Demos

- [iOS](#ios)
- [Android](#android)
- [Windows Linux Mac](#windows-linux-mac)
- [Web browser](#web-browser)

### iOS

Our iOS app, MLCChat, is available on [App Store](https://apps.apple.com/us/app/mlc-chat/id6448482937) for iPhone and iPad.
You can try out the [Testflight app](https://testflight.apple.com/join/57zd7oxa) that sometimes contains beta release of latest models.
This app is tested on iPhone 15 Pro Max, iPhone 14 Pro Max, iPhone 14 Pro and iPhone 12 Pro.
Besides the [Getting Started](https://llm.mlc.ai/docs/get_started/try_out.html) page,
[documentation](https://llm.mlc.ai/docs/deploy/ios.html) is available for building iOS apps with MLC LLM.


<p align="center">
<img src="gif/ios-demo.gif" height="700">
</p>

Note: Llama-7B takes 4GB of RAM and RedPajama-3B takes 2.2GB to run. We recommend a latest device with 6GB RAM for Llama-7B, or 4GB RAM for RedPajama-3B, to run the app. The text generation speed could vary from time to time, for example, slow in the beginning but recover to a normal speed then.

### Android

The demo APK is available to [download](https://github.com/mlc-ai/binary-mlc-llm-libs/releases/download/Android/mlc-chat.apk). The demo is tested on Samsung S23 with Snapdragon 8 Gen 2 chip, Redmi Note 12 Pro with Snapdragon 685 and Google Pixel phones.
Besides the [Getting Started](https://llm.mlc.ai/docs/get_started/try_out.html) page,
[documentation](https://llm.mlc.ai/docs/deploy/android.html) is available for building android apps with MLC LLM.

<p align="center">
<img src="gif/android-demo.gif" height="700">
</p>

### Windows Linux Mac

Our cpp interface runs on AMD, Intel, Apple and NVIDIA GPUs.
Besides the [Getting Started](https://llm.mlc.ai/docs/get_started/try_out.html) page,
[documentation](https://llm.mlc.ai/docs/deploy/cli.html) is available for building C++ apps with MLC LLM.

<p align="center">
<img src="gif/linux-demo.gif" width="80%">
</p>

### Web Browser

[WebLLM](https://webllm.mlc.ai/) is our companion project that deploys MLC LLM natively to browsers using WebGPU and WebAssembly. Still everything runs inside the browser without server resources, and accelerated by local GPUs (e.g. AMD, Intel, Apple or NVIDIA).

## Links

* Our official [GitHub repo](https://github.com/mlc-ai/mlc-llm);
* Our companion project [WebLLM](https://webllm.mlc.ai/) that enables running LLMs purely in browser.
* [Web Stable Diffusion](https://websd.mlc.ai/) is another MLC-series that runs the diffusion models purely in the browser.
* [Machine Learning Compilation course](https://mlc.ai) is available for a systematic walkthrough of our approach to universal deployment.

## Disclaimer

The pre-packaged demos are subject to the model License.
