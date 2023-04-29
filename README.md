# MLC LLM

MLC LLM is a **universal solution** that allows **any language models** to be **deployed natively** on a diverse set of hardware backends and native applications, plus a **productive framework** for everyone to further optimize model performance for their own usecases.

Our mission is to **enable everyone to develop, optimize and deploy AI models natively on everyone's devices**.

Supported platforms include:
* Metal GPUs on iPhone and Intel/ARM MacBooks;
* AMD and NVIDIA GPUs via Vulkan on Windows and Linux;
* CUDA on Windows and Linux;
* WebGPU on broswers (through companion project WebLLM).

**[Check out our instrcution page to try out!](site/index.md)**

<p align="center">
<img src="site/demo.gif" height="700">
</p>

## What is MLC LLM

We have witnessed amazing progress in generative AI and LLM recently as generative AI and LLMs are taking over the world. Thanks to open source efforts, now it is possible to develop personal AI assistants based on open-sourced models. However, LLMs are usually chuncky and compute-heavy, and to build a scalable service, developers might have to rely on powerful clusters and expensive hardwares to run model inference. Besides, key challenges of LLM deployment is its evolving nature, and continuous model innovation, memory constraints and potential optimization techniques.

This project aims to enable development, optimization and deployment of AI models, to run inference everywhere - not only server-class hardwares, but also locally in users' browsers, laptops and even as mobile apps. To make this happen, one step forward is to solve the diverse nature of compute devices and deployment environments. To name a few:

- Different models of CPUs, GPUs and potentially other co-processors and accelerators.
- Deploy on native environment of user devices, which may not necessarily have python or other dependencies available.
- Limited memory that requires careful planning of allocation and aggressive compression of model parameters.

Furthermore, MLC LLM provides a repeatable, systematic, and customizable workflow that empowers developers and AI system researchers to implement models and optimizations in a productivity-focused, python-first approach. This approach allows for quick experimentation with new ideas and models, followed by native deployment to the desired targets. Additionally, we are continuously expanding LLM acceleration by broadening TVM backends to make model compilation more transparent and efficient.

## How

The key technology here is machine learning compilation (MLC). Our solution builts on the shoulders of the open source ecosystem, including tokenizers from huggingface and google, open source LLM models such as Llama, Vicuna, Dolly and more. The main flow builds on on Apache TVM Unity, an exciting on going development in the [Apache TVM Community](https://github.com/apache/tvm/tree/unity) 

- We bake a language model IRModule in TVM with native dynamic shape support, avoiding the need of padding to max length and reducing both computation amount and memory usage.
- Composable ML compilation optimizations: we perform many model deployment optimizations, such as better compilation code transformation, memory planning, library and manual code optimization can be easily incorporated as IRModule transformations in the python API.
- We utilize low bit quantizations to compress the model weights and leverage TensorIR to quickly customize code generations for different compression encoding schemes.
- The final generated libraries run on the native environment, with tvm runtime that comes with minimized dependencies and support GPU driver APIs and native language bindings.

<img src="site/img/diag.svg" alt="Architecture Diagram" height=""/>

We also provide a lightweight cpp based implementation to further wrap up the chat flow and necessary pre/post processing, making it easy to embed into any applications.

To enable as many GPU backends as possible, we leverage ML compilation to generate GPU shaders for different backends, as a starting point, we support CUDA, Vulkan, Metal. It is also possible to add more (such as OpenCL, sycl, webgpu-native) through improvements to tvm runtime.

All these are made possible by the open-source ecosystem that we leverage. Specifically, we make heavy use of [TVM unity](https://discuss.tvm.apache.org/t/establish-tvm-unity-connection-a-technical-strategy/13344), an exciting latest development in the TVM project that enables such Python-first interactive MLC development experiences that allows us to easily compose new optimizations, all in Python, and incrementally bring our app to the environment of interest. We also leveraged optimizations such  such as fused quantization kernels, first class dynamic shape support and diverse GPU backends. 

## Links

- You might also be interested in [WebLLM](https://github.com/mlc-ai/web-llm/tree/main), our companion derived project that focus on bringing LLM to browsers
- Project page for [instructions](site/index.md)
- [Local build Instructions for ios App](ios/README.md)
## Acknowledgements

This project is initiated by members from CMU catalyst, UW SAMPL, SJTU, OctoML and the MLC community. We would love to continue developing and supporting the open-source ML community.

This project is only possible thanks to the shoulders open-source ecosystems that we stand on. We want to thank the Apache TVM community and developers of the TVM Unity effort. The open-source ML community members made these models publicly available. PyTorch and Hugging Face communities that make these models accessible. We would like to thank the teams behind vicuna, SentencePiece, LLaMA, Alpaca. We also would like to thank the Vulkan, Swift, C++, python Rust communities that enables this project.
