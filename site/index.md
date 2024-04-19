---
layout: default
title: Home
notitle: true
---

# MLC LLM

Documentation: [https://llm.mlc.ai/docs](https://llm.mlc.ai/docs)

**M**achine **L**earning **C**ompilation for **L**arge **L**anguage **M**odels (MLC LLM) is a high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration. The mission of this project is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices with ML compilation techniques.

<p align="center">
<img src="https://llm.mlc.ai/docs/_images/project-workflow.svg" height="300">
</p>

## Installation

MLC LLM is available via [pip](https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages).
It is always recommended to install it in an isolated conda virtual environment.

To verify the installation, activate your virtual environment, run

```bash
python -c "import mlc_llm; print(mlc_llm.__path__)"
```

You are expected to see the installation path of MLC LLM Python package.

## Quick Start

Please check out our documentation for the [quick start](https://llm.mlc.ai/docs/get_started/quick_start.html).

## Introduction

Please check out our documentation for the [introduction](https://llm.mlc.ai/docs/get_started/introduction.html).

## Links

- You might want to check out our online public [Machine Learning Compilation course](https://mlc.ai) for a systematic
walkthrough of our approaches.
- [WebLLM](https://webllm.mlc.ai/) is a companion project using MLC LLM's WebGPU and WebAssembly backend.
- [WebStableDiffusion](https://websd.mlc.ai/) is a companion project for diffusion models with the WebGPU backend.

## Disclaimer

The pre-packaged demos are subject to the model License.
