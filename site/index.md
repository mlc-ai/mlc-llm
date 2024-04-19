---
layout: default
title: Home
notitle: true
---

# MLC LLM

[Documentation](https://llm.mlc.ai/docs) | [Blog](https://blog.mlc.ai/) | [Discord][discord-url]

**M**achine **L**earning **C**ompilation for **L**arge **L**anguage **M**odels (MLC LLM) is a high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration. The mission of this project is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices with ML compilation techniques.

**Universal deployment.** MLC LLM supports the following platforms and hardware:

<table style="width:100%">
  <thead>
    <tr>
      <th style="width:15%"> </th>
      <th style="width:20%">AMD GPU</th>
      <th style="width:20%">NVIDIA GPU</th>
      <th style="width:20%">Apple GPU</th>
      <th style="width:24%">Intel GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linux / Win</td>
      <td>✅ Vulkan, ROCm</td>
      <td>✅ Vulkan, CUDA</td>
      <td>N/A</td>
      <td>✅ Vulkan</td>
    </tr>
    <tr>
      <td>macOS</td>
      <td>✅ Metal (dGPU)</td>
      <td>N/A</td>
      <td>✅ Metal</td>
      <td>✅ Metal (iGPU)</td>
    </tr>
    <tr>
      <td>Web Browser</td>
      <td colspan=4>✅ WebGPU and WASM </td>
    </tr>
    <tr>
      <td>iOS / iPadOS</td>
      <td colspan=4>✅ Metal on Apple A-series GPU</td>
    </tr>
    <tr>
      <td>Android</td>
      <td colspan=2>✅ OpenCL on Adreno GPU</td>
      <td colspan=2>✅ OpenCL on Mali GPU</td>
    </tr>
  </tbody>
</table>


## Quick Start

We introduce the quick start examples of chat CLI, Python API and REST server here to use MLC LLM.
We use 4-bit quantized 8B Llama-3 model for demonstration purpose.
The pre-quantized Llama-3 weights is available at https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC.
You can also try out unquantized Llama-3 model by replacing `q4f16_1` to `q0f16` in the examples below.
Please visit our [documentation](https://llm.mlc.ai/docs/index.html) for detailed quick start and introduction.

### Installation

MLC LLM is available via [pip](https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages).
It is always recommended to install it in an isolated conda virtual environment.

To verify the installation, activate your virtual environment, run

```bash
python -c "import mlc_llm; print(mlc_llm.__path__)"
```

You are expected to see the installation path of MLC LLM Python package.

### Chat CLI

We can try out the chat CLI in MLC LLM with 4-bit quantized 8B Llama-3 model.

```bash
mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```

It may take 1-2 minutes for the first time running this command.
After waiting, this command launch a chat interface where you can enter your prompt and chat with the model.

```
You can use the following special commands:
/help               print the special commands
/exit               quit the cli
/stats              print out the latest stats (token/sec)
/reset              restart a fresh chat
/set [overrides]    override settings in the generation config. For example,
                      `/set temperature=0.5;max_gen_len=100;stop=end,stop`
                      Note: Separate stop words in the `stop` option with commas (,).
Multi-line input: Use escape+enter to start a new line.

user: What's the meaning of life
assistant:
What a profound and intriguing question! While there's no one definitive answer, I'd be happy to help you explore some perspectives on the meaning of life.

The concept of the meaning of life has been debated and...
```

### Python API

We can run the Llama-3 model with the chat completion Python API of MLC LLM.
You can save the code below into a Python file and run it.

```python
from mlc_llm import LLMEngine

# Create engine
model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
engine = LLMEngine(model)

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    model=model,
    stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

engine.terminate()
```

**The Python API of `mlc_llm.LLMEngine` fully aligns with OpenAI API**.
You can use LLMEngine in the same way of using
[OpenAI's Python package](https://github.com/openai/openai-python?tab=readme-ov-file#usage)
for both synchronous and asynchronous generation.

If you would like to do concurrent asynchronous generation, you can use `mlc_llm.AsyncLLMEngine` instead.

### REST Server

We can launch a REST server to serve the 4-bit quantized Llama-3 model for OpenAI chat completion requests.
The server has fully OpenAI API completeness.

```bash
mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```

The server is hooked at `http://127.0.0.1:8000` by default, and you can use `--host` and `--port`
to set a different host and port.
When the server is ready (showing `INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`),
we can open a new shell and send a cURL request via the following command:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
        "messages": [
            {"role": "user", "content": "Hello! Our project is MLC LLM. What is the name of our project?"}
        ]
  }' \
  http://127.0.0.1:8000/v1/chat/completions
```

## Universal Deployment APIs

MLC LLM provides multiple sets of APIs across platforms and environments. These include
* [Python API](https://llm.mlc.ai/docs/deploy/python_engine.html)
* [OpenAI-compatible Rest-API](https://llm.mlc.ai/docs/deploy/rest.html)
* [C++ API](https://llm.mlc.ai/docs/deploy/cli.html)
* [JavaScript API](https://llm.mlc.ai/docs/deploy/javascript.html) and [Web LLM](https://github.com/mlc-ai/web-llm)
* [Swift API for iOS App](https://llm.mlc.ai/docs/deploy/ios.html)
* [Java API and Android App](https://llm.mlc.ai/docs/deploy/android.html)

## Links

- You might want to check out our online public [Machine Learning Compilation course](https://mlc.ai) for a systematic
walkthrough of our approaches.
- [WebLLM](https://webllm.mlc.ai/) is a companion project using MLC LLM's WebGPU and WebAssembly backend.
- [WebStableDiffusion](https://websd.mlc.ai/) is a companion project for diffusion models with the WebGPU backend.

## Disclaimer

The pre-packaged demos are subject to the model License.
