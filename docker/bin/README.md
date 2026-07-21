###  container startup scripts


>  NOTE:  Please make sure you are in the `bin` directory when starting these scripts.  Make sure you have write permissions to the `cache` directory. These scripts make use of the `cache` folder there to cache all model weights and custom compiled libraries.

The `<model name>` that are supported (at any time) can be obtained from [MLC AI's Huggingface Repo](https://huggingface.co/mlc-ai). There are *88 supported models at the time of writing* and soon will have hundreds more.

![image](https://github.com/Sing-Li/dockertest/assets/122633/e1068b42-cfe1-4385-8c71-0791d2987d8b)

Some currently popular `model names` that our community are actively exploring include:

* `Llama-2-7b-chat-hf-q4f16_1`
* `Mistral-7B-Instruct-v0.2-q4f16_1`
* `gemma-7b-it-q4ff16_2`
* `phi-1_5_q4f32_1`

Try using these `<model name>` when parameterizing the scripts.

You can modify the `serve` scripts directly to support specific network interfaces (on a multi-homed system, defaults to `0.0.0.0` = all interfaces) and to change the listening port (defaults to port `8000`).

|Command | Description | Usage|
|-------|------|------|
|`startcuda122chat.sh` |  starts up command line interactive chat with specified LLM on Cuda 12.1 linux system | `sh ./startcuda122chat.sh <mlc model name>`|
|`startcuda122serve.sh` | runs a server handling multiple concurrent REST API calls to  the specified LLM on Cuda 12.1 linux system| `sh ./startcuda122serve.sh <mlc model name>`|
|`startrocm57chat.sh` |  starts up command line interactive chat with specified LLM on Rocm 5.7 linux system | `sh ./startrocm57chat.sh <mlc model name>`|
|`startrocm57serve.sh` | runs a server handling multiple concurrent REST API calls to  the specified LLM on Rocm 5.7 linux system| `sh ./startrocm57serve.sh <mlc model name>`|
