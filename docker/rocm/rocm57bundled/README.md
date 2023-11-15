### Combining weights and library of a model into a mega-sized container image

#### Pre-requisites

These containers will ONLY run on:

* System with AMD GPUs supporting the specified ROCm version(s)
* System already with the latest [ROCm supporting drivers and libraries](https://rocm.docs.amd.com/en/latest/rocm.html) installed and tested. 

They will not run on any other system, or on systems with unsupported AMD GPUs.


#### Quick start

Before building the image, you MUST place the associated weights and library of the model into the `fixedmodel` directory.

1. place the weights into the `fixedmodel/dist/prebuilt` folder  (the default example is for Llama-2-7b-chat-hf-q4f16_1 and `mlc-chat-Llama-2-7b-chat-hf-q4f16_1` from huggingface should be `git cloned` here)
2. place the library into `fixedmodel/dist/prebuilt/lib` folder  (in the case of rocm57 for llama2 7b q4f16_1, it will be `Llama-2-7b-chat-hf-q4f16_1-rocm.so`)

Make sure there is no extra files littering under the `fixedmodel` directory hierachy, as _everything_ under this directory will be added into the container image.

Build the image:

```
cd rocm57bundled
sh ./buildimage.sh
```

This will create the `llama2rocm57` bundled image.  You can then push it to your local or public registry for deployment. 

To run it you will use:

```
docker run --device=/dev/kfd --device=/dev/dri  --security-opt seccomp=unconfined --group-add video --rm --network host llama2rocm57:v0.1
```

By default the container will listen to `port 8000` of the host machine for REST API calls.

You can use the python script in the `test` subfolder to test your container:

```
$ python sample_client_for-testing.py 
Without streaming:
Of course! Here is a haiku for you:

Sun sets slowly down
Golden hues upon the sea
Peaceful evening sky

Reset chat: <Response [200]>

With streaming:
Of course! Here is a haiku for you:

Respectful and true
Honest and helpful assistant
Serving with grace

Runtime stats: prefill: 259.2 tok/s, decode: 145.6 tok/s
```

See the `test` subfolder for examples of these scripts.
