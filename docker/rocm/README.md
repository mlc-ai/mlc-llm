#### Pre-requisites

These containers will ONLY run on:

* System with AMD GPUs supporting the specified ROCm version(s)
* System already with the latest [ROCm supporting drivers and libraries](https://rocm.docs.amd.com/en/latest/rocm.html) installed and tested. 

They will not run on any other system, or on systems with unsupported AMD GPUs.


#### Quick start

Build the image:

```
cd rocm57
sh ./buildimage.sh
```

This will create the `mlc-rocm57` image.  You can then push it to your local or public registry for deployment. 

To run it you will use:

```
docker run --device=/dev/kfd --device=/dev/dri  --security-opt seccomp=unconfined --group-add video --rm --network host  -v ./mlcllm:/mlcllm mlc-rocm57:v0.1
```

This will mount the `./mlcllm` directoy into the container.  Before running, you must place the model's weights into `mlcllm/dist/prebuild` and the library into `mlcllm/dist/prebuild/lib`.

By default, the `llama2 7b q4f16` model is expected.c

But you can easily change the model that will be served by adding options to the command line (running red pajama 3b q4f16 instead):

```
docker run --device=/dev/kfd --device=/dev/dri  --security-opt seccomp=unconfined --group-add video --rm --network host  -v ./mlcllm:/mlcllm mlc-rocm57:v0.1  --model RedPajama-INCITE-Chat-3B-v1-q4f16_1 --device rocm --host 0.0.0.0 --port 8000
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
