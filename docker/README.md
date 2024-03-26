## Containers for mlc-llm REST API

A set of docker container templates for your scaled production deployment of GPU-accelerated mlc-llms.  Based on the work in the [LLM  Performance Benchmark](https://github.com/mlc-ai/llm-perf-bench) project.  

These containers are designed to be:

* minimalist - nothing non-essential is included;  you can layer on your own security policy for example
* non-opinionated - use CNCF k8s or docker compose or swarm or whatever you have for orchestration
* adaptive and composable - nobody knows what you intend to do with these containers, and we don't guess
* compatible - with multi-GPU support maturing and batching still in testing, these containers should suvive upcoming changes without needing to be severely revamped. 
* practical NOW - usable and deployable TODAY with 2023/2024 level hardware and mlc-ai

###  Structure

Containers are segregated by GPU acceleration stacks.  See the README of the sub-folders for more information
```
cuda
|-- cuda121
|-- cuda121bundled  

rocm
|-- rocm37
|-- rocm37bundled

test
```
####  Community contribution

This structure enables the greater community to easily contribute new tested templates for other cuda and rocm releases, for example.

####  Bundled for greatly improved UX

Having the weights and library seperated for each and every model powers great flexibility but makes for a horrandous user experience.   

The `bundled` variants of the containers include both weights and the library for a model inside the single mega-sized image.   It is actually constructed based on the non-bundled image (composable), it just "add weights and the lib" to the image.  

Users of such images can simply decide to run "llama2 7b on cuda 12.1" and in a few seconds pull down  an image onto their workstation running AI apps served by Llama 2 already GPU accelerated.

The tradeoff here is the storage requirement, and possibly network bandwidth during deployment.

Here is a glimpse of what the size difference between bundled vs unbundled version. (7.7 GB for the q4f16 llama2 weights, for example)

```
REPOSITORY           TAG       IMAGE ID       CREATED         SIZE
llama2rocm57         v0.1      249adf862921   3 minutes ago   27.7GB
mlc-rocm57           v0.1      f4b7894e8724   3 hours ago     20GB
``` 
Ultimately mlc-ai may decide to publish on its public Docker Hub some images of bundled popular models (such as mistral 7b on cuda 121 and rocm37) to enhance "out of box" UX for the project.

##### Loss of flexibility?

There is no flexibility loss in deploying bundled containers because, as least for 2023/2024 hardware. The research community is working on multi-requests based on batching.  This means that once the weights of a massive LLM is loaded into GPU VRAM, it ain't moving.  Instead inference requests are batched against the loaded LLM for concurrent processing. The "GPU to LLM" mapping will remain static.

#### Unbundled containers

Non-bundled container is coded by default to serve _llama2 7b q4f16_, but you can change the command line during `docker run` to have it serve any supported mlc-llms.  Yes, you will have to download the weights and build the library yourself and place them in the docker mounted volume.  See `red pajama` examples in the `test` directory.

#### Tests
Tests are made global to conserve storage.  
 
