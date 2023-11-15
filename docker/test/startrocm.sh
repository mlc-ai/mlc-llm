docker run --device=/dev/kfd --device=/dev/dri  --security-opt seccomp=unconfined --group-add video --rm --network host  -v ./mlcllm:/mlcllm mlc-rocm57:v0.1
