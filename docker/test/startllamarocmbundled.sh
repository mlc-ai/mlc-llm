docker run --device=/dev/kfd --device=/dev/dri  --security-opt seccomp=unconfined --group-add video --rm --network host llama2rocm57:v0.1
