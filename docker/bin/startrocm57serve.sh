docker run --device=/dev/kfd --device=/dev/dri  --security-opt seccomp=unconfined --group-add video --rm --network host mlcllmrocm57:v0.1 serve HF://mlc-ai/$1-MLC  --host 0.0.0.0 --port 8000
