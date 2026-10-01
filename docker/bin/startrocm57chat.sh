docker run --device=/dev/kfd --device=/dev/dri  --security-opt seccomp=unconfined --group-add video --rm --network host -v ./cache:/root/.cache  mlcllmrocm57:v0.1  chat   HF://mlc-ai/$1-MLC
