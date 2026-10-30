docker run --gpus all --rm -it --network host -v ./cache:/root/.cache  mlcllmcuda122:v0.1  chat  HF://mlc-ai/$1-MLC
