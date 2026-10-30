docker run --gpus all --rm --network host -v ./cache:/root/.cache  mlcllmcuda122:v0.1  serve  HF://mlc-ai/$1-MLC  --host 0.0.0.0 --port 8000
