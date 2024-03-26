# NOTE: This Dockerfile is based on CUDA 12.1.
# To benchmark on other CUDA versions, search and replace "12.1" and "121".
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV MLC_PATH /mlcllm

# setup python 3 and pip, load the mlc-ai nightlies

RUN apt update                      && \
    apt install --yes  python3 pip  && \
    pip install --pre --force-reinstall -f https://mlc.ai/wheels \
    mlc-ai-nightly-cu121 mlc-chat-nightly-cu121   &&\
    mkdir -p $MLC_PATH 

VOLUME ${MLC_PATH} 

WORKDIR ${MLC_PATH}

ENTRYPOINT ["python3", "-m", "mlc_chat.rest"] 

CMD ["--model", "Llama-2-7b-chat-hf-q4f16_1", "--device", "cuda",  "--host", "0.0.0.0", "--port", "8000"]


