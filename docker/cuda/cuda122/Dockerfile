FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV MLC_PATH /mlcllm

# setup python 3 and pip, load the mlc-ai nightlies

RUN apt update                      && \
    apt install --yes  python3.11 pip git git-lfs && \
    pip install --pre -U -f https://mlc.ai/wheels \
    mlc-llm-nightly-cu122 mlc-ai-nightly-cu122   &&\
    mkdir -p $MLC_PATH 

VOLUME ${MLC_PATH} 

WORKDIR ${MLC_PATH}


ENTRYPOINT ["mlc_llm"] 

CMD ["chat", "HF://mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC"]


