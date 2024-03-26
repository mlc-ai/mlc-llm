# NOTE: This Dockerfile is based on ROCm 5.7
FROM rocm/dev-ubuntu-22.04:5.7-complete

ENV MLC_PATH /mlcllm

# setup python 3 and pip, load the mlc-ai nightlies

RUN apt update                      && \
    apt install --yes  python3 pip  && \
    pip install --pre --force-reinstall -f https://mlc.ai/wheels \
    mlc-ai-nightly-rocm57 mlc-chat-nightly-rocm57   &&\
    mkdir -p $MLC_PATH 

VOLUME ${MLC_PATH} 

WORKDIR ${MLC_PATH}

ENTRYPOINT ["python3", "-m", "mlc_chat.rest"] 

CMD ["--model", "Llama-2-7b-chat-hf-q4f16_1", "--device", "rocm",  "--host", "0.0.0.0", "--port", "8000"]


