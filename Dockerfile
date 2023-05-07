FROM nvidia/vulkan:1.3-470

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8   
ENV PATH /opt/conda/bin:$PATH

RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y curl gpg && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg && \
    install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg && \
    gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806 && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y conda && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN conda create -n mlc-chat

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "mlc-chat", "/bin/bash", "-c"]

RUN conda install git git-lfs && \
    conda install -c mlc-ai -c conda-forge mlc-chat-nightly

#RUN mkdir -p dist && git lfs install && \
#  git clone https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int3 dist/vicuna-v1-7b && \
#  git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/lib

CMD ["conda", "run", "--live-stream", "-n", "mlc-chat", "/bin/bash", "-c", "mlc_chat_cli"]
