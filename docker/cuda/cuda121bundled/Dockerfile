FROM mlc-cuda121:v0.1 
ADD fixedmodel ./fixedmodel
WORKDIR ./fixedmodel

CMD ["--model", "Llama-2-7b-chat-hf-q4f16_1", "--device", "cuda",  "--host", "0.0.0.0", "--port", "8000"]


