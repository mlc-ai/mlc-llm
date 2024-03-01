# pylint: skip-file
import os
import shutil

import tvm

tvm.error.InternalError

temp_dir = "/opt/scratch/lesheng/mlc-llm/dist/tmp"

os.environ["MLC_TEMP_DIR"] = temp_dir
os.environ["MLC_CACHE_DIR"] = temp_dir

import logging

from mlc_chat import ChatConfig, ChatModule
from mlc_chat.callback import StreamToStdout

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)

import json


def test_model(url):
    os.makedirs(temp_dir, exist_ok=True)
    cm = ChatModule(
        url,
        chat_config=ChatConfig(max_gen_len=10000, prefill_chunk_size=4096),
        device="cuda",
    )
    output = cm.generate(
        prompt="What is the meaning of life?",
    )
    print(output)
    print(cm.stats())
    for i in range(10):
        prompt = input("User: ")
        output = cm.generate(
            prompt=prompt,
        )
        print(output)
        print(cm.stats())
    return output


url = "HF://junrushao/Mistral-7B-Instruct-v0.2-q4f16_1-MLC"
# url = "HF://junrushao/phi-2-q4f16_1-MLC"
# url = "HF://junrushao/Llama-2-7b-chat-hf-q4f16_1-MLC"
test_model(url)
