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
    try:
        cm = ChatModule(
            url,
            chat_config=ChatConfig(max_gen_len=10000, prefill_chunk_size=4096),
            device="cuda",
        )
        output = cm.generate(
            prompt="What is the meaning of life?",
            # prompt="Write a python program that computes fibonacci sequence."
        )
        print(output)
        for i in range(10):
            output = cm.generate(
                prompt="Talk more",
                # prompt="Write a python program that computes fibonacci sequence."
            )
            print(output)
    except:
        output = "Failed"
    # shutil.rmtree(temp_dir, ignore_errors=True)
    return output


url = "HF://junrushao/Mistral-7B-Instruct-v0.2-q4f16_1-MLC"
output = test_model(url)
print(output)
exit(0)

with open("model_info.json", "r") as f:
    model_dict = json.load(f)

testout_dir = "testout"
os.makedirs(testout_dir, exist_ok=True)

shutil.rmtree(temp_dir, ignore_errors=True)
for model in model_dict:
    model_infos = model_dict[model]
    for quant in ["q3f16_1", "q4f16_1", "q4f32_1"]:
        url = model_infos[quant]
        if len(url) == 0:
            continue
        print(f"{model}\t{quant}: {url}")
        testout_file = os.path.join(testout_dir, f"{model}_{quant}.out")
        if f"{model}_{quant}" in ["CodeLlama-7b-hf_q4f32_1"]:
            continue
        if os.path.exists(testout_file):
            print(f"skip {testout_file}")
            continue
        output = test_model(url)
        with open(testout_file, "w+") as f:
            f.write(output + "\n")
