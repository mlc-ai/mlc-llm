# pylint: skip-file
import json
import logging
import os
import shutil

import tvm

from mlc_chat import ChatConfig, ChatModule

temp_dir = "/opt/scratch/lesheng/mlc-llm/dist/disco_tmp"

os.environ["MLC_TEMP_DIR"] = temp_dir
os.environ["MLC_CACHE_DIR"] = temp_dir

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def test_model(url, num_shards=1):
    try:
        cm = ChatModule(
            url,
            chat_config=ChatConfig(
                context_window_size=1024,
                max_gen_len=200,
                tensor_parallel_shards=num_shards,
            ),
            device="cuda",
        )
        output = cm.generate(
            # prompt="What is the meaning of life?",
            prompt="Write a python program that computes a + b."
        )
    except:
        output = "Failed"
    return output


print(test_model("HF://junrushao/gpt2-q4f16_1-MLC", 4))
exit(0)

with open("model_info.json", "r") as f:
    model_dict = json.load(f)

testout_dir = "disco_out"
os.makedirs(testout_dir, exist_ok=True)

# shutil.rmtree(temp_dir, ignore_errors=True)
for model in model_dict:
    model_infos = model_dict[model]
    for quant in ["q4f16_1", "q4f32_1"]:  # , "q3f16_1"]:
        url = model_infos[quant]
        if len(url) == 0:
            continue

        print(f"{model}/{quant}: {url}")

        os.makedirs(temp_dir, exist_ok=True)

        model_out_dir = os.path.join(testout_dir, model)
        os.makedirs(model_out_dir, exist_ok=True)
        for i in [1, 2, 4]:  # , 8]:
            print(i)
            testout_file = os.path.join(model_out_dir, f"{quant}-{i}.out")

            if os.path.exists(testout_file):
                print(f"skip {testout_file}")
                continue

            output = test_model(url, i)
            with open(testout_file, "w+") as f:
                f.write(output + "\n")

        shutil.rmtree(temp_dir, ignore_errors=True)

        # if f"{model}_{quant}" in ["CodeLlama-7b-hf_q4f32_1"]:
        #     continue
        # if os.path.exists(testout_file):
        #     print(f"skip {testout_file}")
        #     continue
