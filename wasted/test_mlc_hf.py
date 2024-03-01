import os

os.environ["MLC_TEMP_DIR"] = "/opt/scratch/lesheng/mlc-llm/dist"
os.environ["MLC_CACHE_DIR"] = "/opt/scratch/lesheng/mlc-llm/dist"

import logging

from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)

cm = ChatModule("HF://junrushao/Llama-2-7b-chat-hf-q3f16_1-MLC", device="cuda")
output = cm.generate(
    prompt="What is the meaning of life?",
    # progress_callback=StreamToStdout(callback_interval=2),
)
print(output)
