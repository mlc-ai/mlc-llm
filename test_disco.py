# pylint: skip-file

MODEL = "HF://junrushao/Llama-2-7b-chat-hf-q4f16_1-MLC"
# MODEL = "HF://junrushao/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC"
# MODEL = "HF://junrushao/WizardCoder-15B-V1.0-q4f16_1-MLC"
# MODEL = "HF://junrushao/Mistral-7B-Instruct-v0.2-q4f16_1-MLC"
# MODEL = "HF://junrushao/phi-1_5-q4f16_1-MLC"
# MODEL = "HF://junrushao/phi-2-q4f16_1-MLC"
TP_SHARDS = 2
import os

from mlc_chat import ChatConfig, ChatModule, callback
from mlc_chat.support import logging

# temp_dir = "/opt/scratch/lesheng/mlc-llm/dist/tmp"

# os.environ["MLC_TEMP_DIR"] = temp_dir
# os.environ["MLC_CACHE_DIR"] = temp_dir
# os.environ["MLC_JIT_POLICY"] = "REDO"


logging.enable_logging()

cm = ChatModule(
    MODEL,
    device="cuda",
    chat_config=ChatConfig(
        # context_window_size=1024,
        # prefill_chunk_size=1024 if "Mistral" not in MODEL else 4096,
        tensor_parallel_shards=TP_SHARDS,
        # opt="flashinfer=0;cublas_gemm=1;cudagraph=0",
    ),
)
cm.generate(
    "What is the meaning of life?",
    progress_callback=callback.StreamToStdout(callback_interval=2),
)
