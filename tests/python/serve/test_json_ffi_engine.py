import tvm
import json
import mlc_llm
from mlc_llm.tokenizer import Tokenizer
from mlc_llm.serve.engine import ModelInfo, _process_model_args, EventTraceRecorder
from mlc_llm.protocol.openai_api_protocol import ChatCompletionResponse
from typing import List
from collections import defaultdict
import time
import asyncio


jsonDict = {
    "max_single_sequence_length": 128,
    "tokenizer_path": "/ssd1/abohara/MLC/mlc-llm/dist/Llama-2-13b-chat-hf-q4f16_1-MLC/params/tokenizer.json",
    "kv_cache_config": {
        "page_size": 16,
        "max_num_sequence": 2,
        "max_total_sequence_length": 128,
        "prefill_chunk_size": 128,
    },
    "engine_mode": {
        "enable_speculative": False,
        "spec_draft_length": 4,
    },
    "model_info": {
        "model": "/ssd1/abohara/MLC/mlc-llm/dist/Llama-2-13b-chat-hf-q4f16_1-MLC/params",
        "model_lib_path": "/ssd1/abohara/MLC/mlc-llm/dist/Llama-2-13b-chat-hf-q4f16_1-MLC/Llama-2-13b-chat-hf-q4f16_1-MLC-cuda.so",
        "device_type": 2,
    },
}

model = ModelInfo(
    model=jsonDict["model_info"]["model"],
    model_lib_path=jsonDict["model_info"]["model_lib_path"],
)

(
    model_args,
    config_file_paths,
    tokenizer_path,
    max_single_sequence_length,
    prefill_chunk_size,
    conv_template_name,
) = _process_model_args([model])

output_texts = defaultdict(asyncio.Queue)


def request_stream_callback(chat_completion_responses_json_str: str):
    try:
        chat_completion_responses: List[ChatCompletionResponse] = json.loads(
            chat_completion_responses_json_str
        )
        for chat_completion_response in chat_completion_responses:
            output_texts[chat_completion_response["id"]].put_nowait(
                chat_completion_response["choices"][0]
            )
    except Exception as e:
        print(f"Error in request_stream_callback: {e}")


async def print_responses(key):
    while True:
        response = await output_texts[key].get()
        print(response)


create_engine = tvm.get_global_func("mlc.json_ffi.CreateEngine")
engine = create_engine()
init_engine = engine["init"]
chat_completion = engine["chat_completion"]
get_last_error = engine["get_last_error"]


async def main():

    init_engine(
        jsonDict["max_single_sequence_length"],
        jsonDict["tokenizer_path"],
        json.dumps(jsonDict["kv_cache_config"]),
        json.dumps(jsonDict["engine_mode"]),
        request_stream_callback,
        EventTraceRecorder(),
        *model_args,
    )

    request = {
        "messages": [
            {
                "content": [
                    {
                        "type": "text",
                        "text": "What is the meaning of life?",
                    }
                ],
                "role": "user",
            }
        ],
        "model": "dist/Llama-2-13b-chat-hf-q4f16_1-MLC/params/",
    }

    request_id = "0"

    ret = chat_completion(json.dumps(request), request_id)

    print(ret)
    await asyncio.gather(print_responses(request_id))

    print(get_last_error())


asyncio.run(main())
