import asyncio
import json
import threading
from collections import defaultdict
from typing import List

import tvm

from mlc_llm.protocol.openai_api_protocol import ChatCompletionResponse
from mlc_llm.serve.engine_base import EventTraceRecorder, ModelInfo, _process_model_args

json_dict = {
    "max_single_sequence_length": 128,
    "tokenizer_path": "/ssd1/abohara/MLC/mlc-llm/dist/Llama-2-13b-chat-hf-q4f16_1-MLC/params/tokenizer.json",
    "kv_cache_config": {
        "page_size": 16,
        "max_num_sequence": 2,
        "max_total_sequence_length": 1024,
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
    model=json_dict["model_info"]["model"],
    model_lib_path=json_dict["model_info"]["model_lib_path"],
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


async def print_responses(key):
    while True:
        response = await output_texts[key].get()
        print(response)


create_engine = tvm.get_global_func("mlc.json_ffi.CreateEngine")
engine = create_engine()
chat_completion = engine["chat_completion"]
get_last_error = engine["get_last_error"]


async def main():
    async_event_loop = asyncio.get_event_loop()

    def request_stream_callback(chat_completion_responses_json_str: str):
        async_event_loop.call_soon_threadsafe(
            request_stream_callback_impl, chat_completion_responses_json_str
        )

    def request_stream_callback_impl(chat_completion_responses_json_str: str):
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

    def _background_loop():
        engine["init"](
            json_dict["max_single_sequence_length"],
            json_dict["tokenizer_path"],
            json.dumps(json_dict["kv_cache_config"]),
            json.dumps(json_dict["engine_mode"]),
            request_stream_callback,
            EventTraceRecorder(),
            *model_args,
        )
        engine["run_background_loop"]()

    def _background_stream_back_loop():
        engine["run_background_stream_back_loop"]()

    background_loop_thread: threading.Thread = threading.Thread(target=_background_loop)
    background_stream_back_loop_thread: threading.Thread = threading.Thread(
        target=_background_stream_back_loop
    )
    background_loop_thread.start()
    background_stream_back_loop_thread.start()

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

    print(f"success = {ret}")
    await asyncio.gather(print_responses(request_id))

    print(get_last_error())

    engine["exit_background_loop"]()
    background_loop_thread.join()
    background_stream_back_loop_thread.join()


asyncio.run(main())
