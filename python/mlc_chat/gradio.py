"""Gradio interface for LLM Chat."""


import argparse
import os

from chat_module import LLMChatModule

quantization_keys = ["q3f16_0", "q4f16_0", "q4f32_0", "q0f32", "q0f16"]


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--model", type=str, default="vicuna-v1-7b")
    args.add_argument(
        "--quantization",
        type=str,
        choices=quantization_keys,
        default=quantization_keys[0],
    )
    args.add_argument("--device-name", type=str, default="cuda")
    args.add_argument("--device-id", type=int, default=0)
    args.add_argument("--mlc-path", type=str, help="path to the mlc-llm repo")
    parsed = args.parse_args()
    parsed.mlc_lib_path = os.path.join(parsed.mlc_path, "build/libmlc_llm_module.so")
    parsed.model_path = os.path.join(
        parsed.artifact_path, parsed.model + "-" + parsed.quantization
    )
    return parsed


if __name__ == "__main__":
    ARGS = _parse_args()
    chat_mod = LLMChatModule(
        ARGS.mlc_lib_path, ARGS.model_path, ARGS.device_name, ARGS.device_id
    )

    """
    conv = conv_templates[chat_mod.conv_template].copy()
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        if inp[:6] == "/reset":
            chat_mod.reset_chat()
            continue
        if inp[:5] == "/exit":
            break
        if inp[:6] == "/stats":
            stats_text = chat_mod.runtime_stats_text()
            print(stats_text)
            continue

        print(f"{conv.roles[1]}: ")
        chat_mod.encode(inp)
        i = 0
        while not chat_mod.stopped():
            chat_mod.decode()
            if i % chat_mod.stream_interval == 0 or chat_mod.stopped():
                curr_msg = chat_mod.get_message()
        print(curr_msg)
    """
