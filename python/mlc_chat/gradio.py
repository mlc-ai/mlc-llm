"""Gradio interface for LLM Chat."""


import argparse

from mlc_llm.conversation import conv_templates
from python.mlc_chat.chat_module import LLMChatModule


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--model",
        type=str,
        default="auto",
        help='The name of the model to build. If it is "auto", we will automatically set the '
        'model name according to "--model-path", "hf-path" or the model folders under '
        '"--artifact-path/models"',
    )
    args.add_argument(
        "--quantization",
        type=str,
        choices=[*utils.quantization_dict.keys()],
        default=list(utils.quantization_dict.keys())[0],
    )
    parsed = args.parse_args()


if __name__ == "__main__":
    chat_mod = LLMChatModule("/root/dist/llama-7b-hf-q0f16")
    chat_mod.init_chat(model="llama-7b-hf")

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
