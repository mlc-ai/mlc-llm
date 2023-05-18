"""Gradio interface for LLM Chat."""

from mlc_llm.conversation import conv_templates
from python.mlc_chat.chat_module import from_llm_dylib

if __name__ == "__main__":
    chat_mod = from_llm_dylib("/root/dist/llama-7b-hf-q0f16")
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
