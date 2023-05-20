"""Gradio interface for LLM Chat."""


import argparse
import os

import gradio as gr
import tvm
from chat_module import LLMChatModule

quantization_keys = ["q3f16_0", "q4f16_0", "q4f32_0", "q0f32", "q0f16"]


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--quantization",
        type=str,
        choices=quantization_keys,
        default=quantization_keys[0],
    )
    args.add_argument("--device-name", type=str, default="cuda")
    args.add_argument("--device-id", type=int, default=0)
    args.add_argument(
        "--mlc-path", type=str, default="", help="path to the mlc-llm repo"
    )
    parsed = args.parse_args()
    parsed.mlc_lib_path = os.path.join(parsed.mlc_path, "build/libmlc_llm_module.so")
    return parsed


class GradioChatModule(LLMChatModule):
    def __init__(self, ARGS):
        super().__init__(ARGS.mlc_lib_path, ARGS.device_name, ARGS.device_id)
        self.artifact_path = ARGS.artifact_path
        self.quantization = ARGS.quantization
        self.device_name = ARGS.device_name

    def reload_model(self, model_name, text_input, chat_state):
        model_dir = model_name + "-" + self.quantization
        model_lib = model_dir + "-" + self.device_name + ".so"
        lib = tvm.runtime.load_module(
            os.path.join(self.artifact_path, model_dir, model_lib)
        )
        assert lib is not None
        chat_mod.reload_func(lib, os.path.join(self.artifact_path, model_dir, "params"))
        if chat_state is not None:
            chat_state.messages = []
        print(f"reload model: {model_name}")
        return (
            gr.update(interactive=True, placeholder="Type and press Enter"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            None,
            chat_state,
        )

    def reset_model(self, chat_state):
        self.reset_chat()
        if chat_state is not None:
            chat_state.messages = []
        return None, chat_state

    def ask(self, text_input, chatbot):
        self.prefill(text_input)
        chatbot = chatbot + [[text_input, None]]
        return "", chatbot

    def answer(self, chatbot, stream_interval):
        i, cur_utf8_chars = 0, "".encode("utf-8")
        res = ""
        while not self.stopped():
            self.decode()
            if i % stream_interval == 0 or self.stopped():
                new_msg = self.get_message()
                new_utf8_chars = new_msg.encode("utf-8")
                pos = first_idx_mismatch(cur_utf8_chars, new_utf8_chars)
                print_msg = ""
                for _ in range(pos, len(cur_utf8_chars)):
                    print_msg += "\b \b"
                for j in range(pos, len(new_utf8_chars)):
                    print_msg += chr(new_utf8_chars[j])
                cur_utf8_chars = new_utf8_chars
                res += print_msg
                chatbot[-1][1] = res
                yield chatbot

    def get_stats(self, stats_output):
        stats_output = self.runtime_stats_text()
        return stats_output


def launch_gradio(chat_mod):
    title = """<h1 align="center">MLC Chat Demo</h1>"""
    description = """<h3>Welcome to MLC Chat!</h3>"""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        model_choice = gr.Radio(
            ["vicuna-v1-7b", "minigpt4"],
            label="Model Name",
            info="Pick a model to get started!",
        )

        with gr.Row():
            with gr.Column(scale=0.5):
                # image = gr.Image(type="pil")
                # upload_button = gr.Button(
                #    value="Upload & Start Chat", interactive=False, variant="primary"
                # )
                reset_button = gr.Button("Reset chat", interactive=False)
                stream_interval = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=2.0,
                    step=1.0,
                    interactive=True,
                    label="Stream Interval",
                )
                stats_button = gr.Button("Get Runtime Statistics", interactive=False)
                stats_output = gr.Textbox(
                    show_label=False,
                    placeholder="Click to get runtime statistics.",
                    interactive=False,
                ).style(container=False)

            with gr.Column():
                chat_state = gr.State()
                # img_list = gr.State()
                chatbot = gr.Chatbot(label="MLC Chat")
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Select a model to get started",
                    interactive=False,
                ).style(container=False)

        model_choice.change(
            chat_mod.reload_model,
            [model_choice, text_input, chat_state],
            [text_input, reset_button, stats_button, chatbot, chat_state],
        )
        reset_button.click(chat_mod.reset_model, [chat_state], [chatbot, chat_state])
        stats_button.click(chat_mod.get_stats, [stats_output], [stats_output])
        text_input.submit(
            chat_mod.ask, [text_input, chatbot], [text_input, chatbot]
        ).then(chat_mod.answer, [chatbot, stream_interval], [chatbot])

    demo.launch(share=True, enable_queue=True)


def first_idx_mismatch(str1, str2):
    """Find the first index that mismatch in two strings."""
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            return i
    return min(len(str1), len(str2))


if __name__ == "__main__":
    ARGS = _parse_args()
    chat_mod = GradioChatModule(ARGS)
    launch_gradio(chat_mod)
