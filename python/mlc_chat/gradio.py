"""Gradio interface for MLC Chat."""


import argparse
import glob
import os

import gradio as gr
import tvm

from .chat_module import ChatModule

model_local_ids = []


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--device-name", type=str, default="cuda")
    args.add_argument("--device-id", type=int, default=0)
    args.add_argument(
        "--share",
        action="store_true",
        help="create a publicly shareable link for the interface",
    )
    parsed = args.parse_args()
    return parsed


def _check_model_dir(dir):
    if not os.path.isdir(dir):
        return False
    params_exists, model_exists = False, False
    for path in glob.glob(os.path.join(dir, "*")):
        local_id = dir.split("/")[-1]
        if path.split("/")[-1] == "params":
            params_exists = True
        if path.split("/")[-1].startswith(local_id):
            model_exists = True
    return params_exists and model_exists


def get_local_ids(artifact_path):
    for path in glob.glob(os.path.join(artifact_path, "*")):
        if _check_model_dir(path):
            local_id = path.split("/")[-1]
            model_local_ids.append(local_id)


class GradioChatModule(ChatModule):
    def __init__(self, ARGS):
        super().__init__(ARGS.device_name, ARGS.device_id)
        self.artifact_path = ARGS.artifact_path
        self.device_name = ARGS.device_name

    def reload_model(self, model_name, text_input, chat_state, img_list):
        model_lib, model_dir = None, os.path.join(self.artifact_path, model_name)
        for path in glob.glob(os.path.join(model_dir, "*")):
            if path.split("/")[-1].startswith(model_name + "-" + self.device_name):
                model_lib = path
                break
        assert model_lib is not None
        lib = tvm.runtime.load_module(os.path.join(model_dir, model_lib))
        assert lib is not None
        chat_mod.reload_func(lib, os.path.join(model_dir, "params"))
        self.reset_runtime_stats_func()
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        text_input = gr.update(interactive=True, placeholder="Type and press Enter")

        return (
            text_input,
            gr.update(interactive=True),
            gr.update(placeholder="Click to get runtime statistics."),
            gr.update(interactive=True),
            None,
            chat_state,
            img_list,
        )

    def reset_model(self, chat_state, img_list):
        self.reset_chat()
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        return None, chat_state, img_list

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


def launch_gradio(chat_mod, share_link=False):
    title = """<h1 align="center">MLC Chat Demo</h1>"""
    description = """<h3>Welcome to MLC Chat! Pick a model from your local models to get started!</h3>"""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            model_choice = gr.Radio(
                model_local_ids,
                label="Model Name",
                info="Choose a model from your local models",
            )

        with gr.Row():
            with gr.Column(scale=0.5):
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
                img_list = gr.State()
                chatbot = gr.Chatbot(label="MLC Chat")
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Select a model to start chatting!",
                    interactive=False,
                ).style(container=False)

        model_choice.change(
            chat_mod.reload_model,
            [model_choice, text_input, chat_state, img_list],
            [
                text_input,
                reset_button,
                stats_output,
                stats_button,
                chatbot,
                chat_state,
                img_list,
            ],
            queue=False,
        )
        reset_button.click(
            chat_mod.reset_model,
            [chat_state, img_list],
            [chatbot, chat_state, img_list],
        )
        stats_button.click(chat_mod.get_stats, [stats_output], [stats_output])
        text_input.submit(
            chat_mod.ask, [text_input, chatbot], [text_input, chatbot]
        ).then(chat_mod.answer, [chatbot, stream_interval], [chatbot])

    demo.launch(share=share_link, enable_queue=True)


def first_idx_mismatch(str1, str2):
    """Find the first index that mismatch in two strings."""
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            return i
    return min(len(str1), len(str2))


if __name__ == "__main__":
    ARGS = _parse_args()
    get_local_ids(ARGS.artifact_path)
    chat_mod = GradioChatModule(ARGS)
    launch_gradio(chat_mod, ARGS.share)
