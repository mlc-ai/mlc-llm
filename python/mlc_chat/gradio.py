"""Gradio interface for MLC Chat."""
# pylint: disable=import-error,invalid-name,too-many-instance-attributes,too-many-locals
import argparse
import glob
import os
from typing import Dict, Optional

import gradio as gr

from .chat_module import ChatModule


def _parse_args():
    args = argparse.ArgumentParser("MLC-Chat Gradio Interface")
    args.add_argument(
        "--artifact-path",
        type=str,
        default="dist",
        help="Please provide a path containing all the model folders you wish to use.",
    )
    args.add_argument(
        "--device",
        type=str,
        default="auto",
        help="The description of the device to run on. User should provide a string in the \
            form of 'device_name:device_id' or 'device_name', where 'device_name' is one of \
                'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the \
                    local device), and 'device_id' is the device id to run on. If no 'device_id' \
                        is provided, it will be set to 0 by default.",
    )
    args.add_argument("--port", type=int, default=7860, help="The port number to run gradio.")
    args.add_argument("--host", type=str, default="127.0.0.1", help="The local host to run gradio.")
    args.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a publicly shareable link for the interface.",
    )
    parsed = args.parse_args()
    return parsed


def _get_all_available_models_under_dir(artifact_path: str) -> Dict[str, str]:
    r"""Given the artifact path storing all models, returns a dict mapping available model names
    to the correct `model` args passed into ChatModule.

    Note
    ----
    We only search for folders under the artifact_path, without recursive search for subfolders.
    For each folder, we count it as a valid MLC model folder if either it contains an
    `mlc-chat-config.json` file, or it contains a `params` folder which contains an
    `mlc-chat-config.json` file. We will map the name of a valid folder to its full path to the
    folder containing `mlc-chat-config.json`.
    """

    # step 0. retrieve the absolute path of artifact_path
    search_dir = os.path.abspath(artifact_path)
    if not os.path.exists(search_dir):
        err_msg = (
            f"The artifact path {artifact_path} you provided is neither a valid full path nor a "
            "valid path relative to the current working directory. Please provide a correct "
            "artifact path.",
        )
        raise FileNotFoundError(err_msg)

    # step 1. go through all the folders, build the model dict
    model_dict = {}
    for path in glob.glob(os.path.join(search_dir, "*")):
        if os.path.isdir(path):
            model_name = os.path.basename(os.path.normpath(path))
            # check if it contains `mlc-chat-config.json`
            if os.path.exists(os.path.join(path, "mlc-chat-config.json")):
                model_dict[model_name] = os.path.abspath(path)
            # check if it contains `params/mlc-chat-config.json`
            elif os.path.exists(os.path.join(path, "params", "mlc-chat-config.json")):
                model_dict[model_name] = os.path.abspath(os.path.join(path, "params"))

    return model_dict


class GradioModule:
    r"""The Gradio module for MLC Chat. Different from ChatModule Python API, Gradio module allows
    users to load in a directory of models, watch the streaming in web browser, and switch between
    models more easily to compare performance.

    Note: Multimodality will be supported soon, i.e. allowing users to upload an image to chat.
    """

    def __init__(self, artifact_path: str = "dist", device: str = "auto"):
        self.artifact_path = artifact_path
        self.device_str = device
        self.chat_mod: Optional[ChatModule] = None
        self.model_dict = _get_all_available_models_under_dir(artifact_path)

    def gradio_reload_model(self, model_name: str):
        r"""Reload the model given the user-selected model name."""
        self.chat_mod = ChatModule(self.model_dict[model_name], self.device_str)

        updated_dict = {
            "chatbot": None,
            "chat_state": [],
            "img_list": [],
            "image_model": gr.update(interactive=False, visible=False),
            "stream_interval": gr.update(interactive=True, visible=True),
            "reset_llm_button": gr.update(interactive=True, visible=True),
            "stats_button": gr.update(interactive=True, visible=True),
            "stats_output": gr.update(placeholder="Click to get runtime statistics.", visible=True),
            "text_input": gr.update(interactive=True, placeholder="Type and press enter"),
        }

        return list(updated_dict.values())

    def gradio_reset_model(self):
        r"""Reset the current chat model."""
        self.chat_mod.reset_chat()

        updated_dict = {
            "chatbot": None,
            "chat_state": [],
            "img_list": [],
            "text_input": gr.update(interactive=True, placeholder="Type and press enter"),
        }

        return list(updated_dict.values())

    def gradio_ask(self, text_input, chatbot):
        r"""Display user text input in the chatbot."""
        chatbot = chatbot + [[text_input, None]]
        text_input = ""
        return text_input, chatbot

    def gradio_answer(self, chatbot, stream_interval):
        r"""Generate and display the chat module's response.
        Note: Below is a low-level implementation of generate() API, since it's easier
        to yield without delta callback."""
        prompt = chatbot[-1][0]
        # pylint: disable=protected-access
        self.chat_mod._prefill(prompt)
        i, new_msg = 0, ""
        while not self.chat_mod._stopped():
            self.chat_mod._decode()
            if i % stream_interval == 0 or self.chat_mod._stopped():
                new_msg = self.chat_mod._get_message()
                chatbot[-1][1] = new_msg
                yield chatbot
            i += 1
        # pylint: enable=protected-access

    def gradio_stats(self):
        """Get runtime statistics."""
        return self.chat_mod.stats()


def launch_gradio(
    artifact_path: str = "dist",
    device: str = "auto",
    port: int = 7860,
    share: bool = False,
    host: str = "127.0.0.1",
):
    r"""Launch the gradio interface with a given port, creating a publically sharable link if
    specified."""

    # create a gradio module
    mod = GradioModule(artifact_path, device)

    title = """<h1 align="center">MLC Chat Gradio Interface</h1>"""
    description = (
        """<h3>Welcome to MLC Chat! Pick a model from your local ids to get started.</h3>"""
    )

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)

        # ---------------------- user interface design -------------------------
        with gr.Row():
            with gr.Column(scale=0.3):
                llm_model = gr.Dropdown(list(mod.model_dict.keys()), label="Language Model")
                image_model = gr.Dropdown(
                    ["-None-"],
                    label="Do you wanna add an image model?",
                    visible=False,
                    interactive=False,
                )
                image = gr.Image(type="pil", interactive=False, visible=False)
                stream_interval = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=2.0,
                    step=1.0,
                    interactive=True,
                    visible=False,
                    label="Stream Interval",
                )
                reset_llm_button = gr.Button("Reset chat", visible=False, interactive=False)
                stats_button = gr.Button("Get Runtime Statistics", interactive=False, visible=False)
                stats_output = gr.Textbox(
                    show_label=False,
                    placeholder="Click to get runtime statistics.",
                    interactive=False,
                    visible=False,
                    container=False,
                )
            with gr.Column():
                chat_state = gr.State()
                img_list = gr.State()
                chatbot = gr.Chatbot(label="MLC Chat")
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Select a model to start chatting!",
                    interactive=False,
                    container=False,
                )

        # ---------------------- local variables ---------------------------
        # type 1. buttons whose visibility change when llm reload
        llm_buttons = [
            image_model,
            stream_interval,
            reset_llm_button,
            stats_button,
            stats_output,
            text_input,
        ]
        # type 2. buttons whose visibility change when image model reload
        # pylint: disable=unused-variable
        image_model_buttons = [image, text_input]
        # type 3. chatbot state variables
        chatbot_vars = [chatbot, chat_state, img_list]

        # -------------------------- handle control --------------------------
        llm_model.change(
            mod.gradio_reload_model, [llm_model], chatbot_vars + llm_buttons, queue=False
        )
        text_input.submit(mod.gradio_ask, [text_input, chatbot], [text_input, chatbot]).then(
            mod.gradio_answer, [chatbot, stream_interval], [chatbot]
        )
        reset_llm_button.click(mod.gradio_reset_model, [], chatbot_vars + [text_input])
        stats_button.click(mod.gradio_stats, [], [stats_output])

    # launch to the web
    demo.launch(share=share, enable_queue=True, server_port=port, server_name=host)


if __name__ == "__main__":
    ARGS = _parse_args()
    launch_gradio(ARGS.artifact_path, ARGS.device, ARGS.port, ARGS.share, ARGS.host)
