"""Gradio interface for LLM Chat."""


import argparse
import os

import gradio as gr
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


def launch_gradio():
    title = """<h1 align="center">MLC Chat Demo</h1>"""
    description = """<h3>This is the gradio interface for MLC chat, supporting images and text inputs.</h3>"""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=0.5):
                image = gr.Image(type="pil")
                upload_button = gr.Button(
                    value="Upload & Start Chat", interactive=True, variant="primary"
                )
                clear = gr.Button("Restart")
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True,
                    label="beam search numbers",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

            with gr.Column():
                chat_state = gr.State()
                img_list = gr.State()
                chatbot = gr.Chatbot(label="MLC Chat")
                text_input = gr.Textbox(
                    label="User",
                    placeholder="Please upload your image first",
                    interactive=False,
                )

    demo.launch(share=True, enable_queue=True)


if __name__ == "__main__":
    ARGS = _parse_args()
    chat_mod = LLMChatModule(
        ARGS.mlc_lib_path, ARGS.model_path, ARGS.device_name, ARGS.device_id
    )
    launch_gradio()
