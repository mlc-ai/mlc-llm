"""Gradio interface for MLC Chat."""

import argparse
import glob
import os

import gradio as gr
import numpy as np
import tvm

from .chat_module import ChatModule

# pylint-disable=import-error, import-outside-toplevel, invalid-name


def _parse_args():
    args = argparse.ArgumentParser("MLC-Chat Gradio Interface")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--device-name", type=str, default="cuda")
    args.add_argument("--device-id", type=int, default=0)
    args.add_argument(
        "--share",
        action="store_true",
        help="create a publicly shareable link for the interface",
    )
    args.add_argument("--port", type=int, default=8000)
    parsed = args.parse_args()
    return parsed


model_local_ids = []
minigpt_ids = []


class GradioChatModule(ChatModule):
    """Gradio chat module supporting functionalities like reload model, chat, upload image etc."""

    def __init__(self, args):
        super().__init__(args.device_name, args.device_id)
        self.artifact_path = args.artifact_path
        self.device_name = args.device_name
        self.dtype = None
        # vision-related attributes
        self.vision_exec = None
        self.vision_params = None
        self.vision_embed = None

    def reload_model(self, model_name, chat_state, img_list):
        """Reload the model after user selects a model from the local library."""
        quantization_type = model_name.split("-")[-1]
        if quantization_type[3:5] == "16":
            self.dtype = "float16"
        elif quantization_type[3:5] == "32":
            self.dtype = "float32"
        else:
            raise ValueError("dtype not supported yet")

        if model_name.startswith("minigpt4"):
            # find the image module given the dtype, can use a different dtype if given one is not found
            # currently, no quantization is supported for image module
            model_name = model_name.strip(quantization_type) + "q0f" + self.dtype[-2:]
            if model_name not in minigpt_ids:
                new_suffix = "16" if self.dtype == "float32" else "32"
                model_name[-2:] = new_suffix
                assert model_name in minigpt_ids, "failed to find minigpt4"
            model_dir = os.path.join(self.artifact_path, model_name)
            self.vision_exec = load_model(model_dir, self.device_name)
            self.vision_params = load_params(model_dir, self.device)
            self.vision_embed = None
            print(f"loaded image module from path: {model_dir}")
            # find the corresponding text-based chat module
            model_name = ""
            for candidate_name in model_local_ids:
                if candidate_name.startswith("vicuna") and candidate_name.endswith(
                    quantization_type
                ):
                    model_name = candidate_name
                    break
            assert model_name != "", "failed to load vicuna model"
            conv_template_json = '{"conv_template":"minigpt"}'
            text_input = gr.update(
                interactive=False, placeholder="Upload an image to start chatting"
            )
            image = gr.update(visible=True, interactive=True)
        else:
            self.vision_exec, self.vision_params, self.vision_embed = None, None, None
            conv_template_json = ""
            text_input = gr.update(interactive=True, placeholder="Type and press enter")
            image = gr.update(visible=False)

        # load the text-based chat component
        model_dir = os.path.join(self.artifact_path, model_name)
        lib = load_model(model_dir, self.device_name)
        chat_mod.reload_func(lib, os.path.join(model_dir, "params"), conv_template_json)
        print(f"loaded text-based chat module from path: {model_dir}")
        self.process_system_prompts()
        self.reset_runtime_stats_func()
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []

        return (
            text_input,
            gr.update(interactive=True),
            gr.update(placeholder="Click to get runtime statistics."),
            gr.update(interactive=True),
            image,
            None,
            chat_state,
            img_list,
        )

    def reset_model(self, chat_state, img_list):
        # TODO: complete this
        """Reset the chatbot."""
        self.reset_chat()
        self.process_system_prompts()
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        return None, chat_state, img_list

    def ask(self, text_input, chatbot):
        # TODO: complete this
        """Process user text input."""
        if self.has_embed():
            embed_array = self.embed(text_input)
            if len(embed_array) == 2:
                assert self.vision_embed is not None
                # concatenate in python by first converting into numpy arrays, then use np.concatenate()
                embedding = np.concatenate(
                    (
                        embed_array[0].numpy(),
                        self.vision_embed.numpy(),
                        embed_array[1].numpy(),
                    ),
                    axis=1,
                )
                embedding = tvm.nd.array(embedding, self.device)
            else:
                assert len(embed_array) == 1, "should only contain one embed array"
                embedding = embed_array[0]
            self.prefill_with_embed(embedding)
        else:
            self.prefill(text_input)
        chatbot = chatbot + [[text_input, None]]
        return "", chatbot

    def answer(self, chatbot, stream_interval):
        """Process the chatbot's text response."""
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

    def upload_image(self, image):
        """Process user image input."""
        from PIL import Image
        from tvm import relax

        if image is None:
            text_input = gr.update(
                placeholder="Upload an image to get started", interactive=False
            )
        elif not isinstance(image, Image.Image):
            text_input = gr.update(
                placeholder="Uploaded image type is not supported", interactive=False
            )
        else:
            image = transform_image(image, self.dtype)
            image_param = tvm.nd.array(image, self.device)
            vm = relax.vm.VirtualMachine(self.vision_exec, self.device)["embed"]
            self.vision_embed = vm(image_param, self.vision_params)
            text_input = gr.update(placeholder="Type and press enter", interactive=True)
        img_list = []
        return text_input, img_list

    def get_stats(self, stats_output):
        """Get runtime statistics."""
        stats_output = self.runtime_stats_text()
        return stats_output


def launch_gradio(chat_mod, share_link=False):
    """Launch the Gradio interface."""
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
                image = gr.Image(type="pil", interactive=False, visible=False)
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
            [model_choice, chat_state, img_list],
            [
                text_input,
                reset_button,
                stats_output,
                stats_button,
                image,
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
        image.upload(chat_mod.upload_image, [image], [text_input, img_list])
        # TODO: complete this
        # image.clear()

    demo.launch(share=share_link, enable_queue=True, server_port=ARGS.port)


def _check_model_dir(model_dir):
    if not os.path.isdir(model_dir):
        return False
    params_exists, model_exists = False, False
    for path in glob.glob(os.path.join(model_dir, "*")):
        local_id = model_dir.split("/")[-1]
        if path.split("/")[-1] == "params":
            params_exists = True
        if path.split("/")[-1].startswith(local_id):
            model_exists = True
    return params_exists and model_exists


def get_local_ids(artifact_path):
    """Get all local model ids in the local directory."""
    candidate_local_ids = []
    minigpt_names = []
    for path in glob.glob(os.path.join(artifact_path, "*")):
        if _check_model_dir(path):
            local_id = path.split("/")[-1]
            candidate_local_ids.append(local_id)
            quantization_suffix = "-" + local_id.split("-")[-1]
            if local_id.startswith("minigpt"):
                minigpt_ids.append(local_id)
            if (
                local_id.startswith("minigpt")
                and local_id.strip(quantization_suffix) not in minigpt_names
            ):
                minigpt_names.append(local_id.strip(quantization_suffix))
    for local_id in candidate_local_ids:
        if local_id.startswith("minigpt"):
            continue
        if local_id.startswith("vicuna") and len(minigpt_names) != 0:
            quantization_type = local_id.split("-")[-1]
            for minigpt_name in minigpt_names:
                if minigpt_name + "-" + quantization_type not in model_local_ids:
                    model_local_ids.append(minigpt_name + "-" + quantization_type)
        model_local_ids.append(local_id)
    model_local_ids.sort()


def load_params(model_dir, device):
    """Load model parameters from the local directory."""
    from tvm.contrib import tvmjs

    params, meta = tvmjs.load_ndarray_cache(f"{model_dir}/params", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


def load_model(model_dir, device_name):
    """Load model executable from the local directory."""
    model_lib, model_name = None, model_dir.split("/")[-1]
    for path in glob.glob(os.path.join(model_dir, "*")):
        if path.split("/")[-1].startswith(model_name + "-" + device_name):
            model_lib = path
            break
    assert model_lib is not None, "compiled model does not exist"
    lib = tvm.runtime.load_module(os.path.join(model_dir, model_lib))
    assert lib is not None, "model executable cannot be loaded"
    return lib


def first_idx_mismatch(str1, str2):
    """Find the first index that mismatch in two strings."""
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            return i
    return min(len(str1), len(str2))


def transform_image(image, dtype, image_size=224):
    """Preprocess the user image"""
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    image = transform_fn(image).unsqueeze(0)
    if dtype == "float16":
        image = image.half()
    elif dtype == "float32":
        image = image.float()
    else:
        raise ValueError("image dtype not supported yet")
    return image


if __name__ == "__main__":
    ARGS = _parse_args()
    get_local_ids(ARGS.artifact_path)
    chat_mod = GradioChatModule(ARGS)
    launch_gradio(chat_mod, ARGS.share)
