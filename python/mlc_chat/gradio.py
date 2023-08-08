"""Gradio interface for MLC Chat."""

import argparse
import glob
import os

import gradio as gr
import numpy as np
import tvm

from .chat_module import ChatModule, PlaceInPrompt

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
    args.add_argument("--port", type=int, default=7860)
    parsed = args.parse_args()
    return parsed


llm_local_ids = []
image_model_local_ids = []


class GradioChatModule(ChatModule):
    """The main Gradio chat module supporting a variety of functionalities such as model reloading, chatting etc."""

    def __init__(self, args):
        super().__init__(args.device_name, args.device_id)
        self.artifact_path = args.artifact_path
        self.device_name = args.device_name
        # vision-related attributes
        self.vision_exec = None
        self.vision_params = None
        self.vision_embed = None
        self.vision_dtype = None
        # keep track of current llm
        self.curr_llm_lib = None
        self.curr_llm_dir = None
        self.curr_llm_dtype = None
        # handle first user text input after vision embed
        self.is_first_input_after_vision = False

    def reload_llm(self, llm_model, chatbot, chat_state, img_list):
        """Reload the language model."""
        quantization_type = llm_model.split("-")[-1]
        if quantization_type[3:5] == "16":
            self.curr_llm_dtype = "float16"
        elif quantization_type[3:5] == "32":
            self.curr_llm_dtype = "float32"
        else:
            raise ValueError(f"LLM dtype is not supported")

        # load the langauge model
        model_dir = os.path.join(self.artifact_path, llm_model)
        lib = load_model(model_dir, self.device_name)
        chat_mod.reload(lib, os.path.join(model_dir, "params"))
        print(f"loaded {llm_model}")

        # reset environment and chatbot vars
        self.vision_exec, self.vision_params, self.vision_embed = None, None, None
        self.curr_llm_lib, self.curr_llm_dir = lib, model_dir
        self.reset_runtime_stats()
        self.process_system_prompts()
        chatbot = None
        chat_state = []
        img_list = []
        chatbot_vars = [chatbot, chat_state, img_list]

        # change button visibility and state of being interactive
        if llm_model.startswith("vicuna"):
            image_model = gr.update(value=None, interactive=True, visible=True)
        else:
            image_model = gr.update(interactive=False, visible=False)
        stream_interval = gr.update(interactive=True, visible=True)
        reset_llm_button = gr.update(interactive=True, visible=True)
        stats_button = gr.update(interactive=True, visible=True)
        stats_output = gr.update(
            placeholder="Click to get runtime statistics.", visible=True
        )
        text_input = gr.update(interactive=True, placeholder="Type and press enter")
        llm_buttons = [
            image_model,
            stream_interval,
            reset_llm_button,
            stats_button,
            stats_output,
            text_input,
        ]

        return llm_buttons + chatbot_vars

    def reload_image_model(self, image_model, chatbot, chat_state, img_list):
        """Reload the image model."""
        if image_model == "-None-" or image_model is None:
            # reload the current llm with original conv template
            if image_model == "-None-":
                assert self.curr_llm_lib is not None and self.curr_llm_dir is not None
                chat_mod.reload(
                    self.curr_llm_lib, os.path.join(self.curr_llm_dir, "params")
                )
                self.reset_runtime_stats()
                self.process_system_prompts()

            # reset environment and chatbot vars
            self.vision_exec, self.vision_params, self.vision_embed = None, None, None
            chatbot = None
            chat_state = []
            img_list = []
            chatbot_vars = [chatbot, chat_state, img_list]

            # change button visibility and state of being interactive
            image = gr.update(visible=False, interactive=False)
            text_input = gr.update(interactive=True, placeholder="Type and press enter")
            image_model_buttons = [image, text_input]

            return image_model_buttons + chatbot_vars

        # get image model dtype
        quantization_type = image_model.split("-")[-1]
        if quantization_type[3:5] == "16":
            self.vision_dtype = "float16"
        elif quantization_type[3:5] == "32":
            self.vision_dtype = "float32"
        else:
            raise ValueError(f"image model dtype is not supported")

        # reload image model
        model_dir = os.path.join(self.artifact_path, image_model)
        self.vision_exec = load_model(model_dir, self.device_name)
        self.vision_params = load_params(model_dir, self.device)
        print(f"loaded {image_model}")

        # reload the current llm with MiniGPT's conv template
        assert self.curr_llm_lib is not None and self.curr_llm_dir is not None
        app_config_json = '{"conv_template":"minigpt"}'
        chat_mod.reload(
            self.curr_llm_lib,
            os.path.join(self.curr_llm_dir, "params"),
            app_config_json,
        )

        # reset environment and chatbot vars
        self.vision_embed = None
        self.reset_runtime_stats()
        self.process_system_prompts()
        chatbot = None
        chat_state = []
        img_list = []
        chatbot_vars = [chatbot, chat_state, img_list]

        # change button visibility and state of being interactive
        image = gr.update(value=None, visible=True, interactive=True)
        text_input = gr.update(
            interactive=False, placeholder="Upload an image to start chatting"
        )
        image_model_buttons = [image, text_input]

        return image_model_buttons + chatbot_vars

    def reset_llm(self, chatbot, chat_state, img_list):
        """Reset the llm chat (and image)."""
        self.reset_chat()
        self.process_system_prompts()
        self.vision_embed = None
        if self.vision_exec is None:
            image = gr.update(visible=False, interactive=False)
            text_input = gr.update(interactive=True, placeholder="Type and press enter")
        else:
            image = gr.update(value=None, visible=True, interactive=True)
            text_input = gr.update(
                interactive=False, placeholder="Upload an image to start chatting"
            )
        chatbot = None
        chat_state = []
        img_list = []
        return image, text_input, chatbot, chat_state, img_list

    def ask(self, text_input, chatbot):
        """Process user text input."""
        if self.is_first_input_after_vision:
            self.prefill(
                text_input, decode_next_token=True, place_in_prompt=PlaceInPrompt.End
            )
            self.is_first_input_after_vision = False
        else:
            self.prefill(text_input)
        chatbot = chatbot + [[text_input, None]]
        text_input = ""
        return text_input, chatbot

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

        img_list = []
        if image is None:
            text_input = gr.update(
                placeholder="Upload an image to get started", interactive=False
            )
            return img_list, text_input

        if not isinstance(image, Image.Image):
            text_input = gr.update(
                placeholder="Uploaded image type is not supported", interactive=False
            )
            return img_list, text_input

        # generate vision embedding
        image = transform_image(image, self.vision_dtype)
        image_param = tvm.nd.array(image, self.device)
        vm = relax.vm.VirtualMachine(self.vision_exec, self.device)["embed"]
        self.vision_embed = vm(image_param, self.vision_params)
        # convert to dtype of llm
        self.vision_embed = self.vision_embed.numpy().astype(self.curr_llm_dtype)

        # prefill with vision embedding
        self.prefill(
            "<Img>",
            decode_next_token=False,
            place_in_prompt=PlaceInPrompt.Begin,
        )
        self.prefill_with_embed(tvm.nd.array(self.vision_embed, self.device), False)
        self.prefill(
            "</Img> ",
            decode_next_token=False,
            place_in_prompt=PlaceInPrompt.Middle,
        )
        self.is_first_input_after_vision = True

        # change button
        text_input = gr.update(placeholder="Type and press enter", interactive=True)
        return img_list, text_input

    def get_stats(self, stats_output):
        """Get runtime statistics."""
        stats_output = self.stats()
        return stats_output


# main function for the design of Gradio interface


def launch_gradio(chat_mod, share_link=False):
    title = """<h1 align="center">MLC Chat Gradio Interface</h1>"""
    description = """<h3>Welcome to MLC Chat! Pick a model from your local ids to get started.</h3>"""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=0.3):
                llm_model = gr.Dropdown(llm_local_ids, label="Language Model")
                image_model = gr.Dropdown(
                    ["-None-"] + image_model_local_ids,
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
                reset_llm_button = gr.Button(
                    "Reset chat", visible=False, interactive=False
                )
                stats_button = gr.Button(
                    "Get Runtime Statistics", interactive=False, visible=False
                )
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

        # buttons whose visibility change when llm reload
        llm_buttons = [
            image_model,
            stream_interval,
            reset_llm_button,
            stats_button,
            stats_output,
            text_input,
        ]
        # buttons whose visibility change when image model reload
        image_model_buttons = [image, text_input]
        # chatbot state variables
        chatbot_vars = [chatbot, chat_state, img_list]

        # change of state controls
        llm_model.change(
            chat_mod.reload_llm,
            [llm_model] + chatbot_vars,
            llm_buttons + chatbot_vars,
            queue=False,
        )
        image_model.change(
            chat_mod.reload_image_model,
            [image_model] + chatbot_vars,
            image_model_buttons + chatbot_vars,
            queue=False,
        )
        text_input.submit(
            chat_mod.ask, [text_input, chatbot], [text_input, chatbot]
        ).then(chat_mod.answer, [chatbot, stream_interval], [chatbot])
        image.upload(chat_mod.upload_image, [image], [img_list, text_input])
        image.clear(
            chat_mod.reset_llm,
            chatbot_vars,
            image_model_buttons + chatbot_vars,
        )
        reset_llm_button.click(
            chat_mod.reset_llm,
            chatbot_vars,
            image_model_buttons + chatbot_vars,
        )
        stats_button.click(chat_mod.get_stats, [stats_output], [stats_output])

    demo.launch(share=share_link, enable_queue=True, server_port=ARGS.port)


# helper functions below


def _check_model_dir(model_dir):
    """Check the validity of model directory."""
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
    """Get all model ids in the artifact path, and categorize into llm and image models."""
    for path in glob.glob(os.path.join(artifact_path, "*")):
        if _check_model_dir(path):
            local_id = path.split("/")[-1]
            if local_id.startswith("minigpt"):
                image_model_local_ids.append(local_id)
            else:
                llm_local_ids.append(local_id)
    llm_local_ids.sort()
    image_model_local_ids.sort()


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
