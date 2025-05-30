"""Debug compiled models with TVM instrument"""

# pylint: disable=too-many-arguments
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import DataType, relax
from tvm.contrib import tvmjs
from tvm.runtime import Device, Module, Object, ShapeTuple
from tvm.runtime.relax_vm import VirtualMachine

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.interface.help import HELP
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.serve import data, engine_utils
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.style import green, red
from mlc_llm.tokenizers import Tokenizer


def _extract_metadata(mod: Module):
    return json.loads(VirtualMachine(mod, tvm.runtime.device("cpu"))["_metadata"]())


def _load_params(
    model_weight_path: str, device: Device, model_metadata: Dict[str, Any]
) -> List[tvm.nd.NDArray]:
    params, meta = tvmjs.load_ndarray_cache(model_weight_path, device)
    param_names = [param["name"] for param in model_metadata["params"]]
    assert len(param_names) == meta["ParamSize"]

    plist = []
    for param_name in param_names:
        plist.append(params[param_name])
    return plist


def _get_tvm_module(
    model_weight_path: str,
    lib_path: str,
    device: Device,
    instrument: Union[tvm.ffi.Function, None],
):
    ex = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(ex, device)
    if instrument is not None:
        vm.set_instrument(instrument)
    metadata = _extract_metadata(ex)
    params = _load_params(model_weight_path, device, metadata)
    return vm.module, params, metadata


class DefaultDebugInstrument:
    """The default debug instrument to use if users don't specify
    a customized one.

    This debug instrument will dump the arguments and output of each
    VM Call instruction into a .npz file. It will also alert the user
    if any function outputs are NaN or INF.
    """

    def __init__(self, debug_out: Path):
        """Constructor

        Parameters
        ----------
        debug_out : Path
            the directory to dump the .npz files
        """
        self.counter = 0
        self.first_nan_occurred = False
        self.first_inf_occurred = False
        self.debug_out = debug_out
        debug_out.mkdir(exist_ok=True, parents=True)

    def reset(self, debug_out: Path):
        """Reset the state of the Instrument class

        Parameters
        ----------
        debug_out : Path
            the directory to dump the .npz files
        """
        self.counter = 0
        self.first_nan_occurred = False
        self.first_inf_occurred = False
        self.debug_out = debug_out
        debug_out.mkdir(exist_ok=True, parents=True)

    def __call__(self, func, name, before_run, ret_val, *args):
        # Determine what functions to look at
        if before_run:  # Whether before the function is called or after
            return
        if self.first_nan_occurred:
            return
        if self.first_inf_occurred:
            return
        if (
            name.startswith("vm.builtin.")
            and "call_tir_dyn" not in name
            and "attention_with_fused_qkv" not in name
            and "self_attention" not in name
            and "cross_attention" not in name
        ):
            return

        # Decide what to print or save about the function's arguments (where args[-1] is the
        # buffer we write the result to)
        func_name = f"f{self.counter}_{name}"

        # Write your own behavior below. For example, we can count the number of INF/NaN in args[-1]
        def _check_nan_inf(npy):
            num_nans = np.sum(np.isnan(npy))
            num_infs = np.sum(np.isinf(npy))
            if num_nans > 0:
                print(f"{red(f'{func_name} has NaN')}: {num_nans}")
                self.first_nan_occurred = True
            if num_infs > 0:
                print(f"{red(f'{func_name} has INF')}: {num_infs}")
                self.first_inf_occurred = True

        # Save the arguments to npz
        arg_dict = {}
        for i, arg in enumerate(args):
            if isinstance(arg, tvm.nd.NDArray):
                if np.prod(arg.shape) * (DataType(arg.dtype).bits // 8) > 2147483648:
                    # We skip dump large tensors
                    arg_dict[f"arg_{i}"] = np.zeros(())
                elif arg.dtype in ["bfloat16", "float8_e4m3fn"]:
                    arg_dict[f"arg_{i}"] = arg.numpy().astype(np.float32)
                else:
                    arg_dict[f"arg_{i}"] = arg.numpy()
                _check_nan_inf(arg.numpy())
        np.savez(self.debug_out / f"{func_name}.npz", **arg_dict)

        self.counter += 1


class DebugChat:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """A chat interface used only for debugging purpose.

    It debugs auto-regressive decoding fully in Python via the prefill and
    decode interface. It supports debugging instrument (either default or
    customized) to dump intermediate values for each VM function call.

    Given a prompt, it also prints out the parsed prompt, input tokens, output
    tokens and output text.

    Sample usage:

    dc = DebugChat(
        model="./dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
        debug_dir=Path("./debug-llama-2"),
        model_lib="./dist/llama-2-7b-chat-q4f16_1-metal.so",
    )
    dc.generate("hello world", 3)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        model_lib: str,
        debug_dir: Path,
        device: Optional[str] = "auto",
        debug_instrument: Optional[Any] = None,
        is_image_model: Optional[bool] = False,
        disable_instrument: Optional[bool] = False,
    ):
        """_summary_

        Parameters
        ----------
        model: str
            The model folder after compiling with MLC-LLM build process. The parameter
            can either be the model name with its quantization scheme
            (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
            folder. In the former case, we will use the provided name to search
            for the model folder over possible paths.

        model_lib : str
            The full path to the model library file to use (e.g. a ``.so`` file).

        debug_dir: Path
            The output folder to store the dumped debug files.

        device : Optional[str]
            The description of the device to run on. User should provide a string in the
            form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
            'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
            local device), and 'device_id' is the device id to run on. If no 'device_id'
            is provided, it will be set to 0 by default.

        chat_config : Optional[ChatConfig]
            A ``ChatConfig`` instance partially filled. Will be used to override the
            ``mlc-chat-config.json``.

        debug_instrument : Optional[Any]
            An instrument function that will be called before/after each Call instruction.
            The function have the following signature:

            .. code:: python

                def instrument(
                    func: Union[VMClosure, Function],
                    func_symbol: str,
                    before_run: bool,
                    ret_value: any,
                    *args) -> bool:
                    pass

            The instrument takes the following parameters:
            - func: function object to be called.
            - func_symbol: the symbol name of the function.
            - before_run: whether it is before or after call.
            - ret_value: the return value of the call, only valid after run.
            - args: the arguments being passed to call.

        is_image_model: Optional[bool]
            Whether the model support image input. If so, will look for image embedding method.
            Default to False.

        disable_instrument: Optional[bool]
            If true, will not use debug instrument for faster generation. Default to False.
        """
        self.debug_dir = debug_dir
        self.device = detect_device(device)
        if disable_instrument:
            self.instrument = None
        else:
            self.instrument = (
                debug_instrument
                if debug_instrument
                else DefaultDebugInstrument(debug_dir / "prefill")
            )
        self.mod, self.params, self.metadata = _get_tvm_module(
            model, model_lib, self.device, self.instrument
        )
        self.model_path = Path(model)
        self.config_file_path = self.model_path / "mlc-chat-config.json"
        with open(self.config_file_path, mode="rt", encoding="utf-8") as file:
            self.chat_config = MLCChatConfig.model_validate_json(file.read())

        conv_template = self.chat_config.conv_template

        self.conversation = (
            ConvTemplateRegistry.get_conv_template(conv_template)
            if isinstance(conv_template, str)
            else conv_template
        )
        self.tokenizer = Tokenizer(str(self.model_path))

        self.add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self.begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self.end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")
        self.nd_view_func = tvm.get_global_func("vm.builtin.reshape")
        self.sample_topp_from_prob_func = tvm.get_global_func("vm.builtin.sample_top_p_from_prob")

        try:
            self.embed_func = self.mod["embed"]
        except AttributeError as exc:
            raise RuntimeError("DebugChat only supports separate embedding layer") from exc

        if is_image_model:
            try:
                self.embed_image_func = self.mod["image_embed"]
            except AttributeError as exc:
                raise RuntimeError(
                    "Expect the model to be an image model, but cannot find `image_embed`."
                ) from exc

        self.prefill_func = self.mod["prefill"]
        self.decode_func = self.mod["decode"]
        self.create_kv_cache_func = None
        if self.mod.implements_function("create_flashinfer_paged_kv_cache"):
            self.create_kv_cache_func = self.mod["create_flashinfer_paged_kv_cache"]
        elif self.mod.implements_function("create_tir_paged_kv_cache"):
            self.create_kv_cache_func = self.mod["create_tir_paged_kv_cache"]
        else:
            # TODO: Support RNN KVState # pylint: disable=fixme
            raise RuntimeError("DebugChat cannot find create KV cache function")

        self.appeared_token_freq: Dict[int, int] = {}

    def _preprocess_prompts(
        self, prompt: str, image_url: Optional[str] = None
    ) -> List[Union[List[int], data.ImageData]]:
        print("======================= Starts Tokenization & Embedding =======================")
        # Step 0. Generate prompt string using conversation template
        if image_url is None:
            self.conversation.messages.append(("user", prompt))
        else:
            self.conversation.messages.append(
                (
                    "user",
                    [
                        {"type": "image_url", "image_url": image_url},
                        {"type": "text", "text": prompt},
                    ],
                )
            )
        self.conversation.messages.append(("assistant", None))

        with open(self.config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        parsed_prompt = self.conversation.as_prompt(config)
        print(
            "Parsed prompt using conversation template "
            f"{green(self.conversation.name)}: {parsed_prompt}"
        )
        tokens = engine_utils.process_prompts(parsed_prompt, self.tokenizer.encode)  # type: ignore

        if self.conversation.system_prefix_token_ids is not None:
            tokens[0] = self.conversation.system_prefix_token_ids + tokens[0]

        return tokens

    def _embed(
        self, data_inputs: List[Union[List[int], data.ImageData]]
    ) -> Tuple[tvm.nd.NDArray, int]:
        # We currently convert to numpy after embedded, concat in numpy, then convert back to
        # tvm ndarray; could be more optimized; but may suffice for debug purposes.
        embeddings = []
        for data_input in data_inputs:
            if isinstance(data_input, data.ImageData):
                # Process image data
                # print(f"data_input.get_embed_size(): {data_input.embed_size}")
                image_input = data_input.image
                if data_input.image.device != self.device:
                    image_input = data_input.image.copyto(self.device)
                embeddings.append(self.embed_image_func(image_input, self.params).asnumpy())
            else:
                # Process token data
                data_input = tvm.nd.array(np.array(data_input).astype("int32"), device=self.device)
                embeddings.append(self.embed_func(data_input, self.params).asnumpy())
        # for embedding in embeddings:
        #     print(f"embedding.shape: {embedding.shape}")

        # Concatenate
        concat_embeddings = tvm.nd.array(np.concatenate(embeddings, axis=0), device=self.device)
        concat_embeddings = self.nd_view_func(
            concat_embeddings,
            ShapeTuple([1, concat_embeddings.shape[0], concat_embeddings.shape[1]]),
        )
        input_len = concat_embeddings.shape[1]

        return concat_embeddings, input_len

    def _prefill(self, embedding: tvm.nd.NDArray, input_len: int):
        print("======================= Starts Prefill =======================")
        seq_len_shape = ShapeTuple([input_len])
        max_num_sequence = 1
        page_size = 16
        sliding_window_size = (
            self.chat_config.sliding_window_size
            if self.chat_config.sliding_window_size
            else self.metadata["sliding_window_size"]
        )
        context_window_size = (
            self.chat_config.context_window_size
            if self.chat_config.context_window_size
            else self.metadata["context_window_size"]
        )
        prefill_chunk_size = (
            self.chat_config.prefill_chunk_size
            if self.chat_config.prefill_chunk_size
            else self.metadata["prefill_chunk_size"]
        )
        max_total_sequence_length = (
            sliding_window_size if context_window_size == -1 else context_window_size
        )
        support_sliding_window = int(sliding_window_size != -1)

        kv_caches = self.create_kv_cache_func(
            ShapeTuple([max_num_sequence]),
            ShapeTuple([max_total_sequence_length]),
            ShapeTuple([prefill_chunk_size]),
            ShapeTuple([page_size]),
            ShapeTuple([support_sliding_window]),
        )
        self.add_sequence_func(kv_caches, 0)
        self.begin_forward_func(kv_caches, ShapeTuple([0]), seq_len_shape)
        logits, kv_caches = self.prefill_func(embedding, kv_caches, self.params)
        self.end_forward_func(kv_caches)
        return logits, kv_caches

    def _decode(self, token: int, kv_caches: Object):
        embedding, _ = self._embed([[token]])
        self.begin_forward_func(kv_caches, ShapeTuple([0]), ShapeTuple([1]))
        logits, kv_caches = self.decode_func(embedding, kv_caches, self.params)
        self.end_forward_func(kv_caches)
        return logits

    def _softmax_with_temperature(self, logits: np.ndarray, temperature: float):
        # Adjust logits based on the temperature
        logits = np.array(logits) / temperature
        logits -= np.max(logits, axis=-1, keepdims=True)

        exp_logits = np.exp(logits, logits)
        exp_logits /= np.sum(exp_logits, axis=-1, keepdims=True)
        return exp_logits

    def _apply_presence_and_freq_penalty(
        self, logits: np.ndarray, presence_penalty: float, freq_penalty: float
    ):
        for token_id, freq in self.appeared_token_freq.items():
            logits[:, :, token_id] -= freq * freq_penalty + presence_penalty

    def _sample_token_from_logits(
        self,
        logits: tvm.nd.NDArray,
        *,
        temperature=1.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ):
        logits_np = logits.numpy()

        if presence_penalty != 0.0 or frequency_penalty != 0.0:
            self._apply_presence_and_freq_penalty(logits_np, presence_penalty, frequency_penalty)

        logits_np = self._softmax_with_temperature(logits_np, temperature)
        if self.instrument is not None:
            np.savez(self.instrument.debug_out / "logits.npz", logits_np)

        logits = logits.copyfrom(logits_np)
        next_token = self.sample_topp_from_prob_func(logits, top_p, random.random())
        return next_token

    def generate(
        self,
        prompt: str,
        generate_length: int,
        image_url: Optional[str] = None,
    ):
        """Generates the response from the model given a user prompt. User will need to
        specify the generation length for debugging purpose. For example, a generation
        length of 3 will include 1 prefill step and 2 decode steps.

        Parameters
        ----------
        prompt : str
            The user input prompt.

        generate_length : int
            How many tokens to generate.
        """
        out_tokens = []

        data_inputs = self._preprocess_prompts(prompt, image_url)
        print(f"{green('Data inputs: ')}: {data_inputs}")
        embedding, input_len = self._embed(data_inputs)
        logits, kv_caches = self._prefill(embedding, input_len)
        next_token = self._sample_token_from_logits(logits)
        out_tokens.append(next_token)
        if self.instrument is not None:
            path_str = (self.debug_dir / "prefill").as_posix()
            print(f"Debug instrument output dumped to {green(path_str)}")

        print("======================= Starts Decode =======================")
        for i in range(generate_length - 1):
            if self.instrument is not None:
                self.instrument.reset(self.debug_dir / f"decode_{i}")
            logits = self._decode(next_token, kv_caches)
            next_token = self._sample_token_from_logits(logits)
            out_tokens.append(next_token)
            if self.instrument is not None:
                path_str = (self.debug_dir / f"decode_{i}").as_posix()
                print(f"Debug instrument output dumped to {green(path_str)}")

            if next_token in self.conversation.stop_token_ids:
                break

        print(f"{green('Generated output tokens')}: {np.array(out_tokens)}")

        out_text = self.tokenizer.decode(out_tokens)
        print(f"{green('Generated output text')}: {out_text}")


def main():
    """The main function to start a DebugChat CLI"""

    parser = ArgumentParser("MLC LLM Chat Debug Tool")
    parser.add_argument(
        "prompt",
        type=str,
        help="The user input prompt.",
    )
    parser.add_argument(
        "--generate-len", type=int, help="Number of output tokens to generate.", required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        help="An MLC model directory that contains `mlc-chat-config.json`",
        required=True,
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        help="The full path to the model library file to use (e.g. a ``.so`` file).",
        required=True,
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        help="The output folder to store the dumped debug files.",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_compile"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--image-url",
        type=str,
        required=False,
        help="Image to prefill into the model, can only be set for image models",
    )
    parser.add_argument(
        "--disable-instrument",
        action="store_true",
        help=(
            "Disable dumping customizable detailed information of kernel input "
            + "and output, hence making generation faster."
        ),
    )
    parsed = parser.parse_args()
    dc = DebugChat(
        model=parsed.model,
        model_lib=parsed.model_lib,
        debug_dir=Path(parsed.debug_dir),
        device=parsed.device,
        is_image_model=parsed.image_url is not None,
        disable_instrument=parsed.disable_instrument,
    )

    dc.generate(parsed.prompt, parsed.generate_len, parsed.image_url)


if __name__ == "__main__":
    main()
