from typing import Dict, List, Optional
import random
import inspect
import json
import logging
import os
import sys
from enum import Enum
from dataclasses import asdict, dataclass, fields
from transformers import AutoTokenizer  # type: ignore[import]

import numpy as np

from .types import (InferenceEngine, Request, InferenceStepResult, TextGenerationOutput,
                    ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, SamplingParams)
import tvm
from tvm import relax


## amalyshe: below code stays in mlc-llm/python/mlc_chat/chat_module.py
# nothing is changed, ideally we need to remove it and include original one
# TODO(amalyshe): refactor this/reuse original

# pylint: disable=line-too-long
_PYTHON_GET_STARTED_TUTORIAL_URL = "https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb"
# pylint: enable=line-too-long


@dataclass
class ConvConfig:
    r"""A dataclass that represents user-defined partial configuration for conversation template.

    This is an attribute of :class:`mlc_chat.ChatConfig`, which can then be passed in to the
    instantiation of a :class:`mlc_chat.ChatModule` instance to override the default
    setting in ``mlc-chat-config.json`` under the model folder. Note that we will
    first load the predefined template with the name specified in ``conv_template``.

    Since the configuration is partial, everything will be ``Optional``.

    Parameters
    ----------
    name : Optional[str]
        Name of the conversation.
    system : Optional[str]
        The prompt encoded before starting the chat.
    roles : Optional[List[str]]
        An array that describes the role names of the user and the model. These
        names are specific to the model being used.
    messages : Optional[List[str]]
        The chat history represented as an array of string pairs in the following
        format: ``[[role_0, msg_0], [role_1, msg_1], ...]``.
    offset : Optional[str]
        The offset used to begin the chat from the chat history. When offset
        is not ``0``, ``messages[0:offset-1]`` will be encoded.
    separator_style : Optional[int]
        Specifies whether we are in chat-bot mode (``0``) or pure LM prompt mode (``1``).
    seps : Optional[List[str]]
        An array of strings indicating the separators to be used after a user
        message and a model message respectively.
    role_msg_sep : Optional[str]
        A string indicating the separator between a role and a message.
    role_empty_sep : Optional[str]
        A string indicating the separator to append to a role when there is no message yet.
    stop_str : Optional[str]
        When the ``stop_str`` is encountered, the model will stop generating output.
    stop_tokens : Optional[List[int]]
        A list of token IDs that act as stop tokens.
    add_bos : Optional[bool]
        Determines whether a beginning-of-string (bos) token should be added
        before the input tokens.
    """

    name: Optional[str] = None
    system: Optional[str] = None
    roles: Optional[List[str]] = None
    messages: Optional[List[List[str]]] = None
    offset: Optional[str] = None
    separator_style: Optional[int] = None
    seps: Optional[List[str]] = None
    role_msg_sep: Optional[str] = None
    role_empty_sep: Optional[str] = None
    stop_str: Optional[str] = None
    stop_tokens: Optional[List[int]] = None
    add_bos: Optional[bool] = None

    def __post_init__(self):
        if self.messages is not None and self.offset is None:
            self.offset = len(self.messages)


@dataclass
class ChatConfig:
    r"""A dataclass that represents user-defined partial configuration for the
    chat config file.

    An instance of ``ChatConfig`` can be passed in to the instantiation of a
    :class:`mlc_chat.ChatModule` instance to override the default setting in
    ``mlc-chat-config.json`` under the model folder.

    Since the configuraiton is partial, everything will be ``Optional``.

    Note that we will exploit this class to also represent ``mlc-chat-config.json``
    during intermediate processing.

    Parameters
    ----------
    model_lib : Optional[str]
        The necessary model library to launch this model architecture. We recommend
        reuse model library when possible. For example, all LLaMA-7B models can
        use ``vicuna-v1-7b-{matching quantization scheme}``. So you can distribute
        LLaMA-7B weight variants and still use them in prebuilt MLC chat apps.
    local_id : Optional[str]
        Uniquely identifying the model in application. This is also used by
        command line interface app to specify which model to run.
    conv_template : Optional[str]
        The name of the conversation template that this chat uses.
    temperature : Optional[float]
        The temperature applied to logits before sampling. The default value is
        ``0.7``. A higher temperature encourages more diverse outputs, while a
        lower temperature produces more deterministic outputs.
    repetition_penalty : Optional[float]
        The repetition penalty controls the likelihood of the model generating
        repeated texts. The default value is set to ``1.0``, indicating that no
        repetition penalty is applied. Increasing the value reduces the
        likelihood of repeat text generation. However, setting a high
        ``repetition_penalty`` may result in the model generating meaningless
        texts. The ideal choice of repetition penalty may vary among models.

        For more details on how repetition penalty controls text generation, please
        check out the CTRL paper (https://arxiv.org/pdf/1909.05858.pdf).
    top_p : Optional[float]
        This parameter determines the set of tokens from which we sample during
        decoding. The default value is set to ``0.95``. At each step, we select
        tokens from the minimal set that has a cumulative probability exceeding
        the ``top_p`` parameter.

        For additional information on top-p sampling, please refer to this blog
        post: https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling.
    mean_gen_len : Optional[int]
    max_gen_len : Optional[int]
    shift_fill_factor : Optional[float]
    tokenizer_files : Optional[List[str]]
        List of tokenizer files of the model.
    conv_config : Optional[ConvConfig]
        The partial overriding configuration for conversation template. Will first
        load the predefined template with the name specified in ``conv_template``
        and then override some of the configuraitons specified in ``conv_config``.
    model_category : Optional[str]
        The category of the model's architecture (e.g. ``llama``, ``gpt_neox``, ``rwkv``).
    model_name : Optional[str]
        Name of the model (e.g. ``Llama-2-7b-chat-hf``).
    num_shards: Optional[str]
        Tensor parallel degree.
    max_window_size: Optional[str]
        Maximum kv cache window size.
    """

    model_lib: Optional[str] = None
    local_id: Optional[str] = None
    conv_template: Optional[str] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    mean_gen_len: Optional[int] = None
    max_gen_len: Optional[int] = None
    shift_fill_factor: Optional[float] = None
    tokenizer_files: Optional[List[str]] = None
    conv_config: Optional[ConvConfig] = None
    model_category: Optional[str] = None
    model_name: Optional[str] = None
    num_shards: Optional[int] = None
    max_window_size: Optional[int] = None

    @classmethod
    def _from_json(chat_config_cls, json_obj: dict):
        return chat_config_cls(
            **{
                k: v
                for k, v in json_obj.items()
                if k in inspect.signature(chat_config_cls).parameters
            }
        )


class PlaceInPrompt(Enum):
    """The place of an input message in a prompt."""

    # The input message should have role names and corresponding seperators appended both prior to it and after it,
    # making it a complete prompt.
    All = 0
    # The input message is only the beginning part of a prompt, no role name and separator should be appended after
    # the message since there will be future messages appended after the message.
    Begin = 1
    # The input message is in the middle of a prompt, nothing should be appended before or after the message.
    Middle = 2
    # The input message is the ending part of a prompt, no role name and separator should be appended prior to it
    # since the message is concatenated to some prior messages.
    End = 3

def _get_model_path(model: str) -> (str, str):
    """Use user-provided argument ``model`` to search for a valid model path.

    We define "valid" as having an ``mlc-chat-config.json`` right under the folder.

    Parameters
    ----------
    model : str
        User's input; may be a compiled model's name, or a full path.

    Returns
    ------
    model_path : str
        A "valid" path to model folder, with ``os.isfile(os.path.join(model_path,
        "mlc-chat-config.json"))`` being ``True``.
    chat_file : str
        Essentially ``os.path.join(model_path, "mlc-chat-config.json")``.

    Raises
    ------
    FileNotFoundError: if we cannot find a valid `model_path`.
    """
    # Note that the order of this list corresponds to our search priority
    candidate_paths = [
        f"{model}",  # full path, or just the name
        f"dist/prebuilt/{model}",  # Using prebuilt workflow
        f"dist/{model}/params",  # Default directory after mlc_llm.build_model()
        f"dist/prebuilt/mlc-chat-{model}",  # Also prebuilt workflow, but missed prefix
    ]

    # Look for the first folder that has `mlc-chat-config.json` under it
    for candidate in candidate_paths:
        chat_file = os.path.join(candidate, "mlc-chat-config.json")
        if os.path.isfile(chat_file):
            # TODO(amalyshe): add logging
            # logging.info(f"Using model folder: {os.path.abspath(candidate)}")
            # logging.info(f"Using mlc chat config: {os.path.abspath(chat_file)}")
            return candidate, chat_file

    # Failed to find a valid model_path, analyzing error for user

    # First see if any candidate path is an actual folder
    found_folder = False
    valid_dir_str = ""
    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            valid_dir_str += f"- {os.path.abspath(candidate)}\n"
            found_folder = True

    if found_folder:
        # Error 1: there is a folder, but not an mlc-llm model folder (E1)
        err_msg = (
            "The model folder provided does not seem to refer to a valid mlc-llm model folder.\n"
            "Specifically, we cannot find `mlc-chat-config.json`, a required file. You should "
            "provide a path that contains the file.\n"
            "According to your input `model`, we looked at folder(s):\n"
            f"{valid_dir_str}"
            "MLC-Chat consumes models that are processed by the MLC-LLM build process.\n"
            f"Please checkout {_PYTHON_GET_STARTED_TUTORIAL_URL} for an example on "
            "how to load a model."
        )
        raise FileNotFoundError(err_msg)
    else:
        # Error 2: cannot find a folder (E0)
        all_paths_str = ""
        for path in candidate_paths:
            all_paths_str += f"- {path}\n"
        err_msg = (
            "Cannot find the model folder. We searched over the following possible paths:\n"
            f"{all_paths_str}"
            "You can try to pass in `model=/path/to/your-model-path`, and confirm "
            "that it contains `mlc-chat-config.json`, among other essential files.\n"
            f"Please checkout {_PYTHON_GET_STARTED_TUTORIAL_URL} for an "
            "example on how to load a model."
        )
        raise FileNotFoundError(err_msg)


def _get_chat_config(config_file_path: str, user_chat_config: Optional[ChatConfig]) -> ChatConfig:
    """Read in the config file in model path, then potentially override with user input.

    Parameters
    ----------
    config_file_path : str
        ``chat_file`` returned by ``_get_model_path()``.
    user_chat_config : Optional[ChatConfig]
        User's input, a partial ``ChatConfig`` to override the one in ``config_file_path``.

    Returns
    ------
    final_chat_config : ChatConfig
        ``ChatConfig`` corresponding to ``config_file_path``, overriden by ``user_chat_config``.
    """
    final_chat_config = None
    with open(config_file_path, mode="rt", encoding="utf-8") as f:
        json_object = json.load(f)
        final_chat_config = ChatConfig._from_json(json_object)
    if user_chat_config is not None:
        # We override using user's chat config
        for field in fields(user_chat_config):
            field_name = field.name
            field_value = getattr(user_chat_config, field_name)
            if field_value is not None:
                setattr(final_chat_config, field_name, field_value)
    return final_chat_config


def _get_lib_module_path(
    model: str,
    model_path: str,
    chat_config: ChatConfig,
    lib_path: Optional[str],
    device_name: str,
    config_file_path: str,
) -> str:
    """Look up the model library. Then return a corresponding ``tvm`` runtime Module.

    Parameters
    ----------
    model : str
        User's input; may be a compiled model's name, or a full path.
    model_path : str
        Model path found by `_get_model_path`.
    chat_config : ChatConfig
        Chat config after potential overrides. Returned by ``_get_chat_config``.
    lib_path : Optional[str]
        User's input. Supposedly a full path to model library. Prioritized to use.
    device_name : str
        User's input. Used to construct the library model file name.
    config_file_path : str
        The path to ``mlc-chat-config.json``. Used for error message making.

    Returns
    ------
    lib_path : str
        The path pointing to the model library we find.

    Raises
    ------
    FileNotFoundError: if we cannot find a valid model library file.
    """
    # 1. Use user's lib_path if provided
    if lib_path is not None:
        if os.path.isfile(lib_path):
            # TODO(amalyshe): add logging
            # logging.info(f"Using library model: {lib_path}")
            return lib_path
        else:
            err_msg = (
                f"The `lib_path` you passed in is not a file: {lib_path}.\nPlease checkout "
                f"{_PYTHON_GET_STARTED_TUTORIAL_URL} for an example on how to load a model."
            )
            raise FileNotFoundError(err_msg)

    # 2. Generate all possible file names according to OS
    candidate_lib_names = []
    if sys.platform.startswith("linux"):
        candidate_lib_names = [f"{chat_config.model_lib}-{device_name}.so"]
    elif sys.platform.startswith("Darwin"):
        # Note that `dylib` comes before `so` since we prioritize `dylib` for MacOS
        candidate_lib_names = [
            f"{chat_config.model_lib}-{device_name}.dylib",
            f"{chat_config.model_lib}-{device_name}.so",
        ]
    elif sys.platform.startswith("win32"):
        candidate_lib_names = [f"{chat_config.model_lib}-{device_name}.dll"]
    else:
        candidate_lib_names = [
            f"{chat_config.model_lib}-{device_name}.dylib",
            f"{chat_config.model_lib}-{device_name}.so",
            f"{chat_config.model_lib}-{device_name}.dll",
        ]

    # 3. Generate possible model library paths
    candidate_paths = []
    for lib_name in candidate_lib_names:
        # Equivalent to {model_path}/../
        pardir_model_path = os.path.abspath(os.path.join(os.path.abspath(model_path), os.pardir))
        candidate_paths.extend(
            [
                f"{lib_name}",
                f"dist/prebuilt/lib/{lib_name}",  # Using prebuilt workflow
                f"dist/{model}/{lib_name}",  # Default directory after mlc_llm.build_model()
                os.path.join(model_path, lib_name),  # User put library inside `model_path`
                os.path.join(pardir_model_path, lib_name),  # Under parent directory of `model_path`
            ]
        )

    # 4. Search for model library
    for candidate in candidate_paths:
        if os.path.isfile(candidate):
            # TODO(amalyshe): add logging
            # logging.info(f"Using library model: {os.path.abspath(candidate)}\n")
            return candidate

    # 5. Error
    err_msg = (
        f"Cannot find the model library that corresponds to `{chat_config.model_lib}`.\n"
        f"`{chat_config.model_lib}` is either provided in the `chat_config` "
        f"you passed in, or specified in {config_file_path}.\n"
        "We searched over the following possible paths: \n"
    )
    for candidate in candidate_paths:
        err_msg += f"- {candidate}\n"
    err_msg += (
        "If you would like to directly specify the model library path, you may "
        "consider passing in the `lib_path` parameter.\n"
        f"Please checkout {_PYTHON_GET_STARTED_TUTORIAL_URL} for an example "
        "on how to load a model."
    )
    raise FileNotFoundError(err_msg)


## amalyshe: this code is taken from mlc-llm/utils.py
# not to have dependencies on root module, decided to copy function here
def _load_params(artifact_path: str, device) -> List[tvm.nd.NDArray]:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(artifact_path, device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


class Model:
    kv_cache = None
    def __init__(self, model_config) -> None:
        device = model_config.device

        device_err_msg = (
            f"Invalid device name: {device}. Please enter the device in the form "
            "'device_name:device_id' or 'device_name', where 'device_name' needs to be "
            "one of 'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto'."
        )

        # 0. Retrieve device_name and device_id (if any, default 0) from device arg
        device_args = device.split(":")
        if len(device_args) == 1:
            device_name, device_id = device_args[0], 0
        elif len(device_args) == 2:
            device_name, device_id = device_args[0], int(device_args[1])
        elif len(device_args) > 2:
            raise ValueError(device_err_msg)

        # 1. Get self.device
        if device_name == "cuda":
            self.device = tvm.cuda(device_id)
        elif device_name == "metal":
            self.device = tvm.metal(device_id)
        elif device_name == "vulkan":
            self.device = tvm.vulkan(device_id)
        elif device_name == "rocm":
            self.device = tvm.rocm(device_id)
        elif device_name == "opencl":
            self.device = tvm.opencl(device_id)
        elif device_name == "auto":
            self.device, device_name = _detect_local_device(device_id)
            # TODO(amalyshe): add logging
            # logging.info(f"System automatically detected device: {device_name}")
        else:
            raise ValueError(device_err_msg)

        # 3. Look up model_path
        self.model_path, self.config_file_path = _get_model_path(model_config.model)

        # 4. Instantiate chat_config
        self.chat_config = _get_chat_config(self.config_file_path, None)

        self.lib_path = _get_lib_module_path(
            model_config.model, self.model_path, self.chat_config, model_config.lib_path, device_name, self.config_file_path
        )

        self.const_params = _load_params(self.model_path, self.device)
        ex = tvm.runtime.load_module(self.lib_path)
        self.vm = relax.VirtualMachine(ex, self.device)

        self.tot_seq_len = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        from tvm._ffi import get_global_func
        self.sample_top_p_from_prob_ = get_global_func("vm.builtin.sample_top_p_from_prob")
        self.fsample_topp_from_logits_ptr =get_global_func("vm.builtin.sample_top_p_from_logits")

        self.reset_kv_cache_func_ = get_global_func("vm.builtin.attention_kv_cache_array_clear")

    def get_model_config(self):
        return self.chat_config
    def init_kv_cache(self):
        if not self.kv_cache:
            self.kv_cache = self.vm["create_kv_cache"]()
        else:
            self.reset_kv_cache()
        return self.kv_cache

    def reset_kv_cache(self):
        self.tot_seq_len = 0
        if self.kv_cache:
            self.reset_kv_cache_func_(self.kv_cache)

    def tokenizer_encode(self, text):
        return self.tokenizer.encode(text)

    def tokenizer_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def softmax(self, input, temperature):
        return self.vm["softmax_with_temperature"](input, 
            tvm.nd.array(np.array(temperature, dtype = "float32", ndmin = 0), device=self.device))

    def sample_from_prob(self, logits, top_p):
        # ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
        # ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
        # ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
        # TODO(amalyshe): make sure that python random is good enough
        return self.sample_top_p_from_prob_(logits, top_p, random.random())


    def sample_from_logits(self, logits, temperature, top_p):
        # ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
        # ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
        # ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
        self.fsample_topp_from_logits_(logits, temperature, top_p, random.random());


    def sample_token_from_logits(self, logits, sampling_params: SamplingParams) -> int:
        if sampling_params.frequency_penalty == 1.0:
            if sampling_params.temperature >= 1e-6:
                logits = self.softmax(logits, sampling_params.temperature)
        else:
            # TODO(amalyshe): implement handling of tokens repetition 
            raise KeyError("Need to implement this branch")
            # UpdateLogitsOrProbOnCPUSync(logits)
            # ApplyRepetitionPenaltyOnCPU()
            # if sampling_params.temperature >= 1e-6:
            #     ApplySoftmaxWithTemperatureOnCPU();

        if sampling_params.temperature < 1e-6:
            next_token = self.sample_from_logits(logits, sampling_params.temperature, sampling_params.top_p):
        else:
            next_token = self.sample_from_prob(logits, sampling_params.top_p)
        return next_token

    def prefill(self, input, kv_cache, sampling_params: SamplingParams):
        seq_len_shape = tvm.runtime.ShapeTuple([len(input)])
        input = tvm.nd.array(np.array(input, dtype = "int32", ndmin = 2), device=self.device)
        logits, _ = self.vm["prefill"](
                 input, seq_len_shape, kv_cache, self.const_params
             )
        return self.sample_token_from_logits(logits, sampling_params)

    def decode(self, input, all_tokens_len, kv_cache, sampling_params: SamplingParams):
        seq_len_shape = tvm.runtime.ShapeTuple([all_tokens_len])
        input = tvm.nd.array(np.array([input], dtype = "int32", ndmin = 2), device=self.device)
        logits, _ = self.vm["decode"](
                 input, seq_len_shape, kv_cache, self.const_params
             )
        return self.sample_token_from_logits(logits, sampling_params)

def _detect_local_device(device_id: int = 0):
    """Automatically detect the local device if user does not specify.

    Parameters
    ----------
    device_id : int
        The local device id.

    Returns
    ------
    dev : Device
        The local device.
    """
    if tvm.metal().exist:
        return tvm.metal(device_id), "metal"
    if tvm.rocm().exist:
        return tvm.rocm(device_id), "rocm"
    if tvm.cuda().exist:
        return tvm.cuda(device_id), "cuda"
    if tvm.vulkan().exist:
        return tvm.vulkan(device_id), "vulkan"
    if tvm.opencl().exist:
        return tvm.opencl(device_id), "opencl"

    # TODO(amalyshe): add logging
    # logging.info(
    #     "None of the following device is detected: metal, rocm, cuda, vulkan, opencl. Switch to llvm instead."
    # )
    return tvm.cpu(device_id), "llvm"


@dataclass
class IFRequest:
    prompt: str
    prompt_token: list[int]
    decoded: str
    decoded_tokens: list[int]
    sampling_params: SamplingParams
    kv_cache: any

class MlcLLMb1Engine (InferenceEngine):
    requests: Dict[str, IFRequest] = {}
    current_id: str = ""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.model = Model(model_config)

    def add(self, requests: list[Request]) -> list[str]:
        if len(requests):
            output = []
            for r in requests:
                self.requests[r.request_id] = IFRequest (r.prompt, [], "", [], r.sampling_params, None)
                # init of sampling params by model settings if they were not initialized
                if not self.requests[r.request_id].sampling_params.temperature:
                    self.requests[r.request_id].sampling_params.temperature = self.model.get_model_config().temperature
                if not self.requests[r.request_id].sampling_params.top_p:
                    self.requests[r.request_id].sampling_params.top_p = self.model.get_model_config().top_p
                # TODO(amalyshe): there is frequency_penalty and presence_penalty in openai while in mlc-llm there is repetition_penalty
                # what does it stand for?
                if not self.requests[r.request_id].sampling_params.frequency_penalty:
                    self.requests[r.request_id].sampling_params.frequency_penalty = self.model.get_model_config().repetition_penalty

                output.append(r.request_id)
        return output

    def cancel(self, request_id: str):
        del self.requests[request_id]
        if self.current_id == request_id:
            self.model.reset_kv_cache()
            self.current_id = ""

    def step(self) -> InferenceStepResult:
        #determine the current_id
        if self.current_id == "":
            for id in self.requests:
                self.current_id = id
                break

        a = InferenceStepResult([],[])
        if self.current_id != "":
            a.outputs = []
            r = self.requests[self.current_id]
            if len(r.decoded_tokens) == 0:
                # tokenize and do a prefill
                # init kv_cache
                r.kv_cache = self.model.init_kv_cache()

                # tokenize
                r.prompt_token = self.model.tokenizer_encode(r.prompt)

                #call a prefill:
                r.decoded_tokens.append(self.model.prefill(r.prompt_token, r.kv_cache, r.sampling_params))
                # TODO(amalyshe): have a feeling that tokenizer decoder works very long time, need to measure
                r.decoded = self.model.tokenizer_decode(r.decoded_tokens)
                o = TextGenerationOutput(self.current_id, r.decoded)
                a.outputs.append(o)
            else:
                # do a decode
                r.decoded_tokens.append(self.model.decode(r.decoded_tokens[len(r.decoded_tokens) - 1], len(r.decoded_tokens) + len(r.prompt_token), r.kv_cache, r.sampling_params))

                prev_size = len(r.decoded)
                r.decoded = self.model.tokenizer_decode(r.decoded_tokens)

                # TODO(amalyshe): this must be correct de-tokinezition and taking a diff
                # can we do better? Do we need to change API?
                o = TextGenerationOutput(self.current_id, r.decoded[prev_size:len(r.decoded):])

                # TODO(amalyshe): hardcoding of stop token for LLama (==2)
                # Need to move/migrate mlc-llm chat conv_templates.cc
                if r.decoded_tokens[len(r.decoded_tokens) - 1] == 2:
                    o.finish_reason = "stop"
                elif (self.model.get_model_config().max_gen_len <= len(r.decoded_tokens) or 
                    (r.sampling_params.max_tokens and r.sampling_params.max_tokens <= len(r.decoded_tokens) )):
                    o.finish_reason = "length"
                a.outputs.append(o)
        return a
