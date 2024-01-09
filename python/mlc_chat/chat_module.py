"""The Python API for MLC chat."""
#! pylint: disable=too-many-lines
import inspect
import json
import os
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import tvm
from tvm.runtime import disco  # pylint: disable=unused-import

from mlc_chat.support import logging
from mlc_chat.support.auto_device import detect_device

from . import base as _

if TYPE_CHECKING:
    from mlc_chat.interface.openai_api import ChatMessage

# pylint: disable=line-too-long
_PYTHON_GET_STARTED_TUTORIAL_URL = "https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb"
# pylint: enable=line-too-long


logger = logging.getLogger(__name__)


@dataclass
class ConvConfig:  # pylint: disable=too-many-instance-attributes
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
    messages : Optional[List[List[str]]]
        The chat history represented as an array of string pairs in the following
        format: ``[[role_0, msg_0], [role_1, msg_1], ...]``.
    offset : Optional[int]
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
    prefix_tokens : Optional[List[int]]
        Token list prefixing the conversation.
    add_bos : Optional[bool]
        Determines whether a beginning-of-string (bos) token should be added
        before the input tokens.
    """

    name: Optional[str] = None
    system: Optional[str] = None
    roles: Optional[List[str]] = None
    messages: Optional[List[List[str]]] = None
    offset: Optional[int] = None
    separator_style: Optional[int] = None
    seps: Optional[List[str]] = None
    role_msg_sep: Optional[str] = None
    role_empty_sep: Optional[str] = None
    stop_str: Optional[str] = None
    stop_tokens: Optional[List[int]] = None
    prefix_tokens: Optional[List[int]] = None
    add_bos: Optional[bool] = None

    def __post_init__(self):
        if self.messages is not None and self.offset is None:
            self.offset = len(self.messages)


@dataclass
class ChatConfig:  # pylint: disable=too-many-instance-attributes
    r"""A dataclass that represents user-defined partial configuration for the
    chat config file.

    An instance of ``ChatConfig`` can be passed in to the instantiation of a
    :class:`mlc_chat.ChatModule` instance to override the default setting in
    ``mlc-chat-config.json`` under the model folder.

    Since the configuration is partial, everything will be ``Optional``.

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
        The approximated average number of generated tokens in each round. Used
        to determine whether the maximum window size would be exceeded.
    max_gen_len : Optional[int]
        The maximum number of tokens to be generated in each round. Would simply
        stop generating after this number is exceeded.
    shift_fill_factor : Optional[float]
        The fraction of maximum window size to shift when it is exceeded.
    tokenizer_files : Optional[List[str]]
        List of tokenizer files of the model.
    conv_config : Optional[ConvConfig]
        The partial overriding configuration for conversation template. Will first
        load the predefined template with the name specified in ``conv_template``
        and then override some of the configurations specified in ``conv_config``.
    model_category : Optional[str]
        The category of the model's architecture (e.g. ``llama``, ``gpt_neox``, ``rwkv``).
    model_name : Optional[str]
        Name of the model (e.g. ``Llama-2-7b-chat-hf``).
    tensor_parallel_shards : Optional[str]
        Tensor parallel degree.
    use_presharded_weights : Optional[bool]
        If True, the weights were saved with sharding already applied.
    context_window_size : Optional[int]
        Maximum kv cache window size.
    prefill_chunk_size: Optional[int]
        (Experimental) The chunk size during prefilling. By default,
        the chunk size is the same as sliding window or max sequence length.
        This flag subjects to future refactoring.
    attention_sink_size : Optional[int]
        (Experimental) The number of stored sinks. Only supported on Mistral yet. By default,
        the number of sinks is 4. This flag subjects to future refactoring.
    sliding_window_size : Optional[int]
        (Experimental) The sliding window size in sliding window attention (SWA).
        This optional field overrides the `sliding_window_size` in config.json for
        those models that use SWA. Currently only useful when compiling Mistral.
        This flag subjects to future refactoring.
    opt : Optional[str]
        Optimization flags. MLC LLM maintains a predefined set of optimization flags,
        denoted as O0, O1, O2, O3, where O0 means no optimization, O2 means majority of them,
        and O3 represents extreme optimization that could potentially break the system.
        Meanwhile, optimization flags could be explicitly specified via details knobs, e.g.
        --opt="cublas_gemm=1;cudagraph=0".
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
    tensor_parallel_shards: Optional[int] = None
    use_presharded_weights: Optional[bool] = None
    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    opt: Optional[str] = None

    @classmethod
    def _from_json(cls, json_obj: dict):
        return cls(**{k: v for k, v in json_obj.items() if k in inspect.signature(cls).parameters})


@dataclass
class GenerationConfig:  # pylint: disable=too-many-instance-attributes
    r"""A dataclass that represents user-defined generation configuration.

    An instance of ``GenerationConfig`` can be passed in to the generate function
    of a :class:`mlc_chat.ChatModule` instance to override the default generation
    setting in ``mlc-chat-config.json`` and ``ChatConfig`` under the model folder.

    Once the generation ends, ``GenerationConfig`` is discarded, since the values
    will only override the ``ChatConfig`` generation settings during one generation,
    unless it is recurrently passed to generate function. This allows changing generation
    settings over time, without overriding ``ChatConfig`` permanently.

    Since the configuraiton is partial, everything will be ``Optional``.

    Parameters
    ----------
    temperature : Optional[float]
        The temperature applied to logits before sampling. The default value is
        ``0.7``. A higher temperature encourages more diverse outputs, while a
        lower temperature produces more deterministic outputs.
    presence_penalty : Optional[float]
        Number between -2.0 and 2.0. Positive values penalize new tokens based on
        whether they appear in the text so far, increasing the model's likelihood
        to talk about new topics. Negative values can increase the likelihood of
        repetition.
    frequency_penalty : Optional[float]
        Number between -2.0 and 2.0. Positive values penalize new tokens based on their
        existing frequency in the text so far, decreasing the model's likelihood to
        repeat the same line verbatim. Negative values can increase the likelihood of
        repetition.
    repetition_penalty : Optional[float]
        The repetition penalty controls the likelihood of the model generating
        repeated texts. The default value is set to ``1.0``, indicating that no
        repetition penalty is applied. Increasing the value reduces the
        likelihood of repeat text generation. However, setting a high
        ``repetition_penalty`` may result in the model generating meaningless
        texts. The ideal choice of repetition penalty may vary among models. Only
        Active when presence_penalty and frequency_penalty are both 0.0.

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
        The approximated average number of generated tokens in each round. Used
        to determine whether the maximum window size would be exceeded.
    max_gen_len : Optional[int]
        This parameter determines the maximum length of the generated text. If it is
        not set, the model will generate text until it encounters a stop token.
    n : Optional[int]
        This parameter determines the number of text samples to generate. The default
        value is ``1``. Note that this parameter is only used when ``stream`` is set to
        ``False``.
    stop : Optional[Union[str, List[str]]]
        When ``stop`` is encountered, the model will stop generating output.
        It can be a string or a list of strings. If it is a list of strings, the model
        will stop generating output when any of the strings in the list is encountered.
        Note that this parameter does not override the default stop string of the model.
    """

    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    mean_gen_len: Optional[int] = None
    max_gen_len: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = None  # pylint: disable=invalid-name
    stop: Optional[Union[str, List[str]]] = None

    @classmethod
    def _from_chat_config(cls, chat_config_obj: ChatConfig):
        return cls(
            **{
                f.name: getattr(chat_config_obj, f.name)
                for f in fields(chat_config_obj)
                if f.name in inspect.signature(cls).parameters
            }
        )


class PlaceInPrompt(Enum):
    """The place of an input message in a prompt."""

    # The input message should have role names and corresponding seperators appended both prior to
    # it and after it, making it a complete prompt.
    All = 0  # pylint: disable=invalid-name
    # The input message is only the beginning part of a prompt, no role name and separator should
    # be appended after the message since there will be future messages appended after the message.
    Begin = 1  # pylint: disable=invalid-name
    # The input message is in the middle of a prompt, nothing should be appended before or after
    # the message.
    Middle = 2  # pylint: disable=invalid-name
    # The input message is the ending part of a prompt, no role name and separator should be
    # appended prior to it since the message is concatenated to some prior messages.
    End = 3  # pylint: disable=invalid-name


def _get_model_path(model: str) -> Tuple[str, str]:
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
    if model.startswith("HF://"):
        from mlc_chat.support.download import (  # pylint: disable=import-outside-toplevel
            download_mlc_weights,
        )

        logger.info("Downloading model from HuggingFace: %s", model)
        mlc_dir = download_mlc_weights(model)
        cfg_dir = mlc_dir / "mlc-chat-config.json"
        return str(mlc_dir), str(cfg_dir)

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
            logger.info("Using model folder: %s", os.path.abspath(candidate))
            logger.info("Using mlc chat config: %s", os.path.abspath(chat_file))
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
        raise FileNotFoundError(
            "The model folder provided does not seem to refer to a valid mlc-llm model folder.\n"
            "Specifically, we cannot find `mlc-chat-config.json`, a required file. You should "
            "provide a path that contains the file.\n"
            "According to your input `model`, we looked at folder(s):\n"
            f"{valid_dir_str}"
            "MLC-Chat consumes models that are processed by the MLC-LLM build process.\n"
            f"Please checkout {_PYTHON_GET_STARTED_TUTORIAL_URL} for an example on "
            "how to load a model."
        )
    # Error 2: cannot find a folder (E0)
    all_paths_str = "".join(f"- {path}\n" for path in candidate_paths)
    raise FileNotFoundError(
        "Cannot find the model folder. We searched over the following possible paths:\n"
        f"{all_paths_str}"
        "You can try to pass in `model=/path/to/your-model-path`, and confirm "
        "that it contains `mlc-chat-config.json`, among other essential files.\n"
        f"Please checkout {_PYTHON_GET_STARTED_TUTORIAL_URL} for an "
        "example on how to load a model."
    )


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
    with open(config_file_path, mode="rt", encoding="utf-8") as file:
        json_object = json.load(file)
        final_chat_config = ChatConfig._from_json(json_object)  # pylint: disable=protected-access
    if user_chat_config is not None:
        # We override using user's chat config
        for field in fields(user_chat_config):
            field_name = field.name
            field_value = getattr(user_chat_config, field_name)
            if field_value is not None:
                if field_name == "model_lib":
                    warn_msg = (
                        'WARNING: Do not override "model_lib" in ChatConfig. '
                        "This override will be ignored. Please use ChatModule.model_lib_path to "
                        "override the full model library path instead."
                    )
                    warnings.warn(warn_msg)
                else:
                    setattr(final_chat_config, field_name, field_value)
    return final_chat_config


def _get_generation_config(
    user_chat_config: ChatConfig, user_generation_config: Optional[GenerationConfig]
) -> GenerationConfig:
    """Read in the config file in model path, then potentially override with user input.

    Parameters
    ----------
    user_chat_config : ChatConfig
        ``ChatConfig`` that contain the generation settings to be overriden.
    user_generation_config : Optional[GenerationConfig]
        User's input, a partial ``GenerationConfig`` to override the ``ChatConfig``.

    Returns
    ------
    final_generation_config : GenerationConfig
        ``GenerationConfig`` corresponding to ``user_chat_config``, overriden by
        ``user_generation_config``.
    """
    # pylint: disable=protected-access
    final_generation_config = GenerationConfig._from_chat_config(user_chat_config)
    # pylint: enable=protected-access
    if user_generation_config is not None:
        # We override using user's chat config
        for field in fields(user_generation_config):
            field_name = field.name
            field_value = getattr(user_generation_config, field_name)
            if field_value is not None:
                setattr(final_generation_config, field_name, field_value)
    return final_generation_config


def _get_lib_module_path(  # pylint: disable=too-many-arguments
    model: str,
    model_path: str,
    chat_config: ChatConfig,
    model_lib_path: Optional[str],
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
    model_lib_path : Optional[str]
        User's input. Supposedly a full path to model library. Prioritized to use.
    device_name : str
        User's input. Used to construct the library model file name.
    config_file_path : str
        The path to ``mlc-chat-config.json``. Used for error message making.

    Returns
    -------
    model_lib_path : str
        The path pointing to the model library we find.

    Raises
    ------
    FileNotFoundError: if we cannot find a valid model library file.
    """
    # 1. Use user's model_lib_path if provided
    if model_lib_path is not None:
        if os.path.isfile(model_lib_path):
            logger.info("Using library model: %s", model_lib_path)
            return model_lib_path
        raise FileNotFoundError(
            f"The `model_lib_path` you passed in is not a file: {model_lib_path}.\n"
            f"Please refer to {_PYTHON_GET_STARTED_TUTORIAL_URL} as tutorial on model loading."
        )

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
            logger.info("Using library model: %s", os.path.abspath(candidate))
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
        "consider passing in the `ChatModule.model_lib_path` parameter.\n"
        f"Please checkout {_PYTHON_GET_STARTED_TUTORIAL_URL} for an example "
        "on how to load a model."
    )
    raise FileNotFoundError(err_msg)


def _convert_chat_config_to_json_str(
    chat_config: Optional[ChatConfig], conv_template: Optional[str]
) -> str:
    """Convert user's input ChatConfig to a json string, omitting ``None`` fields.

    Parameters
    ----------
    chat_config : Optional[ChatConfig]
        User's input. A partial ChatConfig for overriding ``mlc-chat-config.json``.
    conv_template : Optional[str]
        The ``conv_template`` that will be used after considering potential override.

    Returns
    ------
    json_str : str
        A JSON string that corresponds to user's ``chat_config`` input.
        Returns "" if ``chat_config`` unspecified.
    """
    if chat_config is None:
        return ""
    # Current logic does not allow partial ChatConfig without specifying the
    # conv_template. Hence we use the conv_template after considering potential overrides.
    chat_config.conv_template = conv_template
    # Only want to keep entries that are not None; otherwise, we would override things to None
    assert hasattr(ChatConfig, "conv_config")  # in case dataclass attribute name changes
    chat_dict = {}
    for key, value in asdict(chat_config).items():
        if key == "conv_config" and value is not None:
            # conv template is another dict, do the same thing
            conv_dict = {}
            for conv_k, conv_v in value.items():
                if conv_v is not None:
                    conv_dict[conv_k] = conv_v
            chat_dict[key] = conv_dict
            continue
        if value is not None:
            chat_dict[key] = value

    return json.dumps(chat_dict)


def _convert_generation_config_to_json_str(generation_config: Optional[GenerationConfig]) -> str:
    """Convert user's input GenerationConfig to a json string.

    Parameters
    ----------
    generation_config : Optional[GenerationConfig]
        User's input. A partial GenerationConfig for overriding ChatConfig generation settings.

    Returns
    ------
    json_str : str
        A JSON string that corresponds to user's ``generation_config`` input.
        Returns "" if ``generation_config`` unspecified.
    """
    if generation_config is None:
        return ""
    return json.dumps(asdict(generation_config))


def _inspect_model_lib_metadata_memory_usage(model_lib_path):
    cmd = [
        sys.executable,
        "-m",
        "mlc_chat.cli.model_metadata",
        model_lib_path,
        "--memory-only",
    ]
    subprocess.run(cmd, check=False)


class ChatModule:  # pylint: disable=too-many-instance-attributes
    r"""The ChatModule for MLC LLM.

    Examples
    --------

    .. code:: python

        from mlc_chat import ChatModule
        from mlc_chat.callback import StreamToStdout

        # Create a ChatModule instance
        cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")

        # Generate a response for a given prompt
        output = cm.generate(
            prompt="What is the meaning of life?",
            progress_callback=StreamToStdout(callback_interval=2),
        )

        # Print prefill and decode performance statistics
        print(f"Statistics: {cm.stats()}\n")

        output = cm.generate(
            prompt="How many points did you list out?",
            progress_callback=StreamToStdout(callback_interval=2),
        )


    Parameters
    ----------
    model: str
        The model folder after compiling with MLC-LLM build process. The parameter
        can either be the model name with its quantization scheme
        (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
        folder. In the former case, we will use the provided name to search
        for the model folder over possible paths.

    device : str
        The description of the device to run on. User should provide a string in the
        form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
        'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
        local device), and 'device_id' is the device id to run on. If no 'device_id'
        is provided, it will be set to 0 by default.

    chat_config : Optional[ChatConfig]
        A ``ChatConfig`` instance partially filled. Will be used to override the
        ``mlc-chat-config.json``.

    model_lib_path : Optional[str]
        The full path to the model library file to use (e.g. a ``.so`` file).
        If unspecified, we will use the provided ``model`` to search over
        possible paths.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        device: str = "auto",
        chat_config: Optional[ChatConfig] = None,
        model_lib_path: Optional[str] = None,
    ):
        # 0. Get device:
        # Retrieve device_name and device_id (if any, default 0) from device arg
        self.device = detect_device(device)
        device_type = self.device.device_type
        device_id = self.device.device_id

        # 1. Populate chat module and their functions
        fcreate_chat_mod = tvm.get_global_func("mlc.llm_chat_create")
        assert fcreate_chat_mod is not None
        chat_mod = fcreate_chat_mod(device_type, device_id)

        # chat module related functions
        self._reload_func = chat_mod["reload"]
        self._unload_func = chat_mod["unload"]
        self._prefill_func = chat_mod["prefill"]
        self._embed_func = chat_mod["embed"]
        self._prefill_with_embed_func = chat_mod["prefill_with_embed"]
        self._decode_func = chat_mod["decode"]
        self._raw_generate_func = chat_mod["raw_generate"]
        self._reset_chat_func = chat_mod["reset_chat"]
        self._load_json_override_func = chat_mod["load_json_override"]
        self._stopped_func = chat_mod["stopped"]
        self._get_message_func = chat_mod["get_message"]
        self._runtime_stats_text_func = chat_mod["runtime_stats_text"]
        self._verbose_runtime_stats_text_func = chat_mod["verbose_runtime_stats_text"]
        self._reset_runtime_stats_func = chat_mod["reset_runtime_stats"]
        self._get_config_json_func = chat_mod["get_config_json"]
        self._process_system_prompts_func = chat_mod["process_system_prompts"]
        self._evaluate_func = chat_mod["evaluate"]
        self._get_role0_func = chat_mod["get_role0"]
        self._get_role1_func = chat_mod["get_role1"]

        # 2. Look up model_path
        self.model_path, self.config_file_path = _get_model_path(model)

        # 3. Instantiate chat_config
        self.chat_config = _get_chat_config(self.config_file_path, chat_config)

        # 4. Look up model library
        try:
            self.model_lib_path = _get_lib_module_path(
                model,
                self.model_path,
                self.chat_config,
                model_lib_path,
                self.device.MASK2STR[self.device.device_type],
                self.config_file_path,
            )
        except FileNotFoundError:
            logger.info("Model lib not found. Now compiling model lib on device...")
            from mlc_chat.interface import (  # pylint: disable=import-outside-toplevel
                jit,
            )

            self.model_lib_path = str(
                jit.jit(
                    model_path=Path(self.model_path),
                    chat_config=asdict(self.chat_config),
                    device=self.device,
                )
            )
        _inspect_model_lib_metadata_memory_usage(self.model_lib_path)

        # 5. Call reload
        user_chat_config_json_str = _convert_chat_config_to_json_str(
            self.chat_config, self.chat_config.conv_template
        )
        self._reload(self.model_lib_path, self.model_path, user_chat_config_json_str)

    def generate(
        self,
        prompt: Union[str, List["ChatMessage"]],
        generation_config: Optional[GenerationConfig] = None,
        progress_callback=None,
        stateless=False,
    ) -> Union[str, List[str]]:
        r"""A high-level method that returns the full response from the chat module given a user
        prompt. User can optionally specify which callback method to use upon receiving the
        response. By default, no callback will be applied.

        Parameters
        ----------
        prompt: Union[str, List[ChatMessage]]
            The user input prompt, i.e. a question to ask the chat module.
            It can also be the whole conversation history (list of messages with role and content)
            eg:

            .. code::

                [
                    ChatMessage(role="user", content="Hello, how are you?"),
                    ChatMessage(role="assistant", content="I'm fine, thank you. How about you?"),
                    ChatMessage(role="user", content="I'm good too."),
                ]
        generation_config: Optional[GenerationConfig]
            The generation config object to override the ChatConfig generation settings.
        progress_callback: object
            The optional callback method used upon receiving a newly generated message from the
            chat module. See `mlc_chat/callback.py` for a full list of available callback classes.
            Currently, only streaming to stdout callback method is supported, see `Examples` for
            more detailed usage.

        Returns
        -------
        output : string
            The generated full output from the chat module.

        Examples
        --------
        .. code-block:: python

          # Suppose we would like to stream the response of the chat module to stdout
          # with a refresh interval of 2. Upon calling generate(), We will see the response of
          # the chat module streaming to stdout piece by piece, and in the end we receive the
          # full response as a single string `output`.

          from mlc_chat import ChatModule, GenerationConfig, callback
          cm = ChatModule(xxx)
          prompt = "what's the color of banana?"
          output = cm.generate(
            prompt, GenerationConfig(temperature=0.8), callback.StreamToStdout(callback_interval=2)
          )
          print(output)
        """
        new_msgs = []
        num_return_sequences = 1
        return_str = True
        if (generation_config is not None) and (generation_config.n is not None):
            num_return_sequences = generation_config.n
            return_str = False

        for _ in range(num_return_sequences):
            if stateless:
                self.reset_chat()
            self._prefill(prompt, generation_config=generation_config)

            if not progress_callback:
                while not self._stopped():
                    self._decode(generation_config=generation_config)
                new_msg = self._get_message()
                new_msgs.append(new_msg)
            else:
                # apply callback with a rate of callback_interval
                i, new_msg = 0, ""
                while not self._stopped():
                    self._decode(generation_config=generation_config)
                    if i % progress_callback.callback_interval == 0 or self._stopped():
                        new_msg = self._get_message()
                        progress_callback(new_msg)
                    i += 1
                progress_callback(stopped=True)
                new_msgs.append(new_msg)
        return new_msgs[0] if return_str else new_msgs

    def reset_chat(self, chat_config: Optional[ChatConfig] = None):
        r"""Reset the chat session, clear all chat history, and potentially
        override the original `mlc-chat-config.json`.

        Parameters
        ----------
        chat_config : Optional[ChatConfig]
            A ``ChatConfig`` instance partially filled. If specified, the chat
            module will reload the `mlc-chat-config.json`, and override it with
            ``chat_config``, just like in initialization.

        Note
        ----
        The model remains the same after :func:`reset_chat`.
        To reload module, please either re-initialize a :class:`ChatModule` instance
        or use :func:`_reload` instead.
        """
        self._reset_chat_func()
        if chat_config is not None:
            # Redo the overriding
            self.chat_config = _get_chat_config(self.config_file_path, chat_config)
            user_chat_config_json_str = _convert_chat_config_to_json_str(
                chat_config, self.chat_config.conv_template
            )
            # Second argument is `partial_update = True`
            self._load_json_override_func(user_chat_config_json_str, True)

    def embed_text(self, input: str):  # pylint: disable=redefined-builtin
        r"""Given a text input, returns its embedding in the LLM.

        Parameters
        ----------
        input : str
            The user input string.

        Returns
        -------
        embedding : tvm.runtime.NDArray
            The embedding of the text.

        Note
        ----
        This is a high-level method and is only used for retrieving text embeddings. Users are
        not supposed to call :func:`generate` after calling this method in the same chat session,
        since the input to this method is not prefilled and will cause error. If user needs to
        call :func:`generate` later, please call :func:`reset_chat` first.
        For a more fine-grained embedding API, see :func:`_embed`.
        """
        return self._embed_func(input, PlaceInPrompt.Middle.value)

    def stats(self, verbose=False) -> str:
        r"""Get the runtime stats of the encoding step, decoding step (and embedding step if exists)
        of the chat module in text form.

        Returns
        -------
        stats : str
            The runtime stats text.
        """
        if verbose:
            return self._verbose_runtime_stats_text_func()
        return self._runtime_stats_text_func()

    def benchmark_generate(self, prompt: str, generate_length: int) -> str:
        r"""Controlled generation with input prompt and fixed number of
        generated tokens, ignoring system prompt. For example,

        .. code:: python

            from mlc_chat import ChatModule

            cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")
            output = cm.benchmark_generate("What's the meaning of life?", generate_length=256)
            print(f"Generated text:\n{output}\n")
            print(f"Statistics: {cm.stats()}")

        will generate 256 tokens in total based on prompt "What's the meaning
        of life?". After generation, you can use `cm.stats()` to print the
        generation speed.

        Notes
        -----
        1. This function is typically used in controlled benchmarks. It generates
        text without system prompt (i.e., it is pure text generation with no chat
        style) and ignores the token stop model(s).
        2. To make the benchmark as accurate as possible, we first do a round of
        warmup prefill and decode before text generation.
        3. This function resets the previous performance statistics.

        Parameters
        ----------
        prompt : str
            The prompt of the text generation.

        generate_length : int
            The target length of generation.

        Returns
        -------
        output : str
            The generated text output.
        """
        if generate_length < 0:
            raise ValueError(
                "The generation length is expected to be non-negative, "
                f"while the given length is {generate_length}"
            )

        # warmup run
        self.reset_chat()
        self._prefill(prompt)
        self._decode()

        return self._raw_generate_func(prompt, generate_length)

    def _reload(self, lib: str, model_path: str, app_config_json: str = ""):
        r"""Reload the chat module from the given library and model path.

        Parameters
        ----------
        lib : str
            The library path.
        model_path : str
            The model path.
        app_config_json: str
            The partial config that is used to partially override the model configuration.
        """
        self._reload_func(lib, model_path, app_config_json)

    def _unload(self):
        r"""Unload the chat module and clear memory of all loaded models."""
        self._unload_func()

    def _prefill(
        self,
        input: Union[str, List["ChatMessage"]],  # pylint: disable=redefined-builtin
        decode_next_token: bool = True,
        place_in_prompt: PlaceInPrompt = PlaceInPrompt.All,
        generation_config: Optional[GenerationConfig] = None,
    ):
        r"""Run prefill stage for a given input and optionally decode the first output token.
        User can decide where to place the input in the prompt.

        Parameters
        ----------
        input : Union[str, List[ChatMessage]]
            The user input prompt, i.e. a question to ask the chat module.
            It can also be the whole conversation history (list of messages with role and content)
            eg:

            .. code::

                [
                    ChatMessage(role="user", content="Hello, how are you?"),
                    ChatMessage(role="assistant", content="I'm fine, thank you. How about you?"),
                    ChatMessage(role="user", content="I'm good too."),
                ]
        decode_next_token : bool
            Whether to decode the next token after prefilling.
        place_in_prompt: PlaceInPrompt
            The place of the input message in the prompt. See `class PlaceInPrompt` for details.
        generation_config: Optional[GenerationConfig]
            The generation config to override the ChatConfig generation settings.
        """
        generation_config = _get_generation_config(self.chat_config, generation_config)
        generation_config_str = _convert_generation_config_to_json_str(generation_config)

        if isinstance(input, list):
            # Populate conversation.messages using load_json_override
            if len(input) > 1:
                conv_config = json.loads(self._get_config_json())["conv_config"]
                messages = []
                role0 = self._get_role_0()
                role1 = self._get_role_1()
                for _, msg in enumerate(input[:-1]):
                    role = msg.role
                    content = msg.content
                    if role == "user":
                        messages.append([role0, content])
                    elif role == "assistant":
                        messages.append([role1, content])
                    else:
                        raise ValueError("Only user and assistant roles are supported.")
                if not input[-1].role == "user":
                    raise ValueError("Last message should be from user.")
                conv_config["messages"] = messages
                conv_config["offset"] = 0
                # Otherwise, the offset will be set to the length of the conversation,
                # which means history will be retained even after calling reset_chat
                self._load_json_override(
                    json.dumps({"conv_config": conv_config}),
                    partial_update=True,
                )
            input_str = input[-1].content
        else:
            input_str = input

        self._prefill_func(
            input_str, decode_next_token, place_in_prompt.value, generation_config_str
        )

    def _embed(
        self,
        input: str,  # pylint: disable=redefined-builtin
        place_in_prompt: PlaceInPrompt = PlaceInPrompt.All,
        generation_config: Optional[GenerationConfig] = None,
    ):
        r"""A more fine-grained embedding API. Given a text input, get the embedding of the
        tokenized prompt. User can decide where to place the input in the prompt. This functionality
        usually aids the subsequent call to :func:`_prefill_with_embed`.

        Parameters
        ----------
        input : str
            The user input string.
        place_in_prompt: PlaceInPrompt
            The place of the input message in the prompt. See `class PlaceInPrompt` for details.
        generation_config: Optional[GenerationConfig]
            The generation config to override the ChatConfig generation settings.

        Returns
        -------
        embedding : tvm.runtime.NDArray
            The embedding of the text.
        """
        generation_config = _get_generation_config(self.chat_config, generation_config)
        generation_config_str = _convert_generation_config_to_json_str(generation_config)

        return self._embed_func(input, place_in_prompt.value, generation_config_str)

    def _prefill_with_embed(
        self,
        embedding: tvm.runtime.NDArray,
        decode_next_token: bool = True,
        generation_config: Optional[GenerationConfig] = None,
    ):
        r"""Given an embedding, run the prefill stage and optionally decode the first output token.

        Parameters
        ----------
        embedding : tvm.runtime.NDArray
            The embedding of user input.
        decode_next_token : bool
            Whether to decode the next token after prefilling.
        generation_config: Optional[GenerationConfig]
            The generation config to override the ChatConfig generation settings.
        """
        generation_config = _get_generation_config(self.chat_config, generation_config)
        generation_config_str = _convert_generation_config_to_json_str(generation_config)

        self._prefill_with_embed_func(embedding, decode_next_token, generation_config_str)

    def _decode(self, generation_config: Optional[GenerationConfig] = None):
        r"""Decode the next token, the decoding result is stored in a buffer and
        can be retrieved by :func:`get_message`.

        Parameters
        ----------
        generation_config: Optional[GenerationConfig]
            The generation config to override the ChatConfig generation settings.
        """
        generation_config = _get_generation_config(self.chat_config, generation_config)
        generation_config_str = _convert_generation_config_to_json_str(generation_config)
        self._decode_func(generation_config_str)

    def _stopped(self) -> bool:
        r"""Check if the stop condition is met for the current round.

        Returns
        -------
        stopped : bool
        """
        return self._stopped_func() != 0

    def _get_message(self) -> str:
        r"""Get the output message in the current round.

        Returns
        -------
        message : str

        Note
        ----
        This function returns the message that corresponds to
        all the tokens decoded so far.
        """
        return self._get_message_func()

    def _get_config_json(self):
        r"""Get the configuration of the chat module in a single json string.

        Returns
        -------
        config : str
            The config json string.
        """
        return self._get_config_json_func()

    def _load_json_override(self, config_str: str, partial_update: bool = False):
        r"""Load JSON config and override existing configurations for the chat module.

        Parameters
        ----------
        config_str : str
            A json config string that partially specifies some of the options.
        partial_update : bool
            Whether it's a partial update or full update. If set to true, we perform a partial
            update on some of the provided options; if set to false, all options must be provided.
        """
        self._load_json_override_func(config_str, partial_update)

    def _get_role_0(self):
        r"""Get the name of role 0 in the conversation.

        Returns
        -------
        name : str
            The name of role 0.
        """
        return self._get_role0_func()

    def _get_role_1(self):
        r"""Get the name of role 1 in the conversation.

        Returns
        -------
        name : str
            The name of role 1.
        """
        return self._get_role1_func()

    def _reset_runtime_stats(self):
        r"""Reset the runtime stats, clear all performance history."""
        self._reset_runtime_stats_func()

    def _process_system_prompts(self):
        r"""Pre-process by prefilling the system prompts, running prior to any user input."""
        self._process_system_prompts_func()
