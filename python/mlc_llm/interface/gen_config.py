"""Generator of mlc-chat-config.json and tokenizer configuration."""

import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.model import Model
from mlc_llm.quantization import Quantization
from mlc_llm.support import convert_tiktoken, logging
from mlc_llm.support.style import bold, green, red

from .compiler_flags import ModelConfigOverride

logger = logging.getLogger(__name__)

FOUND = green("Found")
NOT_FOUND = red("Not found")
FAILED = red("Failed")
VERSION = "0.1.0"


@dataclasses.dataclass
class MLCChatConfig:  # pylint: disable=too-many-instance-attributes
    """Fields in the dumped `mlc-chat-config.json` file."""

    model_type: str
    quantization: str
    model_config: Dict[str, Any]
    vocab_size: int
    context_window_size: int
    sliding_window_size: int
    prefill_chunk_size: int
    attention_sink_size: int
    tensor_parallel_shards: int
    # Control the behavior of the runtime
    mean_gen_len: int = None
    max_gen_len: int = None
    shift_fill_factor: float = None
    # Configuration of text generation
    temperature: float = None
    presence_penalty: float = None
    frequency_penalty: float = None
    repetition_penalty: float = None
    top_p: float = None
    # Conversation template
    conv_template: Union[str, Dict[str, Any]] = None
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None
    # Tokenizer configuration
    tokenizer_files: List[str] = dataclasses.field(default_factory=list)
    # The method to post-process the token table. See
    # cpp/tokenizers.h::Tokenizer::PostProcessTokenTable for details
    token_table_postproc_method: Literal["byte_fallback", "byte_level"] = None
    # Version control
    version: str = VERSION

    def apply_defaults(self) -> None:
        """Apply system default value."""
        defaults = {
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "temperature": 0.7,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "top_p": 0.95,
            "mean_gen_len": 128,
            "max_gen_len": 512,
            "shift_fill_factor": 0.3,
        }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
                logger.info("[System default] Setting %s: %s", bold(key), value)


def check_string(s: str) -> bool:
    """Check whether it's a string."""
    delimit = s[1]
    if s[0] != "b" or s[-1] != delimit:
        return False
    for i in range(2, len(s) - 1):
        if s[i] == delimit and s[i - 1] != "\\":
            return False
    return True


def txt2rwkv_tokenizer(vocab: Path, out: Path) -> None:
    """Generate tokenizer_model from RWKV vocab file."""
    idx2token = {}

    with vocab.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for l in lines:
        idx = int(l[: l.index(" ")])
        raw = l[l.index(" ") : l.rindex(" ")].strip()
        if check_string(raw):
            x = eval(raw)  # pylint: disable=eval-used
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            idx2token[idx] = x
        else:
            raise ValueError("Unsupported vocab dictionary")

    with (out / "tokenizer_model").open("wb") as f:
        import msgpack  # pylint: disable=import-outside-toplevel,import-error

        msgpack.pack(idx2token, f)


def json2rwkv_tokenizer(vocab: Path, out: Path) -> None:
    """Generate tokenizer_model from RWKV vocab file."""
    idx2token = {}

    with vocab.open("r", encoding="utf-8") as f:
        data = json.load(f)
        for key, value in data.items():
            x = key.encode("utf-8") if isinstance(key, str) else key
            assert isinstance(x, bytes)
            idx2token[int(value)] = x

    with (out / "tokenizer_model").open("wb") as f:
        import msgpack  # pylint: disable=import-outside-toplevel,import-error

        msgpack.pack(idx2token, f)


def detect_token_table_postproc_method(output_path: Path) -> Literal["byte_fallback", "byte_level"]:
    """Detect the token table postprocessing method from tokenizer.json that is found under
    output_path. If not detected, use ByteFallback as default.

    Check the decoder field of the tokenizer. If it uses ByteFallback decoder, return
    "byte_fallback". If it uses ByteLevel decoder, return "byte_level". Otherwise, use
    ByteFallback as default.

    See also cpp/tokenizers.h::Tokenizer::PostProcessTokenTable.
    """
    output_tokenizer_path = output_path / "tokenizer.json"
    if not output_tokenizer_path.exists():
        logger.warning(
            "Tokenizer token table postprocessing method is not detected as tokenizer.json "
            "is not found, use ByteFallback (the same as LLaMA/LLaMA2) by default"
        )
        return "byte_fallback"

    with output_tokenizer_path.open("r", encoding="utf-8") as in_file:
        tokenizer_json = json.load(in_file)

    # Find all decoders in tokenizer.json
    decoders = []

    if "decoder" not in tokenizer_json:
        logger.warning(
            "Decoder field is not found in tokenizer.json, use ByteFallback (the same as "
            "LLaMA/LLaMA2) as the token table postprocessing method by default"
        )
        return "byte_fallback"

    decoders_json = tokenizer_json["decoder"]
    assert "type" in decoders_json, "Decoder type is not specified in tokenizer.json"
    if decoders_json["type"] == "Sequence":
        assert "decoders" in decoders_json
        decoders = decoders_json["decoders"]
    else:
        decoders = [decoders_json]

    is_byte_level = False
    is_byte_fallback = False

    for decoder in decoders:
        if decoder["type"] == "ByteLevel":
            is_byte_level = True
        if decoder["type"] == "ByteFallback":
            is_byte_fallback = True
    assert not (
        is_byte_level and is_byte_fallback
    ), "Tokenizer decoder cannot have both type ByteLevel and type ByteFallback"

    if is_byte_level:
        return "byte_level"
    if is_byte_fallback:
        return "byte_fallback"

    logger.warning(
        "Neither ByteLevel nor ByteFallback decoder is detected in tokenizer.json, use "
        "ByteFallback (the same as LLaMA/LLaMA2) as the token table postprocessing method "
        "by default"
    )
    return "byte_fallback"


def gen_config(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
    config: Path,
    model: Model,
    quantization: Quantization,
    conv_template: str,
    context_window_size: Optional[int],
    sliding_window_size: Optional[int],
    prefill_chunk_size: Optional[int],
    attention_sink_size: Optional[int],
    tensor_parallel_shards: Optional[int],
    max_batch_size: int,
    output: Path,
):
    """Entrypoint of MLC Chat configuration generation."""
    # Step 1. Initialize `mlc-chat-config.json` using `config.json`
    conversation_reg = ConvTemplateRegistry.get_conv_template(conv_template)
    if conversation_reg is None:
        logger.warning(
            "%s: Conversation template is not registered in ConvTemplateRegistry: %s",
            red("Warning"),
            conv_template,
        )
        conversation = conv_template  # type: ignore
    else:
        conversation = conversation_reg.to_json_dict()  # type: ignore

    model_config = ModelConfigOverride(
        context_window_size=context_window_size,
        sliding_window_size=sliding_window_size,
        prefill_chunk_size=prefill_chunk_size,
        attention_sink_size=attention_sink_size,
        max_batch_size=max_batch_size,
        tensor_parallel_shards=tensor_parallel_shards,
    ).apply(model.config.from_file(config))
    mlc_chat_config = MLCChatConfig(
        model_type=model.name,
        quantization=quantization.name,
        model_config=model_config.asdict(),
        vocab_size=model_config.vocab_size,
        context_window_size=getattr(model_config, "context_window_size", -1),
        sliding_window_size=getattr(model_config, "sliding_window_size", -1),
        prefill_chunk_size=model_config.prefill_chunk_size,
        attention_sink_size=getattr(model_config, "attention_sink_size", -1),
        tensor_parallel_shards=model_config.tensor_parallel_shards,
        conv_template=conversation,
    )
    # Step 2. Load `generation_config.json` and `config.json` for text-generation related configs
    for generation_config_filename in ["generation_config.json", "config.json"]:
        generation_config = config.parent / generation_config_filename
        if generation_config.exists():
            with generation_config.open("r", encoding="utf-8") as in_file:
                generation_config_json = json.load(in_file)
            for key, value in generation_config_json.items():
                if hasattr(mlc_chat_config, key) and getattr(mlc_chat_config, key) is None:
                    setattr(mlc_chat_config, key, value)
                    logger.info("[%s] Setting %s: %s", generation_config_filename, bold(key), value)
        else:
            logger.info("%s %s: %s", NOT_FOUND, generation_config_filename, generation_config)

    # Step 3. Copy tokenizer configuration
    # 3.1. Copy over the files and populate mlc_chat_config
    for filename in TOKENIZER_FILES:
        file = config.parent / filename
        if file.exists():
            mlc_chat_config.tokenizer_files.append(filename)
            dest = output / filename
            shutil.copy(file, dest)
            logger.info("%s tokenizer config: %s. Copying to %s", FOUND, file, bold(str(dest)))
        else:
            logger.info("%s tokenizer config: %s", NOT_FOUND, file)
    # 3.2. Generate `tokenizer_model` for rwkv if `rwkv_vocab_.*` is found
    pattern = re.compile(r"rwkv_vocab_v\d{8}\.(json|txt)")
    for item in config.parent.iterdir():
        if item.is_file() and pattern.match(item.name):
            logger.info(
                "%s RWKV vocab file: %s. Genetating %s", FOUND, item, bold("tokenizer_model")
            )
            if item.name.endswith(".txt"):
                txt2rwkv_tokenizer(item, output)
            else:
                json2rwkv_tokenizer(item, output)
    # 3.3. If we have `tokenizer.model` but not `tokenizer.json`, try convert it to
    # `tokenizer.json` with `transformers`.
    tokenizer_json_file = config.parent / "tokenizer.json"
    tokenizer_model_file = config.parent / "tokenizer.model"
    if tokenizer_model_file.exists() and (not tokenizer_json_file.exists()):
        logger.info(
            "The model has `tokenizer.model` but not `tokenizer.json`. "
            "It is always recommended to prefer JSON instead. "
            "Attempting to convert using HuggingFace transformers library"
        )
        try:
            from transformers import (  # pylint: disable=import-error,import-outside-toplevel
                AutoTokenizer,
            )

            tokenizer_json_save_dest = output / "tokenizer.json"
            fast_tokenizer = AutoTokenizer.from_pretrained(str(config.parent), use_fast=True)
            fast_tokenizer.backend_tokenizer.save(str(tokenizer_json_save_dest))
            mlc_chat_config.tokenizer_files.append("tokenizer.json")
            logger.info("Succesfully converted `tokenizer.model` to: %s", tokenizer_json_save_dest)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Convertion to `tokenizer.json` %s with the exception below. "
                "Skipping the conversion. Tokenizer will only use `tokenizer.model`",
                FAILED,
                exc_info=True,
            )
    # 3.3. If we still don't have "tokenizer.json" at this point, try looking for "*.tiktoken" files
    if (not tokenizer_json_file.exists()) and list(config.parent.glob("*.tiktoken")):
        try:
            logger.info(
                "The model has tiktoken files but not `tokenizer.json`. "
                "Attempting to convert from tiktoken files"
            )
            convert_tiktoken.convert_tiktoken(
                str(config.parent), str(output), mlc_chat_config.context_window_size
            )
            mlc_chat_config.tokenizer_files.append("tokenizer.json")
            mlc_chat_config.tokenizer_files.append("vocab.json")
            mlc_chat_config.tokenizer_files.append("merges.txt")
            mlc_chat_config.tokenizer_files.append("special_tokens_map.json")
            logger.info("Succesfully converted from tiktoken files to: %s", str(output))
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("%s with the exception below. Skipping", FAILED)

    # 3.4. Find the token table postprocessing method from tokenizer.json if it exists. If not
    # detected, use "byte_fallback" as default.
    mlc_chat_config.token_table_postproc_method = detect_token_table_postproc_method(output)

    # Step 4. Load system default value
    mlc_chat_config.apply_defaults()
    # Step 5. Dump the configuration file to output directory
    with (output / "mlc-chat-config.json").open("w", encoding="utf-8") as out_file:
        json.dump(dataclasses.asdict(mlc_chat_config), out_file, indent=2)
        logger.info("Dumping configuration file to: %s", bold(out_file.name))


TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenizer_config.json",
]
# FIXME: Copy RWKV tokenizer file # pylint: disable=fixme

CONV_TEMPLATES = {
    "llama-3",
    "chatml",
    "open_hermes_mistral",
    "neural_hermes_mistral",
    "llama_default",
    "llama-2",
    "mistral_default",
    "gpt2",
    "codellama_completion",
    "codellama_instruct",
    "vicuna_v1.1",
    "conv_one_shot",
    "redpajama_chat",
    "rwkv_world",
    "rwkv",
    "gorilla",
    "gorilla-openfunctions-v2",
    "guanaco",
    "dolly",
    "oasst",
    "stablelm",
    "stablecode_completion",
    "stablecode_instruct",
    "minigpt",
    "moss",
    "LM",
    "stablelm-3b",
    "gpt_bigcode",
    "wizardlm_7b",
    "wizard_coder_or_math",
    "glm",
    "custom",  # for web-llm only
    "phi-2",
    "stablelm-2",
    "gemma_instruction",
    "orion",
    "llava",
}
