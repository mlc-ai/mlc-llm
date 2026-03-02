"""Generator of mlc-chat-config.json and tokenizer configuration."""

# pylint: disable=E1101
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.model import Model
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.quantization import Quantization
from mlc_llm.support import convert_tiktoken, logging
from mlc_llm.support.style import bold, green, red
from mlc_llm.tokenizers import Tokenizer

from .compiler_flags import ModelConfigOverride

logger = logging.getLogger(__name__)

FOUND = green("Found")
NOT_FOUND = red("Not found")
FAILED = red("Failed")


def apply_system_defaults_for_missing_fields(mlc_chat_config: MLCChatConfig) -> None:
    """Apply system default value."""
    for key, value in mlc_chat_config.get_system_defaults_for_missing_fields().items():
        setattr(mlc_chat_config, key, value)
        logger.info("[System default] Setting %s: %s", bold(key), value)


def check_string(s: str) -> bool:
    """Check whether it's a string."""
    s = s[1:] if s[0] == "b" else s
    delimit = s[0]
    if s[-1] != delimit or delimit not in ["'", '"']:
        return False
    for i in range(1, len(s) - 1):
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
    pipeline_parallel_stages: Optional[int],
    disaggregation: Optional[bool],
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
        pipeline_parallel_stages=pipeline_parallel_stages,
        disaggregation=disaggregation,
    ).apply(model.config.from_file(config))
    mlc_chat_config = MLCChatConfig(
        model_type=model.name,
        quantization=quantization.name,
        model_config=model_config.asdict(),
        vocab_size=model_config.vocab_size,
        active_vocab_size=getattr(model_config, "active_vocab_size", model_config.vocab_size),
        context_window_size=getattr(model_config, "context_window_size", -1),
        sliding_window_size=getattr(model_config, "sliding_window_size", -1),
        prefill_chunk_size=model_config.prefill_chunk_size,
        attention_sink_size=getattr(model_config, "attention_sink_size", -1),
        tensor_parallel_shards=model_config.tensor_parallel_shards,
        pipeline_parallel_stages=getattr(model_config, "pipeline_parallel_stages", 1),
        disaggregation=getattr(model_config, "disaggregation", False),
        conv_template=conversation,  # type: ignore
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
                    logger.info(
                        "[%s] Setting %s: %s",
                        generation_config_filename,
                        bold(key),
                        value,
                    )
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
                "%s RWKV vocab file: %s. Genetating %s",
                FOUND,
                item,
                bold("tokenizer_model"),
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
            logger.info(
                "Successfully converted `tokenizer.model` to: %s",
                tokenizer_json_save_dest,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Converting to `tokenizer.json` %s with the exception below. "
                "Skipping the conversion.",
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

    # 3.4. Detect tokenizer info
    mlc_chat_config.tokenizer_info = asdict(Tokenizer.detect_tokenizer_info(str(output)))
    logger.info("Detected tokenizer info: %s", mlc_chat_config.tokenizer_info)

    # 3.5. Ensure added_tokens do not have duplicated added_tokens, a mistake from model releaser
    # that affects correctness of huggingface tokenizer.
    # See https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/discussions/15.
    if tokenizer_json_file.exists():
        with open(tokenizer_json_file, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
            if "added_tokens" in tokenizer_json:
                appeared_content = set()
                for added_token in tokenizer_json["added_tokens"]:
                    content = added_token["content"]
                    if content in appeared_content:
                        logger.exception(
                            "%s with incorrect tokenizer.json which has duplicated token %s. "
                            "This affects correctness of huggingface tokenizer during runtime, "
                            "please check your tokenizer.json to remove duplication manually.",
                            FAILED,
                            content,
                        )
                        raise ValueError("Duplicated vocab in tokenizer.json")
                    appeared_content.add(content)

    # Step 4. Load system default value
    apply_system_defaults_for_missing_fields(mlc_chat_config)

    # Step 5. Use HF tokenizer to detect active vocab size via len(tokenizer)
    if tokenizer_json_file.exists():
        try:
            from transformers import (  # pylint: disable=import-error,import-outside-toplevel
                AutoTokenizer,
            )

            hf_tokenizer = AutoTokenizer.from_pretrained(str(config.parent), use_fast=True)
            active_vocab_size = len(hf_tokenizer)
            if mlc_chat_config.active_vocab_size != active_vocab_size:
                logger.info(
                    "Overriding active_vocab_size from %d to %d using HF tokenizer",
                    mlc_chat_config.active_vocab_size,
                    active_vocab_size,
                )
                mlc_chat_config.active_vocab_size = active_vocab_size
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Detecting active_vocab_size %s with the exception below. Skipping.",
                FAILED,
                exc_info=True,
            )

    # Step 5. Dump the configuration file to output directory
    with (output / "mlc-chat-config.json").open("w", encoding="utf-8") as out_file:
        json.dump(mlc_chat_config.model_dump(by_alias=True), out_file, indent=2)
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
    "llama-4",
    "llama-3",
    "llama-3_1",
    "chatml",
    "chatml_nosystem",
    "qwen2",
    "open_hermes_mistral",
    "neural_hermes_mistral",
    "llama_default",
    "llama-2",
    "mistral_default",
    "ministral3",
    "ministral3_reasoning",
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
    "phi-2",
    "phi-3",
    "phi-3-vision",
    "phi-4",
    "stablelm-2",
    "gemma_instruction",
    "gemma3_instruction",
    "orion",
    "llava",
    "hermes2_pro_llama3",
    "hermes3_llama-3_1",
    "tinyllama_v1_0",
    "aya-23",
    "deepseek",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_r1_qwen",
    "deepseek_r1_llama",
    "olmo",
    "nemotron",
    "llm-jp",
}
