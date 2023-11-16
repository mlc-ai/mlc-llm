"""Generator of mlc-chat-config.json and tokenizer configuration."""
import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..support.style import bold, green, red
from .flags_model_config_override import ModelConfigOverride
from .model import Model
from .quantization import Quantization

logger = logging.getLogger(__name__)

FOUND = green("Found")
NOT_FOUND = red("Not found")
VERSION = "0.1.0"


@dataclasses.dataclass
class MLCChatConfig:  # pylint: disable=too-many-instance-attributes
    """Arguments for `mlc_chat.compiler.gen_config`."""

    version: str = VERSION

    model_type: str = None
    quantization: str = None
    model_config: Dict[str, Any] = None
    vocab_size: int = None
    max_window_size: int = None

    temperature: float = None
    repetition_penalty: float = None
    top_p: float = None

    mean_gen_len: int = None
    max_gen_len: int = None
    shift_fill_factor: float = None

    # Conversation template
    conv_template: str = None
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None
    tokenizer_files: List[str] = dataclasses.field(default_factory=list)


def gen_config(  # pylint: disable=too-many-locals,too-many-arguments
    config: Path,
    model: Model,
    quantization: Quantization,
    conv_template: str,
    context_window_size: Optional[int],
    output: Path,
):
    """Entrypoint of MLC Chat configuration generation."""
    with config.open("r", encoding="utf-8") as in_file:
        model_config_json = json.load(in_file)
    model_config = model.config.from_dict(model_config_json)
    ModelConfigOverride(
        context_window_size=context_window_size,
    ).apply(model_config)

    mlc_chat_config = MLCChatConfig(
        model_type=model.name,
        quantization=quantization.name,
        model_config=model_config_json,
        vocab_size=model_config.vocab_size,
        conv_template=conv_template,
        max_window_size=model_config.context_window_size,
    )
    # Step 1. Load `config.json`
    for key, value in model_config_json.items():
        if hasattr(mlc_chat_config, key) and getattr(mlc_chat_config, key) is None:
            setattr(mlc_chat_config, key, value)
            logger.info("[config.json] Setting %s: %s", bold(key), value)
    # Step 2. Load `generation_config.json`
    generation_config = config.parent / "generation_config.json"
    if generation_config.exists():
        logger.info("%s generation_config.json: %s", FOUND, generation_config)
        with generation_config.open("r", encoding="utf-8") as in_file:
            generation_config_json = json.load(in_file)
        for key, value in generation_config_json.items():
            if hasattr(mlc_chat_config, key) and getattr(mlc_chat_config, key) is None:
                setattr(mlc_chat_config, key, value)
                logger.info("[generation_config.json] Setting %s: %s", bold(key), value)
    else:
        logger.info("%s generation_config.json: %s", NOT_FOUND, generation_config)
    # Step 3. Copy tokenizer configuration
    for filename in TOKENIZER_FILES:
        file = config.parent / filename
        if file.exists():
            mlc_chat_config.tokenizer_files.append(filename)
            dest = output / filename
            shutil.copy(file, dest)
            logger.info("%s tokenizer config: %s. Copying to %s", FOUND, file, bold(str(dest)))
        else:
            logger.info("%s tokenizer config: %s", NOT_FOUND, file)
    # Step 4. Load system default value
    for key, value in DEFAULT_CONFIGS.items():
        if getattr(mlc_chat_config, key) is None:
            setattr(mlc_chat_config, key, value)
            logger.info("[System default] Setting %s: %s", bold(key), value)
    # Dump the configuration file to output directory
    out = output / "mlc-chat-config.json"
    with out.open("w", encoding="utf-8") as out_file:
        json.dump(dataclasses.asdict(mlc_chat_config), out_file, indent=2)
    logger.info("Dumping configuration file to: %s", bold(str(out)))


DEFAULT_CONFIGS = {
    # Conversation
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    # Configuration of text generation
    "temperature": 0.7,
    "repetition_penalty": 1.0,
    "top_p": 0.95,
    # Control the behavior of the runtime
    "mean_gen_len": 128,
    "max_gen_len": 512,
    "shift_fill_factor": 0.3,
}

TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenizer_config.json",
]

CONV_TEMPLATES = {
    "chatml",
    "llama_default",
    "llama-2",
    "mistral_default",
    "codellama_completion",
    "codellama_instruct",
    "vicuna_v1.1",
    "conv_one_shot",
    "redpajama_chat",
    "rwkv_world",
    "rwkv",
    "gorilla",
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
}
