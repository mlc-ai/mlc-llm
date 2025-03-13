import json
import os
from pathlib import Path
import logging
import functools

from mlc_llm.support.auto_weight import _guess_weight_format
from mlc_llm.loader.utils import load_safetensor_shard, load_torch_shard

logger = logging.getLogger(__name__)

functools.lru_cache()
class LoRAConfig:
    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path
        self.source = None
        self.source_format = None
        self.target_modules = None
        self.hf_config = None
        self.r = None
        self.scaling = None
        self.global_layer_index = 0
        self.init_config()

    def init_config(self):
        self.source, self.source_format = _guess_weight_format(self.path, is_lora=True)
        config_path = self.path / "adapter_config.json"
        with open(config_path, 'r') as f:
            self.hf_config = json.load(f)
        if len(self.hf_config["modules_to_save"]):
            logger.warning('Do not support LoRA with modules_to_save')
        # FIXME: Do not support embed_tokens and lm_head currently
        unsupported_targets = ["embed_tokens", "lm_head"]
        for target in unsupported_targets:
            if target in self.hf_config["target_modules"]:
                logger.warning(f'Do not support LoRA module {target}')
        self.target_modules = list(filter(lambda x: x not in unsupported_targets, self.hf_config["target_modules"]))
        self.r = self.hf_config["r"]
        self.scaling = self.hf_config["lora_alpha"] / self.r

    def get_weights(self):
        load_func = load_safetensor_shard if self.source.suffix == ".safetensors" else load_torch_shard
        result = {}
        for name, param in load_func(self.source):
            for hf_module in self.target_modules:
                if hf_module in name:
                    result[name] = param
                    break
        return result