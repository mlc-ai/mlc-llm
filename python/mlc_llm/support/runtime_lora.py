"""Validation and normalized metadata for the initial runtime LoRA path."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple  # noqa: UP035

SUPPORTED_QWEN2_TARGET_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


@dataclasses.dataclass(frozen=True)
class RuntimeLoRAConfig:
    """The subset of PEFT LoRA metadata supported by runtime LoRA."""

    rank: int
    alpha: float
    target_modules: Tuple[str, ...]  # noqa: UP006
    use_rslora: bool = False

    @property
    def scaling(self) -> float:
        """Return the inference-time LoRA scale."""
        denominator = math.sqrt(self.rank) if self.use_rslora else self.rank
        return self.alpha / denominator

    def applies_to(self, module_name: str) -> bool:
        """Return whether the adapter targets a logical projection."""
        return module_name in self.target_modules

    def asdict(self) -> Dict[str, Any]:  # noqa: UP006
        """Return JSON-compatible normalized metadata."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "use_rslora": self.use_rslora,
        }

    @classmethod
    def from_dict(cls, source: Dict[str, Any]) -> RuntimeLoRAConfig:  # noqa: UP006
        """Load normalized runtime metadata."""
        return cls._create(
            rank=source.get("rank"),
            alpha=source.get("alpha"),
            target_modules=source.get("target_modules"),
            use_rslora=source.get("use_rslora", False),
        )

    @classmethod
    def from_peft_directory(cls, adapter_dir: Path) -> RuntimeLoRAConfig:
        """Load and strictly validate a PEFT ``adapter_config.json``."""
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.is_file():
            raise ValueError(f"PEFT adapter config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as config_file:
            source = json.load(config_file)
        if not isinstance(source, dict):
            raise ValueError("PEFT adapter config must be a JSON object.")

        if str(source.get("peft_type", "")).upper() != "LORA":
            raise ValueError('Runtime LoRA requires a PEFT adapter with `peft_type="LORA"`.')
        task_type = source.get("task_type")
        if task_type not in (None, "CAUSAL_LM"):
            raise ValueError(
                "Runtime LoRA currently supports only PEFT causal-language-model adapters."
            )

        unsupported_options = {
            "alora_invocation_tokens": source.get("alora_invocation_tokens"),
            "alpha_pattern": source.get("alpha_pattern"),
            "arrow_config": source.get("arrow_config"),
            "ensure_weight_tying": source.get("ensure_weight_tying"),
            "exclude_modules": source.get("exclude_modules"),
            "layer_replication": source.get("layer_replication"),
            "layers_pattern": source.get("layers_pattern"),
            "layers_to_transform": source.get("layers_to_transform"),
            "lora_bias": source.get("lora_bias"),
            "megatron_config": source.get("megatron_config"),
            "monteclora_config": source.get("monteclora_config"),
            "modules_to_save": source.get("modules_to_save"),
            "rank_pattern": source.get("rank_pattern"),
            "target_parameters": source.get("target_parameters"),
            "trainable_token_indices": source.get("trainable_token_indices"),
            "use_bdlora": source.get("use_bdlora"),
            "use_qalora": source.get("use_qalora"),
        }
        enabled_options = [name for name, value in unsupported_options.items() if value]
        if enabled_options:
            raise ValueError(
                "Runtime LoRA does not yet support PEFT options: " + ", ".join(enabled_options)
            )
        if source.get("bias", "none") != "none":
            raise ValueError('Runtime LoRA currently requires PEFT `bias="none"`.')
        if source.get("fan_in_fan_out", False):
            raise ValueError("Runtime LoRA does not support `fan_in_fan_out=True`.")
        if source.get("use_dora", False):
            raise ValueError("Runtime LoRA does not support DoRA adapters.")

        return cls._create(
            rank=source.get("r"),
            alpha=source.get("lora_alpha"),
            target_modules=source.get("target_modules"),
            use_rslora=source.get("use_rslora", False),
        )

    @classmethod
    def _create(cls, rank, alpha, target_modules, use_rslora) -> RuntimeLoRAConfig:
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("Runtime LoRA requires a positive integer rank.")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError("Runtime LoRA requires a numeric `lora_alpha`.")
        alpha = float(alpha)
        if not math.isfinite(alpha):
            raise ValueError("Runtime LoRA requires a finite `lora_alpha`.")
        if isinstance(target_modules, str) or not isinstance(target_modules, (list, tuple, set)):
            raise ValueError("Runtime LoRA requires PEFT `target_modules` to be an explicit list.")
        if not all(isinstance(target, str) for target in target_modules):
            raise ValueError("Runtime LoRA requires every PEFT target module to be a string.")
        if not isinstance(use_rslora, bool):
            raise ValueError("Runtime LoRA requires PEFT `use_rslora` to be a boolean.")
        targets = tuple(sorted(set(target_modules)))
        if not targets:
            raise ValueError("Runtime LoRA requires at least one target module.")
        unsupported_targets = sorted(set(targets) - SUPPORTED_QWEN2_TARGET_MODULES)
        if unsupported_targets:
            raise ValueError(
                "Runtime LoRA currently supports only Qwen2 projection targets; unsupported: "
                + ", ".join(unsupported_targets)
            )
        return cls(
            rank=rank,
            alpha=alpha,
            target_modules=targets,
            use_rslora=use_rslora,
        )


def validate_runtime_lora_scope(
    *, model_name: str, quantization_name: str, tensor_parallel_shards: int
) -> None:
    """Enforce the deliberately narrow scope of the first runtime LoRA implementation."""
    if model_name != "qwen2":
        raise ValueError("Runtime LoRA currently supports only the Qwen2 model family.")
    if quantization_name != "q0f16":
        raise ValueError("Runtime LoRA currently requires `q0f16` model weights.")
    if tensor_parallel_shards != 1:
        raise ValueError("Runtime LoRA currently requires `tensor_parallel_shards=1`.")


__all__ = ["RuntimeLoRAConfig", "validate_runtime_lora_scope"]
