"""LoRA configuration dataclass for MLC LLM."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) parameters.

    This configuration is used to define LoRA adaptation parameters
    for fine-tuning large language models with low-rank matrices.

    Parameters
    ----------
    r : int
        LoRA rank (dimension of the low-rank matrices). Common values are 4, 8, 16, 32.
        Higher values provide more capacity but increase parameters.

    lora_alpha : float
        LoRA scaling factor. Controls the magnitude of the LoRA adaptation.
        Typically set to the same value as r or higher.

    lora_dropout : float
        Dropout probability for LoRA layers during training.
        Set to 0.0 for inference.

    target_modules : List[str]
        List of module names to apply LoRA to.
        Common targets: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    fan_in_fan_out : bool
        Whether the layer uses fan_in_fan_out convention.
        Set to True for Conv1D layers, False for Linear layers.

    bias : str
        Bias type for LoRA layers. Options: "none", "all", "lora_only"

    task_type : Optional[str]
        Task type for the LoRA adaptation (e.g., "CAUSAL_LM")

    inference_mode : bool
        Whether the model is in inference mode.

    merge_weights : bool
        Whether to merge LoRA weights into base weights during inference.
    """

    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    fan_in_fan_out: bool = False
    bias: str = "none"
    task_type: Optional[str] = None
    inference_mode: bool = False
    merge_weights: bool = True

    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.target_modules is None:
            self.target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    @property
    def scaling(self) -> float:
        """Return the scaling factor for LoRA: alpha / r."""
        return self.lora_alpha / self.r

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "fan_in_fan_out": self.fan_in_fan_out,
            "bias": self.bias,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
            "merge_weights": self.merge_weights,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LoRAConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
