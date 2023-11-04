"""Common configuration for Llama models."""
import dataclasses
from typing import Any, Dict

from ...support.config import ConfigBase


@dataclasses.dataclass
class LlamaConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Llama model."""

    hidden_act: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    max_sequence_length: int = 2048
    position_embedding_base: int = 10000
    num_key_value_heads: int = 0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    head_dim: int = 0

    def __post_init__(self):
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size
