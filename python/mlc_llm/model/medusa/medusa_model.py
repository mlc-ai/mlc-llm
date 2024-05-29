"""Medusa model definition."""

import dataclasses
from typing import Any, Dict, Optional

from tvm.relax.frontend import nn

from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MedusaConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Llama model."""

    medusa_num_heads: int
    medusa_num_layers: int
    hidden_size: int
    vocab_size: int
    max_batch_size: int = 1
    tensor_parallel_shards: int = 1

    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Unused parameters. Kept for compatibility with the compilation flow.
    prefill_chunk_size: int = -1
    context_window_size: int = -1


# pylint: disable=missing-docstring


class ResBlock(nn.Module):
    """Residual block with SiLU activation."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module):
    """Medusa model definition."""

    def __init__(self, config: MedusaConfig):
        self.hidden_size = config.hidden_size
        self.dtype = "float32"
        self.medusa_head = nn.ModuleList(
            [
                nn.ModuleList(
                    [ResBlock(config.hidden_size) for _ in range(config.medusa_num_layers)]
                    + [nn.Linear(config.hidden_size, config.vocab_size, bias=False)]
                )
                for _ in range(config.medusa_num_heads)
            ]
        )

    def get_default_spec(self):
        mod_spec = {
            "get_logits": {
                "hidden_states": nn.spec.Tensor(["batch_size", self.hidden_size], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

    def get_logits(self, hidden_states: nn.Tensor):
        logits = []
        for head in self.medusa_head:
            logits.append(head(hidden_states).astype("float32"))
        return logits

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
