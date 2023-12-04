"""Flags for overriding model config."""
import argparse
import dataclasses
from io import StringIO
from typing import Any, Optional

from ..support import logging
from ..support.style import bold, red

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfigOverride:
    """Flags for overriding model config."""

    context_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    sliding_window: Optional[int] = None
    max_batch_size: Optional[int] = None
    num_shards: Optional[int] = None

    def __repr__(self) -> str:
        out = StringIO()
        print(f"context_window_size={self.context_window_size}", file=out, end="")
        print(f";prefill_chunk_size={self.prefill_chunk_size}", file=out, end="")
        print(f";sliding_window={self.sliding_window}", file=out, end="")
        print(f";max_batch_size={self.max_batch_size}", file=out, end="")
        print(f";num_shards={self.num_shards}", file=out, end="")
        return out.getvalue().rstrip()

    def __post_init__(self):
        # If `sliding_window` is set
        # - 1) Disable `context_window_size`
        # - 2) Require `prefill_chunk_size` to present
        if self.sliding_window is not None:
            self.context_window_size = -1
            logger.info(
                "Setting %s to -1 (disabled), because %s is already set",
                bold("context_window_size"),
                bold("sliding_window"),
            )
            if self.prefill_chunk_size is None:
                logger.info(
                    "Default %s to %s (%d) because it is not provided",
                    bold("prefill_chunk_size"),
                    bold("sliding_window"),
                    self.sliding_window,
                )
                self.prefill_chunk_size = self.sliding_window

    def apply(self, model_config):
        """Apply the overrides to the given model config."""
        if self.context_window_size is not None:
            _model_config_override(model_config, "context_window_size", self.context_window_size)
        if self.prefill_chunk_size is not None:
            _model_config_override(model_config, "prefill_chunk_size", self.prefill_chunk_size)
        if self.sliding_window is not None:
            _model_config_override(model_config, "sliding_window", self.sliding_window)
        if self.max_batch_size is not None:
            _model_config_override(model_config, "max_batch_size", self.max_batch_size)
        if self.num_shards is not None:
            _model_config_override(model_config, "num_shards", self.num_shards)

    @staticmethod
    def from_str(source: str) -> "ModelConfigOverride":
        """Parse model config override values from a string."""

        parser = argparse.ArgumentParser(description="model config override values")
        parser.add_argument("--context_window_size", type=int, default=None)
        parser.add_argument("--prefill_chunk_size", type=int, default=None)
        parser.add_argument("--sliding_window", type=int, default=None)
        parser.add_argument("--max_batch_size", type=int, default=None)
        parser.add_argument("--num_shards", type=int, default=None)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return ModelConfigOverride(
            context_window_size=results.context_window_size,
            prefill_chunk_size=results.prefill_chunk_size,
            sliding_window=results.sliding_window,
            max_batch_size=results.max_batch_size,
            num_shards=results.num_shards,
        )


def _model_config_override(model_config, field: str, value: Any) -> None:
    if hasattr(model_config, field):
        logger.info(
            "Overriding %s from %d to %d",
            bold(field),
            getattr(model_config, field),
            value,
        )
        setattr(model_config, field, value)
    else:
        logger.warning(
            "%s: %s does not have %s",
            red("Warning"),
            bold(type(model_config).__name__),
            bold("sliding_window"),
        )
