"""Flags for overriding model config."""
import dataclasses
import logging
from typing import Optional

from ..support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfigOverride:
    """Flags for overriding model config."""

    context_window_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    num_shards: Optional[int] = None
    sliding_window: Optional[int] = None
    prefill_chunk_size: Optional[int] = None

    def apply(self, model_config):
        """Apply the overrides to the given model config."""
        if self.context_window_size is not None:
            logger.info(
                "Overriding %s from %d to %d",
                bold("context_window_size"),
                model_config.context_window_size,
                self.context_window_size,
            )
            model_config.context_window_size = self.context_window_size
        if self.max_batch_size is not None:
            model_config.max_batch_size = self.max_batch_size
        if self.num_shards is not None:
            model_config.num_shards = self.num_shards

        # Handle sliding window and sliding window chunk size
        if self.sliding_window is not None:
            logger.info(
                "Overriding %s from %d to %d",
                bold("sliding_window"),
                model_config.sliding_window,
                self.sliding_window,
            )
            model_config.sliding_window = self.sliding_window
            if self.prefill_chunk_size is None:
                logger.info(
                    "Provided %s but did not provide %s, setting both to %d",
                    bold("sliding_window"),
                    bold("prefill_chunk_size"),
                    model_config.sliding_window,
                )
                model_config.prefill_chunk_size = self.sliding_window
        if self.prefill_chunk_size is not None:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                model_config.prefill_chunk_size,
                self.prefill_chunk_size,
            )
            model_config.prefill_chunk_size = self.prefill_chunk_size
