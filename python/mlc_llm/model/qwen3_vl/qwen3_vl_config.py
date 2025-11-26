"""
Configuration for Qwen3-VL model.
"""

import dataclasses
from typing import Any, Dict, Optional, Tuple

from mlc_llm.model.qwen3.qwen3_model import Qwen3Config
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Qwen3VLVisionConfig(ConfigBase):
    """Configuration for the vision module of Qwen3-VL."""

    depth: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: list[int]
    initializer_range: float = 0.02
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Qwen3VLConfig(ConfigBase):
    """Configuration for Qwen3-VL model."""

    text_config: Qwen3Config
    vision_config: Qwen3VLVisionConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    max_batch_size: int = 128
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def prefill_chunk_size(self) -> int:
        return self.text_config.prefill_chunk_size

    @property
    def context_window_size(self) -> int:
        return self.text_config.context_window_size

    @property
    def tensor_parallel_shards(self) -> int:
        return self.text_config.tensor_parallel_shards

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = Qwen3Config.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen3VLVisionConfig.from_dict(self.vision_config)

    @classmethod
    def from_huggingface(cls, config_json: Dict[str, Any]) -> "Qwen3VLConfig":
        """Create Qwen3VLConfig from HuggingFace config."""
        # Extract text config
        text_config_dict = config_json.get("text_config", {})
        # Ensure model_type is set correctly for Qwen3Config if needed, or just pass as is
        # Qwen3Config might expect certain fields.
        
        # Extract vision config
        vision_config_dict = config_json.get("vision_config", {})

        # Extract top-level fields
        image_token_id = config_json.get("image_token_id", 151655)
        video_token_id = config_json.get("video_token_id", 151656)
        vision_start_token_id = config_json.get("vision_start_token_id", 151652)
        vision_end_token_id = config_json.get("vision_end_token_id", 151653)
        tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        return cls(
            text_config=Qwen3Config.from_dict(text_config_dict),
            vision_config=Qwen3VLVisionConfig.from_dict(vision_config_dict),
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            tie_word_embeddings=tie_word_embeddings,
            kwargs=config_json,
        )

# Testing command
# conda activate tvm-dev
# export LOCAL_MODEL_PATH=../mlc-models/Qwen3-VL-2B-Instruct/
# export MLC_MODEL_PATH=../mlc-models/mlc-qwen/
# export QUANTIZATION=q0f16
# export CONV_TEMPLATE=qwen3_vl
# python -m mlc_llm gen_config $LOCAL_MODEL_PATH \
#     --quantization $QUANTIZATION \
#     --conv-template $CONV_TEMPLATE \
#     -o $MLC_MODEL_PATH