"""
Configuration for Qwen3-VL model.
"""

import dataclasses
from typing import Any, Dict, Optional, Tuple

from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

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
class Qwen3VLTextConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Qwen3-VL text model."""

    hidden_act: str
    hidden_size: int
    intermediate_size: int
    attention_bias: bool
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    dtype: str = "float32"
    max_batch_size: int = 1
    weight_block_size: Optional[Tuple[int, int]] = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if "quantization_config" in self.kwargs:
            quantization_config = self.kwargs.get("quantization_config")
            if (
                isinstance(quantization_config, dict)
                and quantization_config.get("activation_scheme", "") == "dynamic"
                and quantization_config.get("fmt", "") == "e4m3"
                and quantization_config.get("quant_method", "") == "fp8"
                and "weight_block_size" in quantization_config
            ):
                self.weight_block_size = quantization_config.get("weight_block_size")
                if (
                    not isinstance(self.weight_block_size, (tuple, list))
                    or len(self.weight_block_size) != 2
                ):
                    raise ValueError(
                        "Invalid DeepSeek model quantization config: "
                        "weight_block_size must be a tuple of two integers, "
                        f"got {self.weight_block_size} of type {type(self.weight_block_size)}"
                    )
            else:
                raise ValueError(
                    "Invalid DeepSeek model quantization config: unrecognized quantization config: "
                    f"{quantization_config}"
                )

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                # Default to 128000 for Qwen3-VL text if not found
                self.context_window_size = 128000
                logger.info(
                    "%s not found in config.json. Falling back to default %d",
                    bold("context_window_size"),
                    self.context_window_size,
                )

        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.context_window_size, 2048)


@dataclasses.dataclass
class Qwen3VLConfig(ConfigBase):
    """Configuration for Qwen3-VL model."""

    text_config: Qwen3VLTextConfig
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
            self.text_config = Qwen3VLTextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen3VLVisionConfig.from_dict(self.vision_config)

    @classmethod
    def from_huggingface(cls, config_json: Dict[str, Any]) -> "Qwen3VLConfig":
        """Create Qwen3VLConfig from HuggingFace config."""
        # Extract text config
        text_config_dict = config_json.get("text_config", {})
        
        # Extract vision config
        vision_config_dict = config_json.get("vision_config", {})

        # Extract top-level fields
        image_token_id = config_json.get("image_token_id", 151655)
        video_token_id = config_json.get("video_token_id", 151656)
        vision_start_token_id = config_json.get("vision_start_token_id", 151652)
        vision_end_token_id = config_json.get("vision_end_token_id", 151653)
        tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        return cls(
            text_config=Qwen3VLTextConfig.from_dict(text_config_dict),
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
# python -m mlc_llm gen_config $LOCAL_MODEL_PATH --quantization $QUANTIZATION --conv-template $CONV_TEMPLATE -o $MLC_MODEL_PATH