"""Qwen3.5 Vision-Language model wrapper (hybrid DeltaNet + full attention)."""

import dataclasses
from typing import Any, Dict, Optional

from tvm import relax, target, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Object, Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.vision import ImageProcessor
from mlc_llm.nn.kv_cache import PagedKVCache
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

from .qwen35_model import Qwen35Config, Qwen35LMHeadModel
from .qwen35_vision import Qwen35VisionConfig, Qwen35VisionModel

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name,missing-docstring,too-many-instance-attributes


@dataclasses.dataclass
class Qwen35VConfig(ConfigBase):
    """Configuration for Qwen3.5 Vision-Language model."""

    text_config: Qwen35Config = None
    vision_config: Qwen35VisionConfig = None
    image_token_id: int = 248056
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    image_size: int = 448
    vocab_size: int = -1
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    prefill_chunk_size: int = -1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Parse text_config
        if self.text_config is None:
            raise ValueError("Qwen35VConfig requires text_config")

        if isinstance(self.text_config, Qwen35Config):
            text_dict = dataclasses.asdict(self.text_config)
        else:
            text_dict = dict(self.text_config)
        # Flatten nested kwargs to avoid double-kwargs in from_dict round-trips
        for k, v in text_dict.pop("kwargs", {}).items():
            text_dict[k] = v
        self.text_config = Qwen35Config.from_dict(text_dict)

        # Parse vision_config
        if isinstance(self.vision_config, Qwen35VisionConfig):
            vision_dict = dataclasses.asdict(self.vision_config)
        elif self.vision_config is not None:
            vision_dict = dict(self.vision_config)
        else:
            raise ValueError("Qwen35VConfig requires vision_config")
        for k, v in vision_dict.pop("kwargs", {}).items():
            vision_dict[k] = v
        self.vision_config = Qwen35VisionConfig.from_dict(vision_dict)

        # Propagate sizes from text_config
        for k in ["vocab_size", "context_window_size", "prefill_chunk_size"]:
            if getattr(self, k) <= 0 and hasattr(self.text_config, k):
                setattr(self, k, getattr(self.text_config, k))

    @property
    def grid_h(self) -> int:
        return self.image_size // self.vision_config.patch_size

    @property
    def grid_w(self) -> int:
        return self.image_size // self.vision_config.patch_size

    @property
    def tokens_per_image(self) -> int:
        m = self.vision_config.spatial_merge_size
        return (self.grid_h // m) * (self.grid_w // m)


class Qwen35VForCausalLM(nn.Module):
    def __init__(self, config: Qwen35VConfig):
        self.config = config
        self.language_model = Qwen35LMHeadModel(config.text_config)
        self.visual = Qwen35VisionModel(config.vision_config, config.image_size)
        self.image_processor = ImageProcessor()

        # Expose text model attributes for engine/compile
        self.hidden_size = config.text_config.hidden_size
        self.vocab_size = config.text_config.vocab_size
        self.num_hidden_layers = config.text_config.num_hidden_layers
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_key_value_heads = config.text_config.num_key_value_heads
        self.head_dim = config.text_config.head_dim
        self.dtype = "float32"
        self.image_dtype = (
            "uint32"
            if target.Target.current() and target.Target.current().kind.name == "webgpu"
            else "uint8"
        )

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    # pylint: disable=protected-access
    def image_preprocess(self, pixel_values: Tensor) -> Tensor:
        # NHWC -> NCHW
        pixel_values = op.permute_dims(pixel_values, axes=[0, 3, 1, 2])

        image_size = self.config.image_size

        # Resize to fixed image_size x image_size
        pixel_values = self.image_processor.resize(
            pixel_values, params={"height": image_size, "width": image_size}
        )

        # match_cast to fix shape after resize
        pixel_values = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                pixel_values._expr,
                relax.TensorStructInfo([1, 3, image_size, image_size], pixel_values.dtype),
            ),
            "resized_image",
        )

        # Rescale: uint8 -> float32, /255
        pixel_values = self.image_processor.rescale(pixel_values)

        # Normalize with mean=0.5, std=0.5
        pixel_values = self.image_processor.normalize_siglip(pixel_values)

        return pixel_values

    def image_embed(  # pylint: disable=too-many-arguments,unused-argument
        self,
        pixel_values: Tensor,
        resized_height,
        resized_width,
        crop_height,
        crop_width,
    ) -> Tensor:
        # Preprocess
        pixel_values = self.image_preprocess(pixel_values)

        # Cast to model dtype
        pixel_values = pixel_values.astype(self.dtype)

        # Vision encoder -> (1, tokens_per_image, out_hidden_size)
        vision_outputs = self.visual(pixel_values)

        # Reshape to 2D for C++ runtime (requires ndim == 2)
        tokens_per_image = self.config.tokens_per_image
        vision_outputs = op.reshape(
            vision_outputs, (tokens_per_image, self.hidden_size)
        )
        return vision_outputs

    def embed(self, input_ids: Tensor):
        return self.language_model.embed(input_ids)

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self.language_model.batch_prefill(
            input_embeds, logit_positions, paged_kv_cache, rnn_state
        )

    def batch_decode(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self.language_model.batch_decode(input_embeds, paged_kv_cache, rnn_state)

    def batch_verify(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self.language_model.batch_verify(input_embeds, paged_kv_cache, rnn_state)

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> Object:
        return self.language_model.create_paged_kv_cache(
            max_batch_size,
            max_total_seq_len,
            prefill_chunk_size,
            page_size,
            support_sliding_window,
        )

    def create_rnn_state(
        self,
        max_batch_size: tirx.Var,
        max_history: tirx.Var,
    ) -> Object:
        return self.language_model.create_rnn_state(max_batch_size, max_history)

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "image_embed": {
                "pixel_values": nn.spec.Tensor(
                    [1, "image_height", "image_width", 3], self.image_dtype
                ),
                "resized_height": nn.spec.Int(),
                "resized_width": nn.spec.Int(),
                "crop_height": nn.spec.Int(),
                "crop_width": nn.spec.Int(),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size], self.dtype
                ),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(
                    ["batch_size", 1, self.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
