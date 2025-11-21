"""
Implementation for Phi architecture.
"""

import dataclasses
from typing import Any, Dict, Optional

from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.phi3 import Phi3Model
from mlc_llm.model.vision import CLIPVisionConfig, ImageProcessor
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

from .phi3v_image import Phi3ImageEmbedding

logger = logging.getLogger(__name__)

CLIPVISION_DEFAULT_CONFIG = {
    "hidden_size": 1024,
    "image_size": 336,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "layer_norm_eps": 1e-05,
    "vocab_size": None,
}


@dataclasses.dataclass
class Phi3VConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Phi-3 Vision model."""

    model_type: str
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_eps: float
    num_key_value_heads: int
    max_position_embeddings: int
    vision_config: CLIPVisionConfig = None
    img_processor: Optional[Dict[str, Any]] = None
    position_embedding_base: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    original_max_position_embeddings: int = 0
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    head_dim: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # pylint: disable=too-many-branches, consider-using-min-builtin
    def __post_init__(self):
        vision_config_dict: Dict[str, Any]
        if isinstance(self.vision_config, CLIPVisionConfig):
            vision_config_dict = dataclasses.asdict(self.vision_config)
        else:
            vision_config_dict = dict(CLIPVISION_DEFAULT_CONFIG)

        for k, v in vision_config_dict.pop("kwargs", {}).items():
            vision_config_dict[k] = v

        self.vision_config = CLIPVisionConfig.from_dict(vision_config_dict)

        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.rope_scaling is not None:
            if "type" not in self.rope_scaling:
                self.rope_scaling = None
            else:
                if self.rope_scaling["type"] == "su":
                    self.rope_scaling["type"] = "longrope"

                assert (
                    self.rope_scaling["type"] == "longrope"
                ), f'Unsupported RoPE scaling type {self.rope_scaling["rope_type"]} for Phi3'
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
                (
                    self.rope_scaling["max_position_embeddings"],
                    self.rope_scaling["original_max_position_embeddings"],
                ) = (self.max_position_embeddings, self.original_max_position_embeddings)

        if self.context_window_size == 0:
            self.context_window_size = self.max_position_embeddings

        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)

        if self.num_key_value_heads == 0 or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * self.num_attention_heads == self.hidden_size
        assert self.num_attention_heads % self.num_key_value_heads == 0


# pylint: disable=invalid-name,missing-docstring, too-many-branches


# mypy: disable-error-code="arg-type,annotation-unchecked"
class Phi3VForCausalLM(nn.Module):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Phi3VConfig) -> None:
        super().__init__()

        self.config = config
        self.model = Phi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, "vocab_size", bias=False)
        self.vision_embed_tokens = Phi3ImageEmbedding(config)
        self.image_processor = ImageProcessor()
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_scaling = config.rope_scaling
        self.rope_theta = config.position_embedding_base
        self.rope_ext_factors = (
            (config.rope_scaling["long_factor"] + config.rope_scaling["short_factor"])
            if config.rope_scaling is not None
            else None
        )
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        lm_logits = self.lm_head(hidden_states)
        if lm_logits.dtype != "float32":
            lm_logits = lm_logits.astype("float32")
        return lm_logits

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)

        if logits.dtype != "float32":
            logits = logits.astype("float32")

        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        embeds = self.model.embd(input_ids)
        return embeds

    # pylint: disable=protected-access
    def image_preprocess(
        self, pixel_values: Tensor, resized_height, resized_width, num_crops=16
    ) -> Tensor:
        pixel_values = op.permute_dims(pixel_values, axes=(0, 3, 1, 2))  # NHWC -> NCHW
        pixel_values = self.image_processor.resize(
            pixel_values, params={"height": resized_height, "width": resized_width}
        )
        pixel_values = self.image_processor.pad(pixel_values)
        pixel_values = self.image_processor.rescale(pixel_values)
        pixel_values = self.image_processor.normalize(pixel_values)
        global_image = self.image_processor.resize(
            pixel_values, params={"height": 336, "width": 336}
        )
        global_image = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                global_image._expr,
                relax.TensorStructInfo(
                    [global_image.shape[0], global_image.shape[1], 336, 336], global_image.dtype
                ),
            ),
            "global_image",
        )

        n, c, h, w = pixel_values.shape  # pylint: disable=unused-variable
        assert isinstance(h, tir.Mul) and isinstance(h.b, tir.IntImm) and h.b.value == 336
        pixel_values = op.reshape(pixel_values, shape=(1, 3, h.a, 336, w // 336, 336))
        pixel_values = op.permute_dims(pixel_values, axes=(0, 2, 4, 1, 3, 5))
        pixel_values = op.reshape(pixel_values, shape=(-1, 3, 336, 336))
        combined_image = op.concat([global_image, pixel_values], dim=0)

        # pad to max num crops tensor
        b, c, h, w = combined_image.shape
        zeros = op.zeros((num_crops + 1 - b, c, h, w))
        combined_image = op.concat([combined_image, zeros], dim=0)

        combined_image = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                combined_image._expr,
                relax.TensorStructInfo([num_crops + 1, c, h, w], combined_image.dtype),
            ),
            "combined_image",
        )

        return combined_image

    def image_embed(  # pylint: disable=too-many-arguments
        self, pixel_values: Tensor, resized_height, resized_width, crop_height, crop_width
    ) -> Tensor:
        n, h, w, c = pixel_values.shape  # pylint: disable=unused-variable
        pixel_values = self.image_preprocess(pixel_values, resized_height, resized_width)
        pixel_values = pixel_values.astype(self.dtype)
        return self.vision_embed_tokens(pixel_values, crop_height, crop_width)

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scaling=self.rope_scaling,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_ext_factors=self.rope_ext_factors,
            dtype=self.dtype,
        )

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
                "pixel_values": nn.spec.Tensor([1, "image_height", "image_width", 3], "uint8"),
                "resized_height": nn.spec.Int(),
                "resized_width": nn.spec.Int(),
                "crop_height": nn.spec.Int(),
                "crop_width": nn.spec.Int(),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
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
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
