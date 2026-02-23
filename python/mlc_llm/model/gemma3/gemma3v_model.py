"""Implementation for Gemma3 Vision-Language architecture."""

import dataclasses
from typing import Any, Dict, Optional

from tvm import relax, target, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.vision import ImageProcessor, SigLIPVisionConfig, SigLIPVisionModel
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

from .gemma3_model import Gemma3Config, Gemma3LanguageModel, Gemma3TextConfig

logger = logging.getLogger(__name__)


SIGLIP_DEFAULT_CONFIG = {
    "hidden_size": 1152,
    "image_size": 896,
    "intermediate_size": 4304,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
    "num_channels": 3,
    "layer_norm_eps": 1e-06,
}


@dataclasses.dataclass
class Gemma3VConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Gemma3 Vision-Language model."""

    text_config: Gemma3TextConfig = None
    vision_config: SigLIPVisionConfig = None
    vocab_size: int = 262_208
    mm_tokens_per_image: int = 256
    boi_token_index: int = 255999
    eoi_token_index: int = 256000
    image_token_index: int = 262144
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    sliding_window_size: int = -1
    prefill_chunk_size: int = -1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Parse text_config
        if self.text_config is None:
            raise ValueError("Gemma3VConfig requires text_config")

        text_config_dict: Dict[str, Any]
        if isinstance(self.text_config, Gemma3TextConfig):
            text_config_dict = dataclasses.asdict(self.text_config)
        else:
            text_config_dict = dict(self.text_config)

        for k, v in text_config_dict.pop("kwargs", {}).items():
            text_config_dict[k] = v

        self.text_config = Gemma3TextConfig.from_dict(text_config_dict)

        # Parse vision_config
        vision_config_dict: Dict[str, Any]
        if isinstance(self.vision_config, SigLIPVisionConfig):
            vision_config_dict = dataclasses.asdict(self.vision_config)
        elif self.vision_config is not None:
            vision_config_dict = dict(self.vision_config)
        else:
            vision_config_dict = dict(SIGLIP_DEFAULT_CONFIG)

        for k, v in vision_config_dict.pop("kwargs", {}).items():
            vision_config_dict[k] = v

        self.vision_config = SigLIPVisionConfig.from_dict(vision_config_dict)

        # Propagate sizes from text_config
        for k in ["context_window_size", "prefill_chunk_size", "sliding_window_size"]:
            if getattr(self, k) <= 0:
                if hasattr(self.text_config, k):
                    setattr(self, k, getattr(self.text_config, k))


# pylint: disable=invalid-name,missing-docstring


class Gemma3MultiModalProjector(nn.Module):
    no_quantization: bool = False

    def __init__(self, config: Gemma3VConfig):
        super().__init__()
        vision_hidden = config.vision_config.hidden_size
        text_hidden = config.text_config.hidden_size

        # Standard RMSNorm (NOT Gemma +1 variant)
        self.mm_soft_emb_norm = nn.RMSNorm(vision_hidden, -1, 1e-6, bias=False)
        # Linear projection: vision -> text hidden size
        self.mm_input_projection = nn.Linear(vision_hidden, text_hidden, bias=False)

    def forward(self, vision_features: Tensor) -> Tensor:
        vision_features = self.mm_soft_emb_norm(vision_features)
        projected = self.mm_input_projection(vision_features)
        return projected


class Gemma3VForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Gemma3VConfig):
        super().__init__()
        self.config = config
        self.language_model = Gemma3LanguageModel(
            Gemma3Config(
                text_config=config.text_config,
                vocab_size=config.vocab_size,
                tensor_parallel_shards=config.tensor_parallel_shards,
            )
        )
        self.vision_tower = SigLIPVisionModel(config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.image_processor = ImageProcessor()

        self.num_hidden_layers = config.text_config.num_hidden_layers
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_key_value_heads = config.text_config.num_key_value_heads
        self.head_dim = config.text_config.head_dim
        self.hidden_size = config.text_config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.text_config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
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

    def _avg_pool_4x4(self, x: Tensor) -> Tensor:
        """Average pooling with kernel_size=4, stride=4."""
        return op.wrap_nested(  # pylint: disable=protected-access
            relax.op.nn.avg_pool2d(x._expr, pool_size=(4, 4), strides=(4, 4), layout="NCHW"),
            "avg_pool_4x4",
        )

    # pylint: disable=protected-access
    def image_preprocess(self, pixel_values: Tensor) -> Tensor:
        # NHWC -> NCHW
        pixel_values = op.permute_dims(pixel_values, axes=(0, 3, 1, 2))

        # Resize to fixed 896x896
        pixel_values = self.image_processor.resize(
            pixel_values, params={"height": 896, "width": 896}
        )

        # match_cast to fix shape after resize (returns symbolic dims)
        pixel_values = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                pixel_values._expr,
                relax.TensorStructInfo([1, 3, 896, 896], pixel_values.dtype),
            ),
            "resized_image",
        )

        # Rescale: uint8 -> float32, /255
        pixel_values = self.image_processor.rescale(pixel_values)

        # Normalize with SigLIP values: mean=0.5, std=0.5
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
        # Step 1: Preprocess
        pixel_values = self.image_preprocess(pixel_values)

        # Step 2: Cast to model dtype
        pixel_values = pixel_values.astype(self.dtype)

        # Step 3: Vision encoder -> (1, 4096, 1152)
        vision_outputs = self.vision_tower(pixel_values)

        # Step 4: Reshape for average pooling
        # (1, 4096, 1152) -> (1, 64, 64, 1152) -> (1, 1152, 64, 64)
        grid = self.config.vision_config.image_size // self.config.vision_config.patch_size  # 64
        hidden_dim = self.config.vision_config.hidden_size  # 1152
        vision_outputs = op.reshape(vision_outputs, (1, grid, grid, hidden_dim))
        vision_outputs = op.permute_dims(vision_outputs, (0, 3, 1, 2))  # (1, 1152, 64, 64)

        # Step 5: AvgPool2d(4,4) -> (1, 1152, 16, 16)
        vision_outputs = self._avg_pool_4x4(vision_outputs)

        # Step 6: Reshape back to sequence format
        # (1, 1152, 16, 16) -> (1, 16, 16, 1152) -> (1, 256, 1152)
        vision_outputs = op.permute_dims(vision_outputs, (0, 2, 3, 1))
        tokens_per_image = self.config.mm_tokens_per_image  # 256
        vision_outputs = op.reshape(vision_outputs, (1, tokens_per_image, hidden_dim))

        # Step 7: Multimodal projector (RMSNorm + Linear)
        projected = self.multi_modal_projector(vision_outputs)  # (1, 256, text_hidden)

        # Step 8: Compensate for Gemma's embedding scaling.
        # The language model multiplies ALL input embeddings by sqrt(hidden_size) at the
        # start of its forward pass. In HF, image features are inserted AFTER this scaling,
        # so they don't get scaled. In MLC, image embeddings are inserted BEFORE, so we
        # pre-divide by sqrt(hidden_size) to compensate.
        projected = projected / Tensor.from_scalar(self.hidden_size**0.5, dtype=projected.dtype)

        # Step 9: Reshape to 2D for C++ runtime (requires ndim == 2)
        projected = op.reshape(projected, (tokens_per_image, self.hidden_size))

        return projected

    def get_logits(self, hidden_states: Tensor):
        logits = self.language_model.model.embed_tokens.lm_head_forward(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.language_model.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.get_logits(hidden_states)
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.language_model.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
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

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind=[
                (
                    "mha_sliding"
                    if ((i + 1) % self.config.text_config.sliding_window_pattern)
                    else "mha"
                )
                for i in range(self.num_hidden_layers)
            ],
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
            rope_scale=1,
            rope_theta=self.rope_theta,
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
