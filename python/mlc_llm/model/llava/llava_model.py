"""
Implementation of LLaVa Model
Implements the CLIP Vision Encoder. Uses Llama for the Language Encoder.
"""

import dataclasses
import logging
from typing import Any, Dict, Optional

from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor
from tvm.relax.frontend.nn.op import permute_dims, reshape, wrap_nested
from tvm.relax.op import strided_slice

from mlc_llm import op as op_ext
from mlc_llm.model.model_preset import MODEL_PRESETS
from mlc_llm.model.vision import CLIPVisionConfig, CLIPVisionModel, ImageProcessor
from mlc_llm.nn import PagedKVCache, RopeMode

from ...support.config import ConfigBase
from ..llama.llama_model import LlamaConfig, LlamaForCausalLM
from ..mistral.mistral_model import MistralConfig, MistralForCasualLM

logger = logging.getLogger(__name__)


CONFIG_MAP = {"LlamaForCausalLM": LlamaConfig, "MistralForCausalLM": MistralConfig}
ARCHITECTURE_MAP = {"LlamaForCausalLM": LlamaForCausalLM, "MistralForCausalLM": MistralForCasualLM}


@dataclasses.dataclass
class LlavaConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """
    LLaVa Config
    """

    image_token_index: int
    text_config: LlamaConfig
    vision_config: CLIPVisionConfig
    vocab_size: int
    context_window_size: int = -1
    sliding_window_size: int = -1
    prefill_chunk_size: int = -1
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    text_architecture: str = "LlamaForCausalLM"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        vision_config_dict: Dict[str, Any]
        if isinstance(self.vision_config, CLIPVisionConfig):
            vision_config_dict = dataclasses.asdict(self.vision_config)
        else:
            vision_config_dict = dict(self.vision_config)

        for k, v in vision_config_dict.pop("kwargs", {}).items():
            vision_config_dict[k] = v

        self.vision_config = CLIPVisionConfig.from_dict(vision_config_dict)

        text_config_dict: Dict[str, Any]
        if isinstance(self.text_config, ConfigBase):
            text_config_dict = dataclasses.asdict(self.text_config)
        else:
            text_config_dict = dict(self.text_config)

        if "_name_or_path" in text_config_dict:
            hf_config = self.get_hf_config(text_config_dict)
            text_config_dict.update(hf_config)
            architectures = text_config_dict["architectures"]
            assert len(architectures) == 1
            self.text_architecture = architectures[0]
        else:
            for k, v in text_config_dict.pop("kwargs", {}).items():
                text_config_dict[k] = v

        self.text_config = CONFIG_MAP[self.text_architecture].from_dict(  # type: ignore
            text_config_dict
        )

        for k in ["context_window_size", "sliding_window_size", "prefill_chunk_size"]:
            if getattr(self, k) <= 0:
                if hasattr(self.text_config, k):
                    setattr(self, k, getattr(self.text_config, k))

    def get_hf_config(self, text_config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the Hugging Face config of the text model
        """

        hf_config: Dict[str, Any]
        try:
            # pylint: disable=import-outside-toplevel, import-error
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(text_config_dict["_name_or_path"]).to_dict()
        except (ImportError, OSError) as e:
            # If transformers is not installed, get the config from preset
            # Llama2 is gated so it throws an OSError. Get the config from preset instead
            preset_mapping = {
                "meta-llama/Llama-2-7b-hf": "llama2_7b",
                "meta-llama/Llama-2-13b-hf": "llama2_13b",
                "lmsys/vicuna-7b-v1.5": "llama2_7b",
                "mistralai/Mistral-7B-v0.1": "mistral_7b",
            }
            if text_config_dict["_name_or_path"] in preset_mapping:
                hf_config = MODEL_PRESETS[preset_mapping[text_config_dict["_name_or_path"]]]
            else:
                raise ValueError("Unsupported text model") from e

        return hf_config


# pylint: disable=invalid-name,missing-docstring


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def forward(self, image_features: Tensor) -> Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaForCasualLM(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = CLIPVisionModel(config.vision_config)
        self.image_processor = ImageProcessor()
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = ARCHITECTURE_MAP[config.text_architecture](config.text_config)
        self.vocab_size = config.vocab_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        self.language_model.to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.language_model.embed(input_ids)

    def image_preprocess(self, pixel_values: Tensor) -> Tensor:
        pixel_values = permute_dims(pixel_values, axes=(0, 3, 1, 2))  # NHWC -> NCHW
        pixel_values = self.image_processor.resize(
            pixel_values,
            {
                "shortest_edge": self.config.vision_config.image_size,
            },
        )
        pixel_values = self.image_processor.crop(
            pixel_values,
            {
                "height": self.config.vision_config.image_size,
                "width": self.config.vision_config.image_size,
            },
        )
        pixel_values = self.image_processor.rescale(pixel_values)
        pixel_values = self.image_processor.normalize(pixel_values)
        return pixel_values

    def image_embed(self, pixel_values: Tensor) -> Tensor:
        pixel_values = self.image_preprocess(pixel_values)
        pixel_values = pixel_values.astype(self.dtype)
        image_features_all = self.vision_tower.forward(pixel_values)
        image_features = wrap_nested(
            strided_slice(
                image_features_all._expr,  # pylint: disable=protected-access
                axes=[1],
                begin=[1],
                end=[image_features_all.shape[1]],
            ),
            name="slice",
        )
        image_features = self.multi_modal_projector(image_features)
        image_features = reshape(image_features, shape=(-1, self.config.text_config.hidden_size))
        return image_features

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        return self.language_model.batch_forward(input_embeds, paged_kv_cache, logit_positions)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        return self.language_model.prefill(input_embed, paged_kv_cache)

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        return self.language_model.decode(input_embed, paged_kv_cache)

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        return self.language_model.batch_prefill(input_embeds, logit_positions, paged_kv_cache)

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.language_model.batch_decode(input_embeds, paged_kv_cache)

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.language_model.batch_verify(input_embeds, paged_kv_cache)

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
            num_hidden_layers=self.config.text_config.num_hidden_layers,
            num_attention_heads=self.config.text_config.num_attention_heads
            // self.config.tensor_parallel_shards,
            num_key_value_heads=self.config.text_config.num_key_value_heads
            // self.config.tensor_parallel_shards,
            qk_head_dim=self.config.text_config.head_dim,
            v_head_dim=self.config.text_config.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.language_model.rope_theta,
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
                    [1, "image_height", "image_width", 3],
                    "uint8",
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor(
                    [1, "seq_len", self.config.text_config.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor(
                    [1, 1, self.config.text_config.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.config.text_config.hidden_size], self.dtype
                ),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(
                    ["batch_size", 1, self.config.text_config.hidden_size], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.config.text_config.hidden_size], self.dtype
                ),
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
