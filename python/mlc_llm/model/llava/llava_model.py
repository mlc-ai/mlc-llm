"""
Implementation of LLaVa Model
Implements the CLIP Vision Encoder. Uses Llama for the Language Encoder.
"""

import dataclasses
import logging
from typing import Any, Dict, Optional, Tuple

from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor, op
from tvm.relax.frontend.nn.modules import Conv2D
from tvm.relax.frontend.nn.op import (
    broadcast_to,
    concat,
    matmul,
    permute_dims,
    reshape,
    softmax,
    wrap_nested,
)
from tvm.relax.op import arange, strided_slice

from mlc_llm import op as op_ext
from mlc_llm.model.model_preset import MODEL_PRESETS
from mlc_llm.nn import PagedKVCache, RopeMode

from ...support.config import ConfigBase
from ..llama.llama_model import LlamaConfig, LlamaForCasualLM
from ..mistral.mistral_model import MistralConfig, MistralForCasualLM

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LlavaVisionConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """
    Config for the vision encoder
    """

    hidden_size: int
    image_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    projection_dim: int
    vocab_size: int
    num_channels: int = 3
    layer_norm_eps: float = 1e-06
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


CONFIG_MAP = {"LlamaForCausalLM": LlamaConfig, "MistralForCausalLM": MistralConfig}
ARCHITECTURE_MAP = {"LlamaForCausalLM": LlamaForCasualLM, "MistralForCausalLM": MistralForCasualLM}


@dataclasses.dataclass
class LlavaConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """
    LLaVa Config
    """

    image_token_index: int
    text_config: LlamaConfig
    vision_config: LlavaVisionConfig
    vocab_size: int
    context_window_size: int = -1
    sliding_window_size: int = -1
    prefill_chunk_size: int = -1
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    text_architecture: str = "LlamaForCausalLM"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        vision_config_dict: Dict[str, Any]
        if isinstance(self.vision_config, LlavaVisionConfig):
            vision_config_dict = dataclasses.asdict(self.vision_config)
        else:
            vision_config_dict = dict(self.vision_config)

        for k, v in vision_config_dict.pop("kwargs", {}).items():
            vision_config_dict[k] = v

        self.vision_config = LlavaVisionConfig.from_dict(vision_config_dict)

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

        self.text_config = CONFIG_MAP[self.text_architecture].from_dict(text_config_dict)

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


# pylint: disable=missing-docstring


class CLIPVisionEmbeddings(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter((self.embed_dim,))
        self.patch_embedding = Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(num=self.num_positions, dim=self.embed_dim)

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = reshape(patch_embeds, shape=(batch_size, self.embed_dim, -1))
        patch_embeds = permute_dims(
            patch_embeds, axes=(0, 2, 1)
        )  # shape = [batch,grid*grid,embed_dim]
        class_embeds = broadcast_to(
            self.class_embedding, shape=(batch_size, 1, self.embed_dim)
        )  # shape of (batch,1,embed_dim)
        embeddings = concat([class_embeds, patch_embeds], dim=1)

        posi_ids = reshape(
            wrap_nested(arange(0, self.num_positions, dtype="int32"), name="arange"), shape=(1, -1)
        )
        batch_position_embedding = broadcast_to(
            self.position_embedding(posi_ids),
            shape=(batch_size, self.num_positions, self.embed_dim),
        )
        embeddings = embeddings + batch_position_embedding
        return embeddings


def sigmoid(x: Tensor, name: str = "sigmoid") -> Tensor:
    """Sigmoid of a Tensor

    Parameters
    ----------
    x : Tensor
        Input tensor to expand.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Sigmoid result.
    """
    return wrap_nested(relax.op.sigmoid(x._expr), name)  # pylint: disable=protected-access


class LlavaQuickGELU(Module):
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * sigmoid(input_tensor * 1.702)


class CLIPMLP(Module):
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.activation_fn = LlavaQuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPAttention(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        reshape_tensor = reshape(tensor, shape=(bsz, seq_len, self.num_heads, self.head_dim))
        permute_tensor = permute_dims(reshape_tensor, axes=(0, 2, 1, 3))
        return permute_tensor

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        bsz, tgt_len, embed_dim = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states) * self.scale, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        proj_shape = (
            bsz * self.num_heads,
            -1,
            self.head_dim,
        )  # shape of (batch*num_heads, seq_len,head_dim)

        query_states = reshape(query_states, shape=proj_shape)
        key_states = reshape(key_states, shape=proj_shape)
        value_states = reshape(value_states, shape=proj_shape)

        trans_key_states = permute_dims(key_states, axes=(0, 2, 1))

        attn_weights = matmul(query_states, trans_key_states)
        attn_weights = softmax(attn_weights, axis=-1)
        attn_output = matmul(attn_weights, value_states)
        attn_output = reshape(attn_output, shape=(bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = permute_dims(attn_output, axes=(0, 2, 1, 3))
        attn_output = reshape(attn_output, shape=(bsz, tgt_len, embed_dim))
        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPEncoderLayer(Module):
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs


class CLIPEncoder(Module):
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        hidden_states = inputs_embeds
        encoder_states: Tuple[Any, ...] = ()
        for _, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]
        encoder_states = encoder_states + (hidden_states,)
        return encoder_states


class CLIPVisionTransformer(Module):
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs


class CLIPVisionModel(Module):
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.vision_model(pixel_values)[-2]


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
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = ARCHITECTURE_MAP[config.text_architecture](config.text_config)
        self.vocab_size = config.vocab_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        self.language_model.to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def _embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.language_model.embed(input_ids)

    def _embed_pixel_values_and_input_ids(self, pixel_values: Tensor, input_ids: Tensor) -> Tensor:
        def _index(x, value, batch_size, seq_len):
            return te.compute(
                (batch_size, seq_len),
                lambda i, j: tir.if_then_else(
                    x[i, j] == value,
                    j,
                    tir.IntImm("int32", 0),
                ),
                name="index",
            )

        def _concat(x: Tensor, y: Tensor, new_shape: tuple, insert_index: Tensor):
            return te.compute(
                (new_shape),
                lambda b, i, j: tir.if_then_else(
                    i < insert_index[0],
                    x[b, i, j],
                    tir.if_then_else(
                        i < insert_index[0] + y.shape[1],
                        y[b, i - insert_index[0], j],
                        x[b, i - y.shape[1] + 1, j],
                    ),
                ),
            )

        input_embeddings = self._embed_input_ids(input_ids)

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
        batch_size, seq_len = input_ids.shape
        image_index_tensor = op.tensor_expr_op(
            _index,
            name_hint="index",
            args=[
                input_ids,
                tir.IntImm("int32", self.config.image_token_index),
                batch_size,
                seq_len,
            ],
        ).astype("int32")
        ##! Assume only one <IMAGE> token in input
        ##! Also assume batch_size = 1 for now
        # TODO: Support image_count > 1 and batch_size > 1 # pylint: disable=fixme
        insert_index = op.sum(image_index_tensor, axis=1)

        new_shape = (
            batch_size,
            seq_len + tir.IntImm("int32", image_features.shape[1] - 1),
            self.config.text_config.hidden_size,
        )

        combined_embeddings = op.tensor_expr_op(
            _concat,
            name_hint="combined_embeddings",
            args=[input_embeddings, image_features, new_shape, insert_index],
        )
        return combined_embeddings

    def embed(self, input_ids: Tensor) -> Tensor:
        return self._embed_input_ids(input_ids)

    def embed_with_pixel_values(self, pixel_values: Tensor, input_ids: Tensor) -> Tensor:
        return self._embed_pixel_values_and_input_ids(pixel_values, input_ids)

    def image_embed(self, pixel_values: Tensor) -> Tensor:
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
            head_dim=self.config.text_config.head_dim,
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
            "embed_with_pixel_values": {
                "pixel_values": nn.spec.Tensor(
                    [
                        1,
                        3,
                        self.config.vision_config.image_size,
                        self.config.vision_config.image_size,
                    ],
                    self.dtype,
                ),
                "input_ids": nn.spec.Tensor([1, "seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "image_embed": {
                "pixel_values": nn.spec.Tensor(
                    [
                        1,
                        3,
                        self.config.vision_config.image_size,
                        self.config.vision_config.image_size,
                    ],
                    self.dtype,
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
