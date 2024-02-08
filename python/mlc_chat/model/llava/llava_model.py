import dataclasses
from typing import Optional

from tvm import te, tir
import tvm.relax as relax
import tvm.relax.frontend.nn as nn
from tvm.relax.frontend.nn import op

from tvm.relax.frontend.nn import Tensor, Module
from tvm.relax.frontend.nn.op import (
    reshape,
    permute_dims,
    broadcast_to,
    concat,
    softmax,
    matmul,
    wrap_nested,
)

from tvm.relax.op import strided_slice, arange

import dataclasses
import logging
from typing import Any, Dict, Optional

from tvm import te, tir

from ...support.config import ConfigBase
from ...support.style import bold


from ..llama.llama_model import LlamaForCasualLM, LlamaConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LlavaVisionConfig(ConfigBase):
    hidden_size: int
    image_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    projection_dim: int
    vocab_size: int
    dtype: str = "float16"
    num_channels: int = 3
    layer_norm_eps: float = 1e-06
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LlavaConfig(ConfigBase):
    image_token_index: int
    text_config: LlamaConfig
    vision_config: LlavaVisionConfig
    vocab_size: int
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    dtype: str = "float16"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):

        vision_config = {}
        for k, v in self.vision_config.items():
            if k == "kwargs":
                for kwargs_k, kwargs_v in v.items():
                    vision_config[kwargs_k] = kwargs_v
                continue
            vision_config[k] = v
        self.vision_config = LlavaVisionConfig.from_dict(vision_config)

        text_config = {}
        if "_name_or_path" in self.text_config:
            if self.text_config["_name_or_path"] == "meta-llama/Llama-2-7b-hf":
                text_config["hidden_size"] = self.text_config.pop("hidden_size", 4096)
                text_config["intermediate_size"] = self.text_config.pop("intermediate_size", 11008)
                text_config["num_attention_heads"] = self.text_config.pop("num_attention_heads", 32)
                text_config["num_hidden_layers"] = self.text_config.pop("num_hidden_layers", 32)
                text_config["rms_norm_eps"] = self.text_config.pop("rms_norm_eps", 1e-06)
                text_config["vocab_size"] = self.text_config.pop("vocab_size", 32000)  # 32064
                text_config["context_window_size"] = self.text_config.pop(
                    "context_window_size", 4096
                )
        else:
            for k, v in self.text_config.items():
                if k == "kwargs":
                    for kwargs_k, kwargs_v in v.items():
                        text_config[kwargs_k] = kwargs_v
                    continue
                text_config[k] = v
        self.text_config = LlamaConfig.from_dict(text_config)

        if self.context_window_size <= 0:
            self.context_window_size = self.text_config.context_window_size

        if self.prefill_chunk_size <= 0:
            self.prefill_chunk_size = self.text_config.prefill_chunk_size


class CLIPVisionEmbeddings(Module):
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter((self.embed_dim,), dtype=config.dtype)
        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
            dtype=config.dtype,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(
            num=self.num_positions, dim=self.embed_dim, dtype=config.dtype
        )

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
    return wrap_nested(relax.op.sigmoid(x._expr), name)


class LlavaQuickGELU(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        pass

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * sigmoid(input_tensor * 1.702)


class CLIPMLP(Module):
    def __init__(self, config: LlavaVisionConfig):
        super().__init__()
        self.activation_fn = LlavaQuickGELU(config=config)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, dtype=config.dtype)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, dtype=config.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPAttention(Module):
    def __init__(self, config: LlavaConfig):
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
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)

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

        if attn_weights.shape != [bsz * self.num_heads, tgt_len, tgt_len]:
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, tgt_len)}, but is"
                f" {attn_weights.shape}"
            )
        attn_weights = softmax(attn_weights, axis=-1)
        attn_output = matmul(attn_weights, value_states)
        if attn_output.shape != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )
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
        self.layer_norm1 = nn.LayerNorm(
            normalized_shape=self.embed_dim, eps=config.layer_norm_eps, dtype=config.dtype
        )
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            normalized_shape=self.embed_dim, eps=config.layer_norm_eps, dtype=config.dtype
        )

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
        encoder_states = ()
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
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=config.dtype)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=config.dtype)

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
        self.language_model = LlamaForCasualLM(config.text_config)
        self.vocab_size = config.vocab_size
        self.dtype = config.dtype

    def forward(self, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor) -> Tensor:
        return self.language_model(inputs, total_seq_len, attention_mask)

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def decode(self, inputs: Tensor, total_seq_len: tir.Var):
        batch_size, seq_len = inputs.shape
        attention_mask = op.full(
            shape=[batch_size, 1, seq_len, total_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        return self.forward(inputs, total_seq_len, attention_mask)

    def embed(self, pixel_values: Tensor, inputs: Tensor) -> Tensor:

        def _index(x, value, batch_size, seq_len):
            return te.compute(
                (batch_size, seq_len),
                lambda i, j: tir.if_then_else(
                    x[i, j] == value,
                    # tir.IntImm("int32", 1),
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

        image_features_all = self.vision_tower.forward(pixel_values)
        image_features = wrap_nested(
            strided_slice(
                image_features_all._expr, axes=[1], begin=[1], end=[image_features_all.shape[1]]
            ),
            name="slice",
        )
        image_features = self.multi_modal_projector(image_features)
        input_embeddings = self.language_model.model.embed_tokens(inputs)
        batch_size, seq_len = inputs.shape
        image_index_tensor = op.tensor_expr_op(
            _index,
            name_hint="index",
            args=[inputs, tir.IntImm("int32", self.config.image_token_index), batch_size, seq_len],
        ).astype("int32")
        ##! Assume only one <IMAGE> token in input
        ##! Also assume batch_size = 1 for now

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

    def prefill_with_embed(self, embeddings: Tensor, total_seq_len: tir.Var):
        def _attention_mask(batch_size, seq_len, total_seq_len):
            return te.compute(
                (batch_size, 1, seq_len, total_seq_len),
                lambda b, _, i, j: tir.if_then_else(
                    i < j - (total_seq_len - seq_len),
                    tir.min_value(self.dtype),
                    tir.max_value(self.dtype),
                ),
                name="attention_mask_prefill",
            )

        batch_size, seq_len, embed_dim = embeddings.shape

        attention_mask = op.tensor_expr_op(
            _attention_mask,
            name_hint="attention_mask_prefill",
            args=[batch_size, seq_len, total_seq_len],
        )

        return self.language_model.forward(
            embeddings, total_seq_len, attention_mask, input_embeddings=embeddings
        )

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        batch_size = 1
        mod_spec = {
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "total_seq_len": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "embed": {
                "pixel_values": nn.spec.Tensor(
                    [
                        batch_size,
                        3,
                        self.config.vision_config.image_size,
                        self.config.vision_config.image_size,
                    ],
                    "float16",
                ),
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill_with_embed": {
                "embeddings": nn.spec.Tensor(
                    [batch_size, "seq_len", self.config.text_config.hidden_size], "float16"
                ),
                "total_seq_len": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
