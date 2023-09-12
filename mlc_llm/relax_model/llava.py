# pylint: disable=invalid-name,missing-docstring
import dataclasses
from typing import Tuple, Optional

import tvm
from tvm import te, tir, relax
import tvm.relax.frontend.nn as nn
from tvm.relax.frontend.nn import (
    Embedding,
    KVCache,
    Linear,
    LayerNorm,
    Conv2D,
    Module,
    ModuleList,
    Parameter,
    Tensor,
    take,
    tensor_expr_op,
    concat,
    permute_dims,
    reshape,
    squeeze,
    matmul,
    maximum,
    minimum,
    softmax,
    gelu,
    zeros,
    scaled_dot_product_attention,
    print_,
)


@dataclasses.dataclass
class LlavaConfig:
    image_size: int = 224
    num_channels: int = 3
    hidden_size: int = 1024
    projection_dim: int = 768
    patch_size: int = 14
    grid_size: int = 16
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-5
    dtype: str = "float16"


class LlavaVisionEmbeddings(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.class_embedding = Parameter((config.hidden_size,), dtype=config.dtype)
        self.conv1 = Conv2D(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
            dtype=config.dtype,
        )
        self.position_embedding = Parameter(
            (config.grid_size**2 + 1, config.hidden_size), dtype=config.dtype
        )
        self.ln_pre = LayerNorm(config.hidden_size, dtype=config.dtype)
        self.dtype = config.dtype

    def forward(self, image: Tensor) -> Tensor:
        x = self.conv1(image)
        bs, width, grid1, grid2 = x.shape
        x = reshape(x, (bs, width, grid1 * grid2))
        x = permute_dims(x, (0, 2, 1))
        zero = zeros((bs, 1, width), dtype=self.dtype)
        x = concat([self.class_embedding.astype(self.dtype) + zero, x], dim=1)
        x = x + self.position_embedding.astype(self.dtype)
        x = self.ln_pre(x)
        x = permute_dims(x, (1, 0, 2))

        return x


class LlavaQuickGELU(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.const = relax.const(1.702, dtype=config.dtype)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * relax.op.sigmoid(input_tensor * self.const)


class LlavaCLIPEncoderLayer(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.ln1 = LayerNorm(config.hidden_size, dtype=config.dtype)
        self.ln2 = LayerNorm(config.hidden_size, dtype=config.dtype)
        self.c_fc = Linear(config.hidden_size, config.hidden_size * 4, dtype=config.dtype)
        self.gelu = LlavaQuickGELU(config)
        self.c_proj = Linear(config.hidden_size * 4, config.hidden_size, dtype=config.dtype)
        self.dtype = config.dtype

    def forward(self, input_tensor: Tensor) -> Tensor:
        ln = self.ln1(input_tensor)
        attn = scaled_dot_product_attention(ln, ln, ln)
        attn = squeeze(take(attn, 0, axis=0), axis=0)  # perform attn[0]
        x = input_tensor + attn
        ln2 = self.ln2(x)
        ln2 = self.c_fc(ln2)
        ln2 = self.gelu(ln2)
        ln2 = self.c_proj(ln2)
        x = x + ln2

        return x


class LlavaCLIPEncoder(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.layers = ModuleList(
            [LlavaCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_post = LayerNorm(config.hidden_size, dtype=config.dtype)

    def forward(self, input_tensor: Tensor) -> Tensor:
        x = self.layers(input_tensor)
        x = permute_dims(x, (1, 0, 2))
        x = squeeze(take(x, 0, axis=1), axis=1)  # perform x[:, 0, :]
        x = self.ln_post(x)

        return x


class LlavaVisionModel(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

    def forward(self, image: Tensor) -> Tensor:
        pass


def get_model(args, hf_config):
    pass
