"""Qwen2.5 Omni audio tower with separate chunk encoding and per-clip projection.

Inputs are Whisper log-mel features, not PCM. Each encode call receives a
zero-padded chunk of 2*n_window frames and its valid length. Concatenate valid outputs
before projection, separately for each clip, so pooling never crosses clips.
"""

import math
from dataclasses import dataclass

import numpy as np
from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


@dataclass
class Qwen25OmniAudioConfig:
    """Audio tower dimensions from thinker_config.audio_config."""

    num_mel_bins: int = 128
    d_model: int = 1280
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_layers: int = 32
    output_dim: int = 2048
    n_window: int = 100
    max_source_positions: int = 1500
    activation_function: str = "gelu"

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name != "activation_function":
                value = getattr(self, name)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
        if self.d_model < 4 or self.d_model % 2:
            raise ValueError("Audio sinusoidal positions require an even width >= 4")
        if self.d_model % self.encoder_attention_heads:
            raise ValueError("Audio width must be divisible by the head count")
        if self.n_window > self.max_source_positions:
            raise ValueError("Audio window exceeds positional embedding capacity")
        if self.activation_function != "gelu":
            raise ValueError("Only exact GELU audio checkpoints are supported")


def audio_positions(config: Qwen25OmniAudioConfig) -> np.ndarray:
    """Generate the non-persistent HF sinusoidal table in float32."""
    increment = math.log(10000) / (config.d_model // 2 - 1)
    frequencies = np.exp(-increment * np.arange(config.d_model // 2, dtype="float32"))
    angles = np.arange(config.n_window, dtype="float32")[:, None] * frequencies[None, :]
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


class AudioAttention(nn.Module):
    def __init__(self, config: Qwen25OmniAudioConfig):
        width = config.d_model
        self.heads = config.encoder_attention_heads
        self.head_dim = width // self.heads
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)

    def forward(self, hidden: Tensor, valid_frames: Tensor) -> Tensor:
        length, width = hidden.shape

        def heads(value):
            return op.permute_dims(
                op.reshape(value, (length, self.heads, self.head_dim)), (1, 0, 2)
            )

        query, key, value = (
            heads(self.q_proj(hidden)),
            heads(self.k_proj(hidden)),
            heads(self.v_proj(hidden)),
        )
        scores = op.matmul(query, op.permute_dims(key, (0, 2, 1))) * self.head_dim**-0.5

        def mask_keys(values, length):
            from tvm_tirx import tirx

            return te.compute(
                values.shape,
                lambda h, q, k: tirx.if_then_else(
                    k < (length[0] + 1) // 2,
                    values[h, q, k],
                    tirx.const(float(np.finfo(str(values.dtype)).min), values.dtype),
                ),
            )

        scores = op.tensor_expr_op(mask_keys, "audio_attention_mask", [scores, valid_frames])
        probabilities = op.astype(op.softmax(op.astype(scores, "float32"), axis=-1), hidden.dtype)
        output = op.matmul(probabilities, value)
        return self.out_proj(op.reshape(op.permute_dims(output, (1, 0, 2)), (length, width)))


class AudioLayer(nn.Module):
    def __init__(self, config: Qwen25OmniAudioConfig):
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model, bias=True)

    def forward(self, hidden: Tensor, valid_frames: Tensor) -> Tensor:
        hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden), valid_frames)
        hidden = hidden + self.fc2(op.gelu(self.fc1(self.final_layer_norm(hidden))))
        if hidden.dtype == "float16":
            hidden = op.minimum(
                op.maximum(hidden, op.full((), -64504.0, "float16")),
                op.full((), 64504.0, "float16"),
            )
        return hidden


class Qwen25OmniAudioEncoder(nn.Module):
    """Stateless compiled audio component; request chunking is performed by its caller."""

    def __init__(self, config: Qwen25OmniAudioConfig):
        self.config = config
        self.dtype = "float32"
        self.conv1 = nn.Conv1D(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = nn.Conv1D(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.positions = nn.Parameter((config.n_window, config.d_model))
        self.layers = nn.ModuleList([AudioLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj = nn.Linear(config.d_model, config.output_dim, bias=True)

    def encode_chunk(self, input_features: Tensor, valid_frames: Tensor) -> Tensor:
        hidden = op.gelu(self.conv1(input_features))

        def mask_padding(values, length):
            from tvm_tirx import tirx

            return te.compute(
                values.shape,
                lambda b, d, t: tirx.if_then_else(
                    t < length[0], values[b, d, t], tirx.const(0, values.dtype)
                ),
            )

        hidden = op.tensor_expr_op(mask_padding, "audio_conv_mask", [hidden, valid_frames])
        hidden = op.gelu(self.conv2(hidden))
        hidden = op.reshape(
            op.permute_dims(hidden, (0, 2, 1)), (hidden.shape[2], self.config.d_model)
        )

        def add_positions(values, positions):
            return te.compute(values.shape, lambda t, d: values[t, d] + positions[t, d])

        hidden = op.tensor_expr_op(add_positions, "audio_positions", [hidden, self.positions])
        for layer in self.layers:
            hidden = layer(hidden, valid_frames)
        return hidden

    def project(self, hidden: Tensor) -> Tensor:
        pooled = op.sum(hidden, axis=1) / 2
        return self.proj(self.ln_post(pooled))

    def to(self, dtype=None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "encode_chunk": {
                    "input_features": nn.spec.Tensor(
                        (1, self.config.num_mel_bins, 2 * self.config.n_window), self.dtype
                    ),
                    "valid_frames": nn.spec.Tensor((1,), "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "project": {
                    "hidden": nn.spec.Tensor(("pooled_frames", 2, self.config.d_model), self.dtype),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
            },
            self,
        )
