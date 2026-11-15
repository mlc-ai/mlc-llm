"""Stateless flow predictor with explicit, precomputed reference-speaker conditioning.

The ECAPA reference-mel encoder and Talker are separate stages. This module keeps
noise and time explicit so the pinned RK4 sampler can be checked independently.
"""

import math
from dataclasses import dataclass

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


@dataclass
class Qwen25OmniDiTConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    head_dim: int = 64
    ff_mult: int = 2
    emb_dim: int = 512
    mel_dim: int = 80
    enc_dim: int = 128
    enc_emb_dim: int = 192
    repeats: int = 2
    num_embeds: int = 8193
    max_position_embeddings: int = 32768
    block_size: int = 24
    look_ahead_layers: tuple = (10,)
    look_backward_layers: tuple = (0, 20)
    rope_theta: float = 10000.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name in ("look_ahead_layers", "look_backward_layers", "rope_theta"):
                continue
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.head_dim % 2:
            raise ValueError("DiT RoPE requires an even head dimension")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("Invalid DiT RoPE base")
        for layers in (self.look_ahead_layers, self.look_backward_layers):
            if any(type(i) is not int or not 0 <= i < self.num_hidden_layers for i in layers):
                raise ValueError("Invalid block-attention layer index")


class Identity(nn.Module):
    def forward(self, value):
        return value


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        self.time_mlp = nn.ModuleList([nn.Linear(256, width), Identity(), nn.Linear(width, width)])

    def forward(self, time):
        def positions(value):
            from tvm_tirx import tirx

            return te.compute(
                (1, 256),
                lambda b, d: tirx.if_then_else(
                    d < 128,
                    tirx.sin(
                        value[0]
                        * 1000
                        * tirx.exp(-(d % 128).astype("float32") * (math.log(10000) / 127))
                    ),
                    tirx.cos(
                        value[0]
                        * 1000
                        * tirx.exp(-(d % 128).astype("float32") * (math.log(10000) / 127))
                    ),
                ),
            )

        hidden = op.tensor_expr_op(positions, "dit_time_embedding", [time])
        return self.time_mlp[2](op.silu(self.time_mlp[0](hidden)))


class CodecEmbedding(nn.Module):
    def __init__(self, config):
        self.codec_embed = nn.Embedding(config.num_embeds + 1, config.emb_dim)
        self.repeats = config.repeats

    def forward(self, code, noise):
        def repeat(value, frames):
            return te.compute(
                (frames.shape[0], value.shape[1]), lambda t, d: value[t // self.repeats, d]
            )

        return op.tensor_expr_op(repeat, "dit_repeat_codec", [self.codec_embed(code), noise])


class InputEmbedding(nn.Module):
    def __init__(self, config):
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.emb_dim + config.enc_emb_dim,
            config.hidden_size,
        )

    def forward(self, noise, reference, code, speaker):
        def expand(vector, frames):
            return te.compute((frames.shape[0], vector.shape[0]), lambda t, d: vector[d])

        reference = op.tensor_expr_op(expand, "dit_reference_expand", [reference, noise])
        speaker = op.tensor_expr_op(expand, "dit_speaker_expand", [speaker, noise])
        return self.proj(op.concat([noise, reference, code, speaker], dim=-1))


class AdaNorm(nn.Module):
    def __init__(self, width, final=False):
        self.final = final
        self.linear = nn.Linear(width, width * (2 if final else 6))
        self.norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)

    def forward(self, hidden, time):
        parts = op.split(self.linear(op.silu(time)), 2 if self.final else 6, axis=-1)
        if self.final:
            scale, shift = parts
            return self.norm(hidden) * (1 + scale) + shift
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = parts
        return self.norm(hidden) * (1 + scale) + shift, gate, mlp_shift, mlp_scale, mlp_gate


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        self.inv_freq = nn.Parameter((config.head_dim // 2,))


class Attention(nn.Module):
    def __init__(self, config, ahead, backward):
        self.config, self.ahead, self.backward = config, ahead, backward
        inner = config.num_attention_heads * config.head_dim
        self.to_q = nn.Linear(config.hidden_size, inner)
        self.to_k = nn.Linear(config.hidden_size, inner)
        self.to_v = nn.Linear(config.hidden_size, inner)
        self.to_out = nn.ModuleList([nn.Linear(inner, config.hidden_size), Identity()])

    def forward(self, hidden, frequencies):
        c = self.config

        def heads(value):
            return op.permute_dims(
                op.reshape(value, (hidden.shape[0], c.num_attention_heads, c.head_dim)), (1, 0, 2)
            )

        def rotary(value, frequency):
            from tvm_tirx import tirx

            half = c.head_dim // 2

            def element(h, t, d):
                index = d % half
                angle = t.astype("float32") * frequency[index]
                first, second = value[h, t, 2 * index], value[h, t, 2 * index + 1]
                rotated = tirx.if_then_else(
                    d < half,
                    first * tirx.cos(angle) - second * tirx.sin(angle),
                    second * tirx.cos(angle) + first * tirx.sin(angle),
                )
                return tirx.if_then_else(h == 0, rotated, value[h, t, d])

            return te.compute(value.shape, element)

        query = op.tensor_expr_op(
            rotary, "dit_first_head_rope", [heads(self.to_q(hidden)), frequencies]
        )
        key = op.tensor_expr_op(
            rotary, "dit_first_head_rope", [heads(self.to_k(hidden)), frequencies]
        )
        value = heads(self.to_v(hidden))
        scores = op.matmul(query, op.permute_dims(key, (0, 2, 1))) * c.head_dim**-0.5

        def mask(value):
            from tvm_tirx import tirx

            return te.compute(
                value.shape,
                lambda h, q, k: tirx.if_then_else(
                    tirx.all(
                        k // c.block_size - q // c.block_size >= -self.backward,
                        k // c.block_size - q // c.block_size <= self.ahead,
                    ),
                    value[h, q, k],
                    tirx.const(-3.4028234663852886e38, "float32"),
                ),
            )

        scores = op.tensor_expr_op(mask, "dit_block_mask", [scores])
        attended = op.matmul(op.softmax(scores, axis=-1), value)
        attended = op.reshape(
            op.permute_dims(attended, (1, 0, 2)),
            (hidden.shape[0], c.num_attention_heads * c.head_dim),
        )
        return self.to_out[0](attended)


class MLP(nn.Module):
    def __init__(self, c):
        self.ff = nn.ModuleList(
            [
                nn.Linear(c.hidden_size, c.hidden_size * c.ff_mult),
                Identity(),
                Identity(),
                nn.Linear(c.hidden_size * c.ff_mult, c.hidden_size),
            ]
        )

    def forward(self, value):
        return self.ff[3](op.gelu(self.ff[0](value), approximate="tanh"))


class DiTLayer(nn.Module):
    def __init__(self, c, index):
        self.attn_norm = AdaNorm(c.hidden_size)
        self.attn = Attention(
            c, int(index in c.look_ahead_layers), int(index in c.look_backward_layers)
        )
        self.ff_norm = nn.LayerNorm(c.hidden_size, eps=1e-6, elementwise_affine=False)
        self.ff = MLP(c)

    def forward(self, hidden, time, frequencies):
        norm, gate, shift, scale, mlp_gate = self.attn_norm(hidden, time)
        hidden = hidden + gate * self.attn(norm, frequencies)
        return hidden + mlp_gate * self.ff(self.ff_norm(hidden) * (1 + scale) + shift)


class Qwen25OmniDiT(nn.Module):
    def __init__(self, config):
        self.config = config
        self.time_embed = TimeEmbedding(config.hidden_size)
        self.text_embed = CodecEmbedding(config)
        self.rotary_embed = RotaryEmbedding(config)
        self.input_embed = InputEmbedding(config)
        self.transformer_blocks = nn.ModuleList(
            [DiTLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm_out = AdaNorm(config.hidden_size, final=True)
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

    def forward(
        self, noise: Tensor, time: Tensor, code: Tensor, reference: Tensor, speaker: Tensor
    ):
        hidden = self.input_embed(noise, reference, self.text_embed(code, noise), speaker)
        timestep = self.time_embed(time)
        for layer in self.transformer_blocks:
            hidden = layer(hidden, timestep, self.rotary_embed.inv_freq)
        return self.proj_out(self.norm_out(hidden, timestep))

    def to(self, dtype=None):
        if dtype not in (None, "float32"):
            raise ValueError("DiT flow prediction supports float32 only")
        return super().to(dtype)

    def get_default_spec(self):
        c = self.config
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "noise": nn.spec.Tensor(["frames", c.mel_dim], "float32"),
                    "time": nn.spec.Tensor([1], "float32"),
                    "code": nn.spec.Tensor(["codes"], "int32"),
                    "reference": nn.spec.Tensor([c.enc_dim], "float32"),
                    "speaker": nn.spec.Tensor([c.enc_emb_dim], "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )
