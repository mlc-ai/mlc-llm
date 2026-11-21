"""Qwen2.5 Omni BigVGAN: bounded log-mel clips to 24 kHz waveforms.

This is the waveform stage only. Talker codec generation and DiT sampling are
separate components; input features are 80-bin synthesis mels, not Whisper mels.
"""

import math
from dataclasses import dataclass

import numpy as np
from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


@dataclass
class Qwen25OmniVocoderConfig:
    """Architecture fields from token2wav_config.bigvgan_config."""

    mel_dim: int = 80
    upsample_initial_channel: int = 1536
    upsample_rates: tuple = (5, 3, 2, 2, 2, 2)
    upsample_kernel_sizes: tuple = (11, 7, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    use_bias_at_final: bool = False

    def __post_init__(self):
        for value in (self.mel_dim, self.upsample_initial_channel):
            if type(value) is not int or value <= 0:
                raise ValueError("Vocoder channel sizes must be positive integers")
        if not self.upsample_rates or len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("Vocoder upsampling rates and kernels must match")
        for rate, kernel in zip(self.upsample_rates, self.upsample_kernel_sizes):
            if any(type(x) is not int or x <= 0 for x in (rate, kernel)):
                raise ValueError("Vocoder upsampling sizes must be positive integers")
            if kernel < rate or (kernel - rate) % 2:
                raise ValueError("Vocoder upsampling must preserve exact frame multiples")
        if self.upsample_initial_channel % (2 ** len(self.upsample_rates)):
            raise ValueError("Vocoder channels must halve at every upsampling stage")
        if not self.resblock_kernel_sizes or len(self.resblock_kernel_sizes) != len(
            self.resblock_dilation_sizes
        ):
            raise ValueError("Vocoder residual kernels and dilations must match")
        for kernel, dilations in zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes):
            if type(kernel) is not int or kernel <= 0 or kernel % 2 == 0:
                raise ValueError("Vocoder residual kernels must be positive odd integers")
            if len(dilations) != 3 or any(type(x) is not int or x <= 0 for x in dilations):
                raise ValueError("Vocoder residual blocks require three positive dilations")
        if self.use_bias_at_final is not False:
            raise ValueError("The pinned vocoder uses no final convolution bias")

    @property
    def samples_per_frame(self):
        return math.prod(self.upsample_rates)


def vocoder_filter() -> np.ndarray:
    """Non-persistent 12-tap Kaiser-windowed sinc filter from the HF definition."""
    attenuation = 2.285 * 5 * math.pi * 1.2 + 7.95
    beta = 0.1102 * (attenuation - 8.7)
    time = np.arange(-6, 6, dtype="float32") + np.float32(0.5)
    window = np.kaiser(12, beta).astype("float32")
    values = np.float32(0.5) * window * np.sinc(np.float32(0.5) * time)
    return (values / values.sum()).astype("float32").reshape(1, 1, 12)


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        self.alpha = nn.Parameter((channels,))
        self.beta = nn.Parameter((channels,))

    def forward(self, hidden: Tensor) -> Tensor:
        alpha = op.exp(op.reshape(self.alpha, (1, -1, 1)))
        beta = op.exp(op.reshape(self.beta, (1, -1, 1)))

        def sine_elementwise(value):
            from tvm_tirx import tirx

            return te.compute(value.shape, lambda *indices: tirx.sin(value[indices]))

        sine = op.tensor_expr_op(sine_elementwise, "vocoder_snake_sine", [hidden * alpha])
        return hidden + op.square(sine) / (beta + 1e-9)


class AntiAliasedActivation(nn.Module):
    def __init__(self, channels):
        self.channels = channels
        self.act = SnakeBeta(channels)
        self.filter = nn.Parameter((channels, 1, 12))

    def forward(self, hidden: Tensor) -> Tensor:
        # Polyphase FIR implements the reference replicate-pad/transposed-conv/crop
        # exactly, without materializing inserted zeros or a grouped transpose.
        def upsample(value, kernel):
            from tvm_tirx import tirx

            tap = te.reduce_axis((0, 6), name="tap")
            length = value.shape[2]
            return te.compute(
                (value.shape[0], value.shape[1], 2 * length),
                lambda b, c, t: te.sum(
                    2
                    * value[b, c, tirx.min(tirx.max((t + 15) // 2 - tap - 5, 0), length - 1)]
                    * kernel[c, 0, 2 * tap + (t + 15) % 2],
                    axis=tap,
                ),
            )

        def downsample(value, kernel):
            from tvm_tirx import tirx

            tap = te.reduce_axis((0, 12), name="tap")
            length = value.shape[2]
            return te.compute(
                (value.shape[0], value.shape[1], length // 2),
                lambda b, c, t: te.sum(
                    value[b, c, tirx.min(tirx.max(2 * t + tap - 5, 0), length - 1)]
                    * kernel[c, 0, tap],
                    axis=tap,
                ),
            )

        hidden = op.tensor_expr_op(upsample, "vocoder_upsample_fir", [hidden, self.filter])
        hidden = self.act(hidden)
        return op.tensor_expr_op(downsample, "vocoder_downsample_fir", [hidden, self.filter])


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel, dilations):
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1D(channels, channels, kernel, dilation=d, padding=(kernel - 1) * d // 2)
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1D(channels, channels, kernel, padding=(kernel - 1) // 2) for _ in dilations]
        )
        self.activations = nn.ModuleList([AntiAliasedActivation(channels) for _ in range(6)])

    def forward(self, hidden: Tensor) -> Tensor:
        for index in range(3):
            residual = hidden
            hidden = self.convs1[index](self.activations[2 * index](hidden))
            hidden = self.convs2[index](self.activations[2 * index + 1](hidden))
            hidden = residual + hidden
        return hidden


class Qwen25OmniVocoder(nn.Module):
    """Stateless full-clip waveform decoder; arbitrary clip stitching is not supported."""

    def __init__(self, config: Qwen25OmniVocoderConfig):
        self.config = config
        width = config.upsample_initial_channel
        self.conv_pre = nn.Conv1D(config.mel_dim, width, 7, padding=3)
        self.ups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ConvTranspose1D(
                            width // 2**i,
                            width // 2 ** (i + 1),
                            kernel,
                            stride=rate,
                            padding=(kernel - rate) // 2,
                        )
                    ]
                )
                for i, (rate, kernel) in enumerate(
                    zip(config.upsample_rates, config.upsample_kernel_sizes)
                )
            ]
        )
        self.resblocks = nn.ModuleList(
            [
                AMPBlock(width // 2 ** (i + 1), kernel, dilation)
                for i in range(len(config.upsample_rates))
                for kernel, dilation in zip(
                    config.resblock_kernel_sizes, config.resblock_dilation_sizes
                )
            ]
        )
        self.activation_post = AntiAliasedActivation(width // 2 ** len(config.upsample_rates))
        self.conv_post = nn.Conv1D(
            width // 2 ** len(config.upsample_rates), 1, 7, padding=3, bias=False
        )

    def forward(self, log_mel: Tensor) -> Tensor:
        amplitude = op.exp(op.reshape(log_mel, (1, self.config.mel_dim, log_mel.shape[1])))
        floor = op.full((), float(np.exp(-115 / 20 * math.log(10))), "float32")
        decibels = op.log(op.maximum(amplitude, floor)) * (20 / math.log(10)) - 20
        normalized = 2 * ((decibels + 115) / 115) - 1
        hidden = self.conv_pre(
            op.minimum(
                op.maximum(normalized, op.full((), -1, "float32")), op.full((), 1, "float32")
            )
        )
        count = len(self.config.resblock_kernel_sizes)
        for i, up in enumerate(self.ups):
            hidden = up[0](hidden)
            results = [self.resblocks[i * count + j](hidden) for j in range(count)]
            hidden = results[0]
            for value in results[1:]:
                hidden = hidden + value
            hidden = hidden / count
        hidden = self.conv_post(self.activation_post(hidden))
        hidden = op.minimum(
            op.maximum(hidden, op.full((), -1, "float32")), op.full((), 1, "float32")
        )
        return op.reshape(hidden, (hidden.shape[2],))

    def to(self, dtype=None):
        if dtype is not None and dtype != "float32":
            raise ValueError("Qwen2.5 Omni waveform decoding requires float32")
        super().to(dtype=dtype)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "log_mel": nn.spec.Tensor((self.config.mel_dim, "frames"), "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )
