"""Qwen2.5 Omni reference-mel conditioner, including the nonzero null embedding.

Inputs are unpadded synthesis-mel clips [frames, mel_dim], not Whisper features.
The caller must enforce min_frames before invoking the raw compiled function.
"""

from dataclasses import dataclass

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.loader import ExternMapping


@dataclass
class Qwen25OmniECAPAConfig:
    mel_dim: int = 80
    enc_dim: int = 128
    enc_channels: tuple = (256, 256, 256, 256, 768)
    enc_kernel_sizes: tuple = (5, 3, 3, 3, 1)
    enc_dilations: tuple = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 64
    enc_res2net_scale: int = 2
    enc_se_channels: int = 64

    def __post_init__(self):
        sequences = (self.enc_channels, self.enc_kernel_sizes, self.enc_dilations)
        if len(self.enc_channels) < 3 or any(len(x) != len(self.enc_channels) for x in sequences):
            raise ValueError("ECAPA channel, kernel and dilation lists must align")
        values = [
            self.mel_dim,
            self.enc_dim,
            self.enc_attention_channels,
            self.enc_res2net_scale,
            self.enc_se_channels,
        ]
        if any(type(x) is not int or x < 1 for x in values + [x for seq in sequences for x in seq]):
            raise ValueError("ECAPA dimensions must be positive integers")
        if self.enc_res2net_scale < 2 or any(k % 2 != 1 for k in self.enc_kernel_sizes):
            raise ValueError("ECAPA requires scale >= 2 and odd convolution kernels")
        width = self.enc_channels[0]
        if any(c != width for c in self.enc_channels[:-1]) or width % self.enc_res2net_scale:
            raise ValueError("ECAPA residual channels must agree and divide into scale groups")
        if self.enc_channels[-1] != sum(self.enc_channels[1:-1]):
            raise ValueError("ECAPA aggregation channels must equal concatenated residual widths")

    @property
    def min_frames(self):
        return 1 + max((k - 1) * d // 2 for k, d in zip(self.enc_kernel_sizes, self.enc_dilations))


class ReflectConv(nn.Conv1D):
    """Same-length odd convolution with PyTorch reflection, retaining weight names."""

    def __init__(self, in_channels, out_channels, kernel, dilation=1):
        super().__init__(in_channels, out_channels, kernel, dilation=dilation)
        self.reflect_padding = (kernel - 1) * dilation // 2
        self.pointwise = kernel == 1

    def forward(self, hidden):
        if self.pointwise:
            # NLC matmul preserves the checkpoint's OI1 weights and avoids the
            # low-batch GEMV scheduler's unsupported dynamic NCW convolution.
            weight = op.reshape(self.weight, (self.weight.shape[0], self.weight.shape[1]))
            value = op.matmul(op.permute_dims(hidden, (0, 2, 1)), op.permute_dims(weight, (1, 0)))
            if self.bias is not None:
                value = value + self.bias
            return op.permute_dims(value, (0, 2, 1))
        padding = self.reflect_padding
        if padding:

            def reflect(value):
                from tvm_tirx import tirx

                length = value.shape[2]
                return te.compute(
                    (value.shape[0], value.shape[1], length + 2 * padding),
                    lambda b, c, t: value[
                        b,
                        c,
                        tirx.if_then_else(
                            t < padding,
                            padding - t,
                            tirx.if_then_else(
                                t - padding >= length, 2 * length - 2 - (t - padding), t - padding
                            ),
                        ),
                    ],
                )

            hidden = op.tensor_expr_op(reflect, "ecapa_reflect", [hidden])
        return super().forward(hidden)


class TimeDelay(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, dilation=1):
        self.conv = ReflectConv(in_channels, out_channels, kernel, dilation)

    def forward(self, hidden):
        return op.relu(self.conv(hidden))


def temporal_mean(hidden):
    def compute(total, value):
        return te.compute(
            total.shape, lambda b, c, t: total[b, c, t] / value.shape[2].astype("float32")
        )

    return op.tensor_expr_op(compute, "ecapa_mean", [op.sum(hidden, axis=2, keepdims=True), hidden])


class Res2Net(nn.Module):
    def __init__(self, width, scale, kernel, dilation):
        self.scale = scale
        self.blocks = nn.ModuleList(
            [TimeDelay(width // scale, width // scale, kernel, dilation) for _ in range(scale - 1)]
        )

    def forward(self, hidden):
        chunks = op.split(hidden, self.scale, axis=1)
        outputs = [chunks[0]]
        for i in range(1, self.scale):
            value = chunks[i] if i == 1 else chunks[i] + outputs[-1]
            outputs.append(self.blocks[i - 1](value))
        return op.concat(outputs, dim=1)


class SqueezeExcitation(nn.Module):
    def __init__(self, width, inner):
        self.conv1 = ReflectConv(width, inner, 1)
        self.conv2 = ReflectConv(inner, width, 1)

    def forward(self, hidden):
        gate = op.sigmoid(self.conv2(op.relu(self.conv1(temporal_mean(hidden)))))
        return hidden * gate


class Residual(nn.Module):
    def __init__(self, width, scale, se_channels, kernel, dilation):
        self.tdnn1 = TimeDelay(width, width)
        self.res2net_block = Res2Net(width, scale, kernel, dilation)
        self.tdnn2 = TimeDelay(width, width)
        self.se_block = SqueezeExcitation(width, se_channels)

    def forward(self, hidden):
        return hidden + self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(hidden))))


class StatisticsPooling(nn.Module):
    def __init__(self, width, attention_width):
        self.tdnn = TimeDelay(3 * width, attention_width)
        self.conv = ReflectConv(attention_width, width, 1)

    def forward(self, hidden):
        mean = temporal_mean(hidden)
        std = op.sqrt(
            op.maximum(temporal_mean(op.square(hidden - mean)), op.full((), 1e-12, "float32"))
        )

        def expand(value, frames):
            return te.compute(frames.shape, lambda b, c, t: value[b, c, 0])

        context = op.concat(
            [
                hidden,
                op.tensor_expr_op(expand, "ecapa_mean_expand", [mean, hidden]),
                op.tensor_expr_op(expand, "ecapa_std_expand", [std, hidden]),
            ],
            dim=1,
        )
        attention = op.softmax(self.conv(op.tanh(self.tdnn(context))), axis=2)
        mean = op.sum(attention * hidden, axis=2, keepdims=True)
        variance = op.sum(attention * op.square(hidden - mean), axis=2, keepdims=True)
        std = op.sqrt(op.maximum(variance, op.full((), 1e-12, "float32")))
        return op.concat([mean, std], dim=1)


class Qwen25OmniECAPA(nn.Module):
    def __init__(self, config):
        self.config = config
        c = config
        self.blocks = nn.ModuleList(
            [TimeDelay(c.mel_dim, c.enc_channels[0], c.enc_kernel_sizes[0], c.enc_dilations[0])]
            + [
                Residual(
                    c.enc_channels[i],
                    c.enc_res2net_scale,
                    c.enc_se_channels,
                    c.enc_kernel_sizes[i],
                    c.enc_dilations[i],
                )
                for i in range(1, len(c.enc_channels) - 1)
            ]
        )
        width = c.enc_channels[-1]
        self.mfa = TimeDelay(width, width, c.enc_kernel_sizes[-1], c.enc_dilations[-1])
        self.asp = StatisticsPooling(width, c.enc_attention_channels)
        self.fc = ReflectConv(width * 2, c.enc_dim, 1)

    def forward(self, mel: Tensor):
        hidden = op.reshape(op.permute_dims(mel, (1, 0)), (1, self.config.mel_dim, mel.shape[0]))
        outputs = []
        for block in self.blocks:
            hidden = block(hidden)
            outputs.append(hidden)
        return op.reshape(
            self.fc(self.asp(self.mfa(op.concat(outputs[1:], dim=1)))), (self.config.enc_dim,)
        )

    def to(self, dtype=None):
        if dtype not in (None, "float32"):
            raise ValueError("ECAPA conditioner supports float32 only")
        return super().to(dtype)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "mel": nn.spec.Tensor(["frames", self.config.mel_dim], "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )


def huggingface(config: Qwen25OmniECAPAConfig) -> ExternMapping:
    """Map the exact standalone reference conditioner checkpoint subtree."""
    model = Qwen25OmniECAPA(config)
    _, parameters, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, _ in parameters:
        mapping.add_mapping(
            name,
            [f"token2wav.code2wav_dit_model.input_embed.spk_encoder.{name}"],
            lambda value: value.astype("float32", copy=False),
        )
    return mapping
