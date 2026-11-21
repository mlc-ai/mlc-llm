"""Qwen3-Omni CNN chunks and packed audio attention windows."""

import dataclasses

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from mlc_llm.model.qwen2_5_omni.audio_model import AudioLayer, Qwen25OmniAudioConfig


@dataclasses.dataclass
class Qwen3OmniAudioConfig(Qwen25OmniAudioConfig):
    """Pinned three-convolution audio encoder dimensions."""

    n_window: int = 50
    n_window_infer: int = 800
    downsample_hidden_size: int = 480

    def __post_init__(self):
        super().__post_init__()
        if self.n_window != 50 or self.n_window_infer != 800 or self.downsample_hidden_size <= 0:
            raise ValueError("Unsupported Qwen3-Omni audio window")


def stem_length(frames):
    return (frames + 7) // 8


class Qwen3OmniAudioEncoder(nn.Module):
    """Separate per-chunk CNN from self-attention over packed time windows."""

    def __init__(self, config):
        self.cfg = config
        channels = config.downsample_hidden_size
        width = config.d_model
        self.conv2d1 = nn.Conv2D(1, channels, (3, 3), stride=2, padding=1)
        self.conv2d2 = nn.Conv2D(channels, channels, (3, 3), stride=2, padding=1)
        self.conv2d3 = nn.Conv2D(channels, channels, (3, 3), stride=2, padding=1)
        self.conv_out = nn.Linear(channels * stem_length(config.num_mel_bins), width, bias=False)
        self.positions = nn.Parameter((stem_length(config.n_window * 2), width))
        self.layers = nn.ModuleList([AudioLayer(config) for _ in range(config.encoder_layers)])
        for layer in self.layers:
            layer.self_attn.k_proj = nn.Linear(width, width, bias=True)
        self.ln_post = nn.LayerNorm(width)
        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, config.output_dim)

    def stem(self, features):
        hidden = op.gelu(self.conv2d1(features))
        hidden = op.gelu(self.conv2d2(hidden))
        hidden = op.gelu(self.conv2d3(hidden))
        _, channels, bins, frames = hidden.shape
        hidden = op.permute_dims(hidden, (0, 3, 1, 2)).reshape(frames, channels * bins)
        hidden = self.conv_out(hidden)

        def add_positions(values, positions):
            return te.compute(values.shape, lambda r, c: values[r, c] + positions[r, c])

        return op.tensor_expr_op(add_positions, "audio_positions", [hidden, self.positions])

    def encode_window(self, hidden, valid_frames):
        for layer in self.layers:
            hidden = layer(hidden, valid_frames)
        return self.proj2(op.gelu(self.proj1(self.ln_post(hidden))))

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "stem": {
                    "features": nn.spec.Tensor(
                        (1, 1, self.cfg.num_mel_bins, "mel_frames"), "float32"
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "encode_window": {
                    "hidden": nn.spec.Tensor(("window_frames", self.cfg.d_model), "float32"),
                    "valid_frames": nn.spec.Tensor((1,), "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
            },
            self,
        )
