"""Whisper-compatible PCM frontend for the Qwen2.5 Omni audio component."""

import math

import numpy as np
from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm_tirx import tirx


def whisper_frontend_parameters(mel_filters: np.ndarray) -> dict[str, np.ndarray]:
    """Use the checkpoint processor's Slaney mel bank, with a periodic Hann DFT."""
    if mel_filters.shape != (201, 128) or not np.isfinite(mel_filters).all():
        raise ValueError("Expected a finite 201 by 128 Whisper mel filter bank")
    phase = 2 * math.pi * np.arange(400)[:, None] * np.arange(201)[None, :] / 400
    window = (0.5 - 0.5 * np.cos(2 * math.pi * np.arange(400) / 400))[:, None]
    matrix = np.concatenate((np.cos(phase), -np.sin(phase)), axis=1) * window
    return {"dft": matrix.astype("float32"), "mel_filters": mel_filters.astype("float32")}


class Qwen25OmniAudioFeatures(nn.Module):
    """16 kHz mono PCM, already zero-padded to the processor's clip capacity.

    Center reflection, power spectrum, base-10 log and a clip-wide dynamic
    range floor follow Whisper. The caller crops to ceil(valid_samples / 160)
    mel frames, as the processor's subsampled attention mask does. No automatic
    resampling, amplitude normalization or truncation is performed here.
    """

    def __init__(self, num_samples: int = 4800000):
        if type(num_samples) is not int or num_samples < 400 or num_samples % 160:
            raise ValueError("PCM capacity must be a multiple of 160 and at least 400")
        self.num_samples = num_samples
        self.dft = nn.Parameter((400, 402), "float32")
        self.mel_filters = nn.Parameter((201, 128), "float32")

    def forward(self, samples: Tensor) -> Tensor:
        def frame(waveform):
            def sample(t, i):
                index = t * 160 + i - 200
                reflected = tirx.if_then_else(index < 0, -index, index)
                reflected = tirx.if_then_else(
                    reflected >= self.num_samples, 2 * self.num_samples - 2 - reflected, reflected
                )
                return waveform[reflected]

            return te.compute((self.num_samples // 160, 400), sample)

        frames = op.tensor_expr_op(frame, "whisper_frames", [samples])
        spectrum = op.matmul(frames, self.dft)
        real, imaginary = op.split(spectrum, 2, axis=-1)
        mel = op.matmul(real * real + imaginary * imaginary, self.mel_filters)
        logarithm = op.log(op.maximum(mel, op.full((), 1e-10, "float32"))) / math.log(10)
        logarithm = op.maximum(logarithm, op.max(logarithm, axis=None) - 8)
        return op.permute_dims((logarithm + 4) / 4, (1, 0))

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "samples": nn.spec.Tensor((self.num_samples,), "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )
