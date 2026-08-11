"""Gemma 4 audio preprocessing and encoder."""

from __future__ import annotations

import math

import numpy as np
from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .gemma4_config import Gemma4AudioConfig, Gemma4TextConfig


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMSNorm, including its explicit float32 accumulation."""

    def __init__(self, hidden_size: int, eps: float, with_scale: bool = True):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter((hidden_size,)) if with_scale else None

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        values = op.astype(hidden_states, "float32")
        variance = op.sum(values * values, axis=-1, keepdims=True) / self.hidden_size
        values = values / op.sqrt(variance + self.eps)
        if self.weight is not None:
            values = values * op.astype(self.weight, "float32")
        return op.astype(values, dtype)


class Gemma4ClippableLinear(nn.Module):
    """Checkpoint-compatible weight-clipped linear layer."""

    def __init__(self, config: Gemma4AudioConfig, in_features: int, out_features: int):
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.input_min = nn.Parameter(())
            self.input_max = nn.Parameter(())
            self.output_min = nn.Parameter(())
            self.output_max = nn.Parameter(())

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_clipped_linears:
            hidden_states = op.minimum(op.maximum(hidden_states, self.input_min), self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = op.minimum(op.maximum(hidden_states, self.output_min), self.output_max)
        return hidden_states


class Gemma4ScaleLayerNorm(nn.Module):
    """LayerNorm with a learned scale and no learned bias."""

    def __init__(self, hidden_size: int, eps: float):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter((hidden_size,))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return op.layer_norm(
            hidden_states,
            normalized_shape=self.hidden_size,
            weight=self.weight,
            bias=None,
            eps=self.eps,
        )


class Gemma4AudioFeatureExtractor(nn.Module):
    """Compiled 16 kHz PCM-to-log-mel adapter used by ``audio_embed``."""

    frame_length = 320
    frame_step = 160
    fft_length = 512
    num_frequency_bins = 257

    def __init__(self, config: Gemma4AudioConfig):
        if config.feature_size != 128 or config.sampling_rate != 16_000:
            raise ValueError("The Gemma 4 E2B adapter requires 128 mel bins at 16 kHz")
        self.feature_size = config.feature_size
        self.dft_matrix = nn.Parameter(
            (self.frame_length, self.num_frequency_bins * 2), dtype="float32"
        )
        self.mel_filters = nn.Parameter(
            (self.num_frequency_bins, self.feature_size), dtype="float32"
        )

    def to(self, dtype: str | None = None) -> None:
        # The PCM frontend intentionally accumulates in float32, including in
        # otherwise-float16 browser artifacts.
        del dtype

    def forward(self, samples: Tensor) -> Tensor:
        def _frame(waveform: te.Tensor):
            num_samples = waveform.shape[0]
            num_frames = (
                tirx.floordiv(
                    num_samples + self.frame_length // 2 - (self.frame_length + 1),
                    self.frame_step,
                )
                + 1
            )

            def _value(frame: tirx.Var, index: tirx.Var):
                sample_index = frame * self.frame_step + index - self.frame_length // 2
                return tirx.if_then_else(
                    sample_index >= 0,
                    waveform[sample_index],
                    tirx.const(0, waveform.dtype),
                )

            return te.compute((num_frames, self.frame_length), _value, name="gemma4_audio_frames")

        frames = op.tensor_expr_op(_frame, "gemma4_audio_frames", [samples])
        spectrum = op.matmul(frames, self.dft_matrix)
        real, imaginary = op.split(spectrum, 2, axis=-1)
        magnitude = op.sqrt(real * real + imaginary * imaginary)
        mel = op.matmul(magnitude, self.mel_filters)
        mel = op.log(mel + 1.0e-3)
        return op.reshape(mel, (1, mel.shape[0], self.feature_size))


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float):
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm = Gemma4ScaleLayerNorm(out_channels, eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = op.permute_dims(hidden_states, axes=(0, 2, 3, 1))
        hidden_states = op.relu(self.norm(hidden_states))
        return op.permute_dims(hidden_states, axes=(0, 3, 1, 2))


class Gemma4AudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        channels0, channels1 = config.subsampling_conv_channels
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(1, channels0, config.rms_norm_eps)
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            channels0, channels1, config.rms_norm_eps
        )
        self.input_proj_linear = nn.Linear(
            (channels0 // 4) * channels1,
            config.hidden_size,
            bias=False,
        )

    def forward(self, input_features: Tensor) -> Tensor:
        hidden_states = op.unsqueeze(input_features, dim=1)
        hidden_states = self.layer0(hidden_states)
        hidden_states = self.layer1(hidden_states)
        hidden_states = op.permute_dims(hidden_states, axes=(0, 2, 3, 1))
        batch, seq_len, width, channels = hidden_states.shape
        hidden_states = op.reshape(hidden_states, (batch, seq_len, width * channels))
        return self.input_proj_linear(hidden_states)


class Gemma4AudioAttention(nn.Module):
    """The E2B audio tower's causal 12-token local attention."""

    def __init__(self, config: Gemma4AudioConfig):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_scale = (self.head_dim**-0.5) / math.log(2.0)
        self.k_scale = math.log1p(math.e) / math.log(2.0)
        self.logit_cap = config.attention_logit_cap
        self.invalid_logit = config.attention_invalid_logits_value
        self.window = config.attention_chunk_size

        self.q_proj = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.k_proj = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.v_proj = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.post = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.relative_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.per_dim_scale = nn.Parameter((self.head_dim,))
        self.relative_positions = nn.Parameter((13, config.hidden_size), dtype="float32")

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, self.num_heads, self.head_dim)
        query = op.reshape(self.q_proj(hidden_states), hidden_shape)
        key = op.reshape(self.k_proj(hidden_states), hidden_shape)
        value = op.reshape(self.v_proj(hidden_states), hidden_shape)

        query = op.astype(query, "float32")
        key = op.astype(key, "float32")
        value = op.astype(value, "float32")
        query = query * self.q_scale * op.softplus(op.astype(self.per_dim_scale, "float32"))
        key = key * self.k_scale

        relative = self.relative_positions
        relative = op.astype(relative, hidden_states.dtype)
        relative = self.relative_k_proj(relative)
        relative = op.reshape(relative, (13, self.num_heads, self.head_dim))
        relative = op.astype(relative, "float32")

        def _attention_scores(q: te.Tensor, k: te.Tensor, rel: te.Tensor):
            reduce_dim = te.reduce_axis((0, self.head_dim), name="audio_head_dim")

            def _value(b: tirx.Var, s: tirx.Var, h: tirx.Var, w: tirx.Var):
                key_position = s - (self.window - 1) + w
                safe_position = tirx.max(key_position, 0)
                return te.sum(
                    q[b, s, h, reduce_dim]
                    * (k[b, safe_position, h, reduce_dim] + rel[w + 1, h, reduce_dim]),
                    axis=reduce_dim,
                )

            return te.compute(
                (q.shape[0], q.shape[1], q.shape[2], self.window),
                _value,
                name="gemma4_audio_attention_scores",
            )

        scores = op.tensor_expr_op(
            _attention_scores,
            "gemma4_audio_attention_scores",
            [query, key, relative],
        )
        scores = op.tanh(scores / self.logit_cap) * self.logit_cap

        def _mask_logits(values: te.Tensor):
            return te.compute(
                values.shape,
                lambda b, s, h, w: tirx.if_then_else(
                    s - (self.window - 1) + w >= 0,
                    values[b, s, h, w],
                    tirx.const(self.invalid_logit, values.dtype),
                ),
                name="gemma4_audio_attention_logits",
            )

        logits = op.tensor_expr_op(
            _mask_logits,
            "gemma4_audio_attention_logits",
            [scores],
        )
        weights = op.softmax(logits, axis=-1)

        def _attention_output(attn: te.Tensor, values: te.Tensor):
            reduce_window = te.reduce_axis((0, self.window), name="audio_window")

            def _value(b: tirx.Var, s: tirx.Var, h: tirx.Var, d: tirx.Var):
                key_position = tirx.max(s - (self.window - 1) + reduce_window, 0)
                return te.sum(
                    attn[b, s, h, reduce_window] * values[b, key_position, h, d],
                    axis=reduce_window,
                )

            return te.compute(values.shape, _value, name="gemma4_audio_attention_output")

        output = op.tensor_expr_op(
            _attention_output,
            "gemma4_audio_attention_output",
            [weights, value],
        )
        output = op.reshape(output, (batch, seq_len, self.hidden_size))
        return self.post(op.astype(output, hidden_states.dtype))


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.ffw_layer_1 = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 4)
        self.ffw_layer_2 = Gemma4ClippableLinear(config, config.hidden_size * 4, config.hidden_size)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_layer_scale = config.residual_weight
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = _clip_for_dtype(hidden_states, self.gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.ffw_layer_2(op.silu(self.ffw_layer_1(hidden_states)))
        hidden_states = _clip_for_dtype(hidden_states, self.gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        return residual + hidden_states * self.post_layer_scale


class Gemma4AudioLightConv1d(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.linear_start = Gemma4ClippableLinear(
            config, config.hidden_size, config.hidden_size * 2
        )
        self.linear_end = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.depthwise_conv1d = nn.Conv1D(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.left_pad = config.conv_kernel_size - 1
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.linear_start(self.pre_layer_norm(hidden_states))
        gate, value = op.split(hidden_states, 2, axis=-1)
        hidden_states = gate * op.sigmoid(value)
        hidden_states = op.permute_dims(hidden_states, axes=(0, 2, 1))
        hidden_states = op.pad(hidden_states, [0, 0, 0, 0, self.left_pad, 0])
        hidden_states = self.depthwise_conv1d(hidden_states)
        hidden_states = op.permute_dims(hidden_states, axes=(0, 2, 1))
        hidden_states = _clip_for_dtype(hidden_states, self.gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.linear_end(op.silu(hidden_states))
        return hidden_states + residual


class Gemma4AudioLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config)
        self.lconv1d = Gemma4AudioLightConv1d(config)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        hidden_states = _clip_for_dtype(hidden_states, self.gradient_clipping)
        hidden_states = self.self_attn(self.norm_pre_attn(hidden_states))
        hidden_states = self.norm_post_attn(_clip_for_dtype(hidden_states, self.gradient_clipping))
        hidden_states = hidden_states + residual
        hidden_states = self.feed_forward2(self.lconv1d(hidden_states))
        return self.norm_out(_clip_for_dtype(hidden_states, self.gradient_clipping))


class Gemma4AudioModel(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.layers = nn.ModuleList(
            [Gemma4AudioLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)

    def forward(self, input_features: Tensor) -> Tensor:
        hidden_states = self.subsample_conv_projection(input_features)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.output_proj(hidden_states)


class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, audio_config: Gemma4AudioConfig, text_config: Gemma4TextConfig):
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            audio_config.output_proj_dims,
            audio_config.rms_norm_eps,
            with_scale=False,
        )
        self.embedding_projection = nn.Linear(
            audio_config.output_proj_dims,
            text_config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(hidden_states))


def _clip_for_dtype(hidden_states: Tensor, configured_limit: float) -> Tensor:
    dtype_limit = 65_504.0 if hidden_states.dtype == "float16" else 3.3895313892515355e38
    limit = min(configured_limit, dtype_limit)
    lower = Tensor.from_scalar(-limit, hidden_states.dtype)
    upper = Tensor.from_scalar(limit, hidden_states.dtype)
    return op.minimum(op.maximum(hidden_states, lower), upper)


def _dft_matrix(frame_length: int, fft_length: int) -> np.ndarray:
    indices = np.arange(frame_length, dtype="float64")
    frequencies = np.arange(fft_length // 2 + 1, dtype="float64")
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * indices / frame_length)
    angles = 2.0 * np.pi * np.outer(indices, frequencies) / fft_length
    real = window[:, None] * np.cos(angles)
    imaginary = -window[:, None] * np.sin(angles)
    return np.concatenate([real, imaginary], axis=1).astype("float32")


def _htk_mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    sampling_rate: int,
) -> np.ndarray:
    mel_min = 0.0
    mel_max = 2595.0 * np.log10(1.0 + (sampling_rate / 2) / 700.0)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = 700.0 * (np.power(10.0, mel_freqs / 2595.0) - 1.0)
    fft_freqs = np.linspace(0.0, sampling_rate // 2, num_frequency_bins)
    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(0.0, np.minimum(down_slopes, up_slopes)).astype("float32")


def _audio_relative_positions(config: Gemma4AudioConfig) -> np.ndarray:
    context_size = (
        config.attention_chunk_size
        + config.attention_context_left
        - 1
        + config.attention_context_right
    )
    positions = np.arange(context_size // 2, -1, -1, dtype="float32")[:, None]
    num_timescales = config.hidden_size // 2
    increment = math.log(10_000.0) / max(num_timescales - 1, 1)
    inv_timescales = np.exp(np.arange(num_timescales, dtype="float32") * -increment)[None, :]
    scaled = positions * inv_timescales
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=-1).astype("float32")


def gemma4_audio_generated_parameters(config: Gemma4AudioConfig) -> dict[str, np.ndarray]:
    """Return deterministic adapter parameters that are absent from the HF checkpoint."""

    parameters = {
        "audio_preprocessor.dft_matrix": _dft_matrix(
            Gemma4AudioFeatureExtractor.frame_length,
            Gemma4AudioFeatureExtractor.fft_length,
        ),
        "audio_preprocessor.mel_filters": _htk_mel_filter_bank(
            Gemma4AudioFeatureExtractor.num_frequency_bins,
            config.feature_size,
            config.sampling_rate,
        ),
    }
    relative_positions = _audio_relative_positions(config)
    for layer_idx in range(config.num_hidden_layers):
        parameters[f"audio_tower.layers.{layer_idx}.self_attn.relative_positions"] = (
            relative_positions
        )
    return parameters


__all__ = [
    "Gemma4AudioFeatureExtractor",
    "Gemma4AudioModel",
    "Gemma4MultimodalEmbedder",
    "Gemma4RMSNorm",
    "gemma4_audio_generated_parameters",
]
