"""Correctness tests for the Gemma 4 E2B text+audio implementation."""

import math

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.frontend import nn

from mlc_llm.model import MODELS
from mlc_llm.model.gemma4.gemma4_audio import (
    Gemma4AudioAttention,
    Gemma4AudioFeatureExtractor,
    _audio_relative_positions,
    gemma4_audio_generated_parameters,
)
from mlc_llm.model.gemma4.gemma4_config import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
)
from mlc_llm.model.gemma4.gemma4_model import (
    Gemma4TextRotaryEmbedding,
    _replace_modality_embeddings,
)
from mlc_llm.protocol.artifact_manifest import (
    AudioDecodeProcessor,
    build_compiled_program_artifact,
)
from mlc_llm.quantization import QUANTIZATION


def _build_and_run(module: nn.Module, spec, *inputs):
    mod, named_parameters, _ = module.export_tvm(spec=spec, allow_extern=True)
    executable = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(executable, tvm.cpu())
    return vm, named_parameters


def _reference_log_mel(samples: np.ndarray) -> np.ndarray:
    frame_length = 320
    frame_step = 160
    fft_length = 512
    padded = np.pad(samples[None, :], ((0, 0), (frame_length // 2, 0)))
    num_frames = (padded.shape[1] - (frame_length + 1)) // frame_step + 1
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(1, num_frames, frame_length + 1),
        strides=(padded.strides[0], frame_step * padded.strides[1], padded.strides[1]),
    )[..., :-1]
    window = np.hanning(frame_length + 1)[:-1].astype("float32")
    magnitude = np.abs(np.fft.rfft(frames * window, n=fft_length, axis=-1))

    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
    mel_freqs = np.linspace(mel_min, mel_max, 130)
    filter_freqs = 700.0 * (np.power(10.0, mel_freqs / 2595.0) - 1.0)
    fft_freqs = np.linspace(0.0, 8000.0, 257)
    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    filters = np.maximum(
        0.0,
        np.minimum(-slopes[:, :-2] / filter_diff[:-1], slopes[:, 2:] / filter_diff[1:]),
    )
    features = np.log(np.matmul(magnitude, filters) + np.float64(1.0e-3))
    return features.astype("float32")


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    values = values - np.max(values, axis=axis, keepdims=True)
    values = np.exp(values)
    return values / np.sum(values, axis=axis, keepdims=True)


def _reference_block_audio_attention(
    hidden_states: np.ndarray,
    parameters: dict[str, np.ndarray],
    config: Gemma4AudioConfig,
) -> np.ndarray:
    batch, seq_len, hidden_size = hidden_states.shape
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    chunk_size = config.attention_chunk_size
    past = config.attention_context_left - 1
    context_size = chunk_size + past + config.attention_context_right

    def linear(name: str, values: np.ndarray) -> np.ndarray:
        return values @ parameters[name].T

    query = linear("a.q_proj.linear.weight", hidden_states).reshape(
        batch, seq_len, num_heads, head_dim
    )
    key = linear("a.k_proj.linear.weight", hidden_states).reshape(
        batch, seq_len, num_heads, head_dim
    )
    value = linear("a.v_proj.linear.weight", hidden_states).reshape(
        batch, seq_len, num_heads, head_dim
    )
    query *= (head_dim**-0.5) / math.log(2.0)
    query *= np.logaddexp(0.0, parameters["a.per_dim_scale"])
    key *= math.log1p(math.e) / math.log(2.0)

    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    padded_len = num_blocks * chunk_size
    query = np.pad(query, ((0, 0), (0, padded_len - seq_len), (0, 0), (0, 0)))
    query = query.reshape(batch, num_blocks, chunk_size, num_heads, head_dim)

    context_pad = ((0, 0), (past, config.attention_context_right + chunk_size - 1), (0, 0), (0, 0))
    padded_key = np.pad(key, context_pad)
    padded_value = np.pad(value, context_pad)
    key_blocks = np.stack(
        [
            padded_key[:, block * chunk_size : block * chunk_size + context_size]
            for block in range(num_blocks)
        ],
        axis=1,
    )
    value_blocks = np.stack(
        [
            padded_value[:, block * chunk_size : block * chunk_size + context_size]
            for block in range(num_blocks)
        ],
        axis=1,
    )

    relative = parameters["a.relative_positions"]
    relative = linear("a.relative_k_proj.weight", relative).reshape(13, num_heads, head_dim)
    queries = query.transpose(0, 3, 1, 2, 4)
    matrix_ac = np.einsum("bhnqd,bnkhd->bhnqk", queries, key_blocks)
    matrix_bd = np.einsum("bhnqd,rhd->bhnqr", queries, relative)
    matrix_bd = np.pad(matrix_bd, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 12)))
    matrix_bd = matrix_bd.reshape(batch, num_heads, num_blocks, chunk_size * 25)
    matrix_bd = matrix_bd[..., : chunk_size * context_size]
    matrix_bd = matrix_bd.reshape(batch, num_heads, num_blocks, chunk_size, context_size)

    scores = np.tanh((matrix_ac + matrix_bd) / config.attention_logit_cap)
    scores *= config.attention_logit_cap
    for block in range(num_blocks):
        for query_offset in range(chunk_size):
            query_position = block * chunk_size + query_offset
            for key_offset in range(context_size):
                key_position = block * chunk_size - past + key_offset
                distance = query_position - key_position
                if not (0 <= key_position < seq_len and 0 <= distance < past):
                    scores[:, :, block, query_offset, key_offset] = (
                        config.attention_invalid_logits_value
                    )

    weights = _softmax(scores.astype("float32"), axis=-1)
    output = np.einsum("bhnqk,bnkhd->bnqhd", weights, value_blocks)
    output = output.reshape(batch, padded_len, hidden_size)[:, :seq_len]
    return linear("a.post.linear.weight", output)


def _reference_rope(values: np.ndarray, positions: np.ndarray, theta: float, active: int):
    half_dim = values.shape[-1] // 2
    frequencies = np.arange(half_dim, dtype="float32")
    inverse = np.where(
        frequencies < active,
        1.0 / np.power(theta, 2.0 * frequencies / values.shape[-1]),
        0.0,
    )
    angles = positions[:, :, None, None] * inverse[None, None, None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    rotated = np.concatenate([-values[..., half_dim:], values[..., :half_dim]], axis=-1)
    return values * cos + rotated * sin


def test_gemma4_registration_config_and_artifact():
    entry = MODELS["gemma4"]
    config = Gemma4Config.from_dict({})
    assert entry.supports_flashinfer is False
    assert config.vocab_size == 262_144
    assert config.text_config.num_hidden_layers == 35
    assert config.text_config.first_kv_shared_layer == 15
    assert config.prefill_chunk_size == config.text_config.sliding_window == 512
    assert config.sliding_window_size == -1

    tasks = entry.artifact.tasks(config)
    audio = tasks["chat.completions"]["inputs"]["audio"]
    processor = AudioDecodeProcessor.model_validate(audio["processor"])
    assert (processor.sample_rate_hz, processor.channels) == (16_000, 1)
    assert processor.max_samples == 480_000
    assert audio["prompt"]["placeholder_token_id"] == 258_881


def test_audio_feature_extractor_matches_reference():
    class FeatureModule(nn.Module):
        def __init__(self):
            self.extractor = Gemma4AudioFeatureExtractor(Gemma4AudioConfig())

        def forward(self, samples):
            return self.extractor(samples)

    samples = np.random.default_rng(0).standard_normal(1601).astype("float32")
    vm, named_parameters = _build_and_run(
        FeatureModule(),
        {"forward": {"samples": nn.spec.Tensor(samples.shape, "float32")}},
        samples,
    )
    generated = gemma4_audio_generated_parameters(Gemma4AudioConfig())
    parameter_values = {
        "extractor.dft_matrix": generated["audio_preprocessor.dft_matrix"],
        "extractor.mel_filters": generated["audio_preprocessor.mel_filters"],
    }
    actual = vm["forward"](
        tvm.runtime.tensor(samples),
        *[tvm.runtime.tensor(parameter_values[name]) for name, _ in named_parameters],
    ).numpy()
    expected = _reference_log_mel(samples)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "layer_idx,head_dim,active,theta", [(0, 256, 128, 10_000.0), (4, 512, 64, 1_000_000.0)]
)
def test_text_rope_matches_reference(layer_idx, head_dim, active, theta):
    class RopeModule(nn.Module):
        def __init__(self):
            self.rope = Gemma4TextRotaryEmbedding(Gemma4TextConfig(), layer_idx)

        def forward(self, values, positions):
            return self.rope.apply_query(values, positions)

    rng = np.random.default_rng(layer_idx)
    values = rng.standard_normal((1, 4, 2, head_dim)).astype("float32")
    positions = np.array([0, 1, 17, 1024], dtype="int32")
    vm, _ = _build_and_run(
        RopeModule(),
        {
            "forward": {
                "values": nn.spec.Tensor(values.shape, "float32"),
                "positions": nn.spec.Tensor(positions.shape, "int32"),
            }
        },
        values,
        positions,
    )
    actual = vm["forward"](tvm.runtime.tensor(values), tvm.runtime.tensor(positions)).numpy()
    expected = _reference_rope(values, positions[None, :], theta, active)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_audio_attention_matches_block_reference():
    config = Gemma4AudioConfig(
        hidden_size=8,
        num_attention_heads=2,
        use_clipped_linears=False,
    )

    class AttentionModule(nn.Module):
        def __init__(self):
            self.a = Gemma4AudioAttention(config)

        def forward(self, hidden_states):
            return self.a(hidden_states)

    rng = np.random.default_rng(1)
    hidden_states = rng.normal(0.0, 0.2, (1, 25, 8)).astype("float32")
    vm, named_parameters = _build_and_run(
        AttentionModule(),
        {"forward": {"hidden_states": nn.spec.Tensor(hidden_states.shape, "float32")}},
        hidden_states,
    )
    parameter_values = {}
    for name, parameter in named_parameters:
        if name == "a.relative_positions":
            parameter_values[name] = _audio_relative_positions(config)
        else:
            parameter_values[name] = rng.normal(
                0.0, 0.2, tuple(int(dim) for dim in parameter.shape)
            ).astype("float32")
    actual = vm["forward"](
        tvm.runtime.tensor(hidden_states),
        *[tvm.runtime.tensor(parameter_values[name]) for name, _ in named_parameters],
    ).numpy()
    expected = _reference_block_audio_attention(hidden_states, parameter_values, config)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_audio_positions_use_pad_embedding_for_per_layer_projection():
    class ReplaceModule(nn.Module):
        def forward(self, input_embeds, modality_ids, pad_embedding):
            return _replace_modality_embeddings(
                input_embeds,
                modality_ids,
                pad_embedding,
            )

    input_embeds = np.arange(12, dtype="float32").reshape(1, 3, 4)
    modality_ids = np.array([[0, 1, 0]], dtype="int32")
    pad_embedding = np.array([[1.0, -1.0, 2.0, -2.0]], dtype="float32")
    vm, named_parameters = _build_and_run(
        ReplaceModule(),
        {
            "forward": {
                "input_embeds": nn.spec.Tensor(input_embeds.shape, "float32"),
                "modality_ids": nn.spec.Tensor(modality_ids.shape, "int32"),
                "pad_embedding": nn.spec.Tensor(pad_embedding.shape, "float32"),
            }
        },
        input_embeds,
        modality_ids,
        pad_embedding,
    )
    assert not named_parameters
    actual = vm["forward"](
        tvm.runtime.tensor(input_embeds),
        tvm.runtime.tensor(modality_ids),
        tvm.runtime.tensor(pad_embedding),
    ).numpy()
    expected = input_embeds.copy()
    expected[:, 1, :] = pad_embedding[0]
    np.testing.assert_array_equal(actual, expected)


def test_loader_covers_unquantized_and_q4_parameter_schemas():
    entry = MODELS["gemma4"]
    config = Gemma4Config.from_dict({"vision_config": {"num_hidden_layers": 16}})
    mapping = entry.source["huggingface-safetensor"](config, QUANTIZATION["q4f16_1"])
    model = entry.model(config)
    _, unquantized_parameters, _ = model.export_tvm(
        spec=model.get_default_spec(), allow_extern=True
    )
    named_parameters = dict(unquantized_parameters)
    ple_names = [
        name
        for name in named_parameters
        if name.startswith("language_model.embed_tokens_per_layer.")
    ]
    assert len(ple_names) == config.text_config.num_hidden_layers
    assert set(mapping.param_map) == {name for name, _ in unquantized_parameters}
    generated_names = set(gemma4_audio_generated_parameters(config.audio_config))
    assert all(mapping.param_map[name] == [] for name in generated_names)
    assert all(
        mapping.map_func[name]().shape == tuple(int(dim) for dim in named_parameters[name].shape)
        for name in generated_names
    )
    assert not any("vision" in name for name in mapping.param_map)
    assert any("vision_tower" in name for name in mapping.unused_params)
    assert all(
        f"model.language_model.layers.{layer_idx}.self_attn.v_norm.weight" in mapping.unused_params
        for layer_idx in range(
            config.text_config.first_kv_shared_layer,
            config.text_config.num_hidden_layers,
        )
    )

    packed_name = "model.language_model.embed_tokens_per_layer.weight"
    packed = np.arange(2 * 35 * 256, dtype="float32").reshape(2, 35 * 256)
    for layer_idx in (0, 17, 34):
        name = f"language_model.embed_tokens_per_layer.{layer_idx}.weight"
        assert mapping.param_map[name] == [packed_name]
        np.testing.assert_array_equal(
            mapping.map_func[name](packed),
            packed[:, layer_idx * 256 : (layer_idx + 1) * 256].astype("float16"),
        )

    quantized_model, quantize_mapping = entry.quantize["group-quant"](
        config, QUANTIZATION["q4f16_1"]
    )
    mod, quantized_parameters, _ = quantized_model.export_tvm(
        spec=quantized_model.get_default_spec(), allow_extern=True
    )
    quantized_names = {name for name, _ in quantized_parameters}
    for name, _ in unquantized_parameters:
        expected_names = quantize_mapping.param_map.get(name, [name])
        assert set(expected_names).issubset(quantized_names)

    artifact = build_compiled_program_artifact(
        entry.artifact.tasks(config),
        entry.artifact.programs(config),
        quantized_parameters,
        entry.artifact.required_features,
    )
    assert artifact.resources.max_storage_buffer_binding_size <= 256 * 1024 * 1024
    exported_functions = {global_var.name_hint for global_var in mod.get_global_vars()}
    assert {"audio_embed", "prefill_prompt", "decode_tokens"}.issubset(exported_functions)
