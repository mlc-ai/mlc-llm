# pylint: disable=invalid-name,missing-docstring
"""Unit tests for Gemma3V vision-language model architecture."""

from mlc_llm.model import MODELS
from mlc_llm.protocol.artifact_manifest import (
    ImageDecodeProcessor,
    build_compiled_program_artifact,
    build_model_package_manifest,
)
from mlc_llm.quantization import QUANTIZATION

# Minimal config dict with small dimensions for fast testing.
# Mirrors the structure of a real HuggingFace gemma-3-4b-it config.json.
SMALL_GEMMA3V_CONFIG = {
    "text_config": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "context_window_size": 1024,
        "prefill_chunk_size": 512,
        "sliding_window_size": 256,
        "sliding_window_pattern": 6,
        "hidden_activation": "gelu_pytorch_tanh",
    },
    "vision_config": {
        "hidden_size": 64,
        # image_size / patch_size = 4 (grid), 4x4 avg_pool -> 1x1 -> 1 token
        "image_size": 56,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "patch_size": 14,
        "num_channels": 3,
        "layer_norm_eps": 1e-6,
    },
    "vocab_size": 262208,
    "mm_tokens_per_image": 1,
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
}


def test_gemma3v_model_registered():
    """Verify Gemma3V model is in the registry."""
    assert "gemma3_v" in MODELS, "gemma3_v should be registered in MODELS"


def test_gemma3v_creation():
    """Test Gemma3V model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from dict
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters include vision_tower, language_model, and projector components
    """
    model_info = MODELS["gemma3_v"]
    config = model_info.config.from_dict(SMALL_GEMMA3V_CONFIG)
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    # Verify export succeeded
    assert mod is not None
    assert len(named_params) > 0

    # Verify VLM composition: params from all three components
    param_names = [name for name, _ in named_params]
    has_vision = any(n.startswith("vision_tower.") for n in param_names)
    has_language = any(n.startswith("language_model.") for n in param_names)
    has_projector = any(n.startswith("multi_modal_projector.") for n in param_names)
    assert has_vision, "Should have vision_tower parameters"
    assert has_language, "Should have language_model parameters"
    assert has_projector, "Should have multi_modal_projector parameters"

    # Verify all expected functions are exported
    expected_funcs = [
        "embed",
        "image_embed",
        "prefill",
        "decode",
        "batch_prefill",
        "batch_decode",
        "batch_verify",
        "create_paged_kv_cache",
    ]
    for func_name in expected_funcs:
        assert func_name in mod, f"Module should contain '{func_name}' function"

    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


def test_gemma3v_artifact_declares_image_input():
    entry = MODELS["gemma3_v"]
    config = entry.config.from_dict(SMALL_GEMMA3V_CONFIG)
    image = entry.artifact.tasks(config)["chat.completions"]["inputs"]["image"]
    processor = ImageDecodeProcessor.model_validate(image["processor"])
    assert (processor.resize.mode, processor.resize.height, processor.resize.width) == (
        "stretch",
        56,
        56,
    )
    assert processor.num_embeddings == config.mm_tokens_per_image
    assert image["prompt"] == {
        "prefix_token_ids": [255999],
        "placeholder_token_id": 262144,
        "suffix_token_ids": [256000],
    }


def test_gemma3v_artifact_points_at_exported_functions():
    entry = MODELS["gemma3_v"]
    config = entry.config.from_dict(SMALL_GEMMA3V_CONFIG)
    quantization = QUANTIZATION["q4f16_1"]
    model, _ = entry.quantize[quantization.kind](config, quantization)
    mod, named_parameters, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)

    tasks = entry.artifact.tasks(config)
    programs = entry.artifact.programs(config)
    artifact = build_compiled_program_artifact(
        tasks, programs, named_parameters, symbolic_sizes={"vocab_size": config.vocab_size}
    )
    package = build_model_package_manifest(tasks, named_parameters)
    assert artifact.interface_id == package.interface_id
    assert artifact.parameter_schema_id == package.weights.parameter_schema_id
    assert artifact.resources.estimated_device_memory_bytes > 0

    program = programs["generation"]
    assert program["exports"] == {
        "embed_tokens": "embed",
        "prefill_embeds": "prefill",
        "decode_embeds": "decode",
        "create_kv_cache": "create_tir_paged_kv_cache",
    }
    assert program["adapters"] == {"image": "embed_image"}
    exported_functions = {global_var.name_hint for global_var in mod.get_global_vars()}
    assert set(program["exports"].values()) - {"create_tir_paged_kv_cache"} <= exported_functions
    assert set(program["adapters"].values()) <= exported_functions


def test_gemma3v_takes_uint32_pixels_on_webgpu():
    import tvm

    entry = MODELS["gemma3_v"]
    config = entry.config.from_dict(SMALL_GEMMA3V_CONFIG)
    with tvm.target.Target("webgpu"):
        model = entry.model(config)
        programs = entry.artifact.programs(config)
    assert model.image_dtype == "uint32"
    assert programs["generation"]["adapter_dtypes"] == {"image": "uint32"}
    assert entry.model(config).image_dtype == "uint8"
    assert "adapter_dtypes" not in entry.artifact.programs(config)["generation"]


if __name__ == "__main__":
    test_gemma3v_model_registered()
    test_gemma3v_creation()
    test_gemma3v_artifact_declares_image_input()
    test_gemma3v_artifact_points_at_exported_functions()
    test_gemma3v_takes_uint32_pixels_on_webgpu()
