"""Tests for the LLaVA artifact contract."""

from mlc_llm.model import MODELS
from mlc_llm.model.llava.llava_model import LlavaConfig
from mlc_llm.protocol.artifact_manifest import (
    ImageDecodeProcessor,
    build_compiled_program_artifact,
    build_model_package_manifest,
)
from mlc_llm.quantization import QUANTIZATION


def _config():
    return LlavaConfig.from_dict(
        {
            "image_token_index": 120,
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "max_position_embeddings": 1024,
                "vocab_size": 128,
            },
            "vision_config": {
                "hidden_size": 32,
                "image_size": 336,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "patch_size": 14,
                "projection_dim": 32,
                "vocab_size": 128,
            },
            "vocab_size": 128,
        }
    )


def test_llava_artifact_declares_image_input():
    config = _config()
    entry = MODELS["llava"]
    image = entry.artifact.tasks(config)["chat.completions"]["inputs"]["image"]
    processor = ImageDecodeProcessor.model_validate(image["processor"])
    assert (processor.resize.mode, processor.resize.height, processor.resize.width) == (
        "center_crop",
        336,
        336,
    )
    assert processor.num_embeddings == 576
    assert image["prompt"] == {"placeholder_token_id": 120}


def test_llava_artifact_points_at_exported_functions():
    config = _config()
    entry = MODELS["llava"]
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
    exported_functions = {global_var.name_hint for global_var in mod.get_global_vars()}
    assert set(program["exports"].values()) - {"create_tir_paged_kv_cache"} <= exported_functions
    assert set(program["adapters"].values()) <= exported_functions

    # The vision tower and projector stay in the model dtype.
    names = [name for name, _ in named_parameters]
    quantized = [name for name in names if name.endswith(".q_weight")]
    assert quantized and all(name.startswith("language_model.") for name in quantized)
    assert "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight" in names
    assert "multi_modal_projector.linear_1.weight" in names


def test_llava_takes_uint32_pixels_on_webgpu():
    import tvm

    config = _config()
    entry = MODELS["llava"]
    with tvm.target.Target("webgpu"):
        model = entry.model(config)
        programs = entry.artifact.programs(config)
    assert model.image_dtype == "uint32"
    assert programs["generation"]["adapter_dtypes"] == {"image": "uint32"}
    assert entry.model(config).image_dtype == "uint8"
    assert "adapter_dtypes" not in entry.artifact.programs(config)["generation"]
