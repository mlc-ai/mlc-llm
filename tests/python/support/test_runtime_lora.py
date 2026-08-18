import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlc_llm.loader.runtime_lora import (
    make_runtime_lora_mapping,
    resolve_runtime_lora_weight,
)
from mlc_llm.support.runtime_lora import RuntimeLoRAConfig, validate_runtime_lora_scope

pytestmark = [pytest.mark.unittest]


def _write_adapter_config(adapter_dir: Path, **overrides) -> None:
    config = {
        "bias": "none",
        "fan_in_fan_out": False,
        "lora_alpha": 16,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config.update(overrides)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_safetensor_header(path: Path, names) -> None:
    header = {name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for name in names}
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes)


def test_runtime_lora_config_standard_and_rslora_scaling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_dir = Path(tmp_dir)
        _write_adapter_config(adapter_dir)
        config = RuntimeLoRAConfig.from_peft_directory(adapter_dir)
        assert config.rank == 8
        assert config.target_modules == ("q_proj", "v_proj")
        assert config.scaling == 2.0

        _write_adapter_config(adapter_dir, use_rslora=True)
        config = RuntimeLoRAConfig.from_peft_directory(adapter_dir)
        assert config.scaling == pytest.approx(16 / math.sqrt(8))
        assert RuntimeLoRAConfig.from_dict(config.asdict()) == config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_modules": ["lm_head"]}, "unsupported"),
        ({"rank_pattern": {"q_proj": 4}}, "rank_pattern"),
        ({"alora_invocation_tokens": [1, 2]}, "alora_invocation_tokens"),
        ({"arrow_config": {"top_k": 1}}, "arrow_config"),
        ({"exclude_modules": ["k_proj"]}, "exclude_modules"),
        ({"lora_bias": True}, "lora_bias"),
        ({"use_qalora": True}, "use_qalora"),
        ({"use_dora": True}, "DoRA"),
        ({"bias": "all"}, "bias"),
    ],
)
def test_runtime_lora_config_rejects_unsupported_peft_features(overrides, message):
    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_dir = Path(tmp_dir)
        _write_adapter_config(adapter_dir, **overrides)
        with pytest.raises(ValueError, match=message):
            RuntimeLoRAConfig.from_peft_directory(adapter_dir)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora_alpha": float("inf")}, "finite"),
        ({"target_modules": ["q_proj", 1]}, "string"),
        ({"use_rslora": "false"}, "use_rslora"),
    ],
)
def test_runtime_lora_config_rejects_malformed_values(overrides, message):
    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_dir = Path(tmp_dir)
        _write_adapter_config(adapter_dir, **overrides)
        with pytest.raises(ValueError, match=message):
            RuntimeLoRAConfig.from_peft_directory(adapter_dir)


def test_runtime_lora_scope_is_deliberately_narrow():
    validate_runtime_lora_scope(
        model_name="qwen2", quantization_name="q0f16", tensor_parallel_shards=1
    )
    with pytest.raises(ValueError, match="Qwen2"):
        validate_runtime_lora_scope(
            model_name="llama", quantization_name="q0f16", tensor_parallel_shards=1
        )
    with pytest.raises(ValueError, match="q0f16"):
        validate_runtime_lora_scope(
            model_name="qwen2", quantization_name="q4f16_1", tensor_parallel_shards=1
        )
    with pytest.raises(ValueError, match="tensor_parallel_shards"):
        validate_runtime_lora_scope(
            model_name="qwen2", quantization_name="q0f16", tensor_parallel_shards=2
        )


def test_runtime_lora_mapping_matches_peft_prefix_and_adapter_name():
    class _FakeParam:
        dtype = "float16"

        def __init__(self, suffix, scale=1.0):
            self.attrs = {"peft_source_suffix": suffix}
            if scale != 1.0:
                self.attrs["runtime_lora_scale"] = scale

    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_dir = Path(tmp_dir)
        adapter_weight = adapter_dir / "adapter_model.safetensors"
        a_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        b_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        _write_safetensor_header(adapter_weight, [a_name, b_name])

        named_parameters = {
            "model.layers.0.self_attn.q_proj_lora.lora_a.weight": _FakeParam(
                "model.layers.0.self_attn.q_proj.lora_A.weight"
            ),
            "model.layers.0.self_attn.q_proj_lora.lora_b.weight": _FakeParam(
                "model.layers.0.self_attn.q_proj.lora_B.weight", scale=0.1
            ),
        }
        mapping = make_runtime_lora_mapping(named_parameters, adapter_weight)
        assert resolve_runtime_lora_weight(adapter_dir) == adapter_weight
        assert mapping.param_map["model.layers.0.self_attn.q_proj_lora.lora_a.weight"] == [a_name]
        assert mapping.param_map["model.layers.0.self_attn.q_proj_lora.lora_b.weight"] == [b_name]
        weight = np.array([-0.13210486, -0.53731406, 1.9878846, -0.29650125]).reshape(2, 2)
        mapped_a = mapping.map_func["model.layers.0.self_attn.q_proj_lora.lora_a.weight"](weight)
        mapped_b = mapping.map_func["model.layers.0.self_attn.q_proj_lora.lora_b.weight"](weight)
        np.testing.assert_array_equal(mapped_a, weight.astype("float16"))
        np.testing.assert_array_equal(mapped_b, (weight * 0.1).astype("float16"))
        assert not np.array_equal(mapped_b, weight.astype("float16") * np.float16(0.1))


def test_runtime_lora_math_matches_merged_weight():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 8)).astype("float32")
    weight = rng.normal(size=(6, 8)).astype("float32")
    lora_a = rng.normal(size=(4, 8)).astype("float32")
    lora_b = rng.normal(size=(6, 4)).astype("float32")
    scale = 3.0

    scaled_lora_b = lora_b * scale
    runtime_output = x @ weight.T + (x @ lora_a.T) @ scaled_lora_b.T
    merged_output = x @ (weight + scale * (lora_b @ lora_a)).T
    np.testing.assert_allclose(runtime_output, merged_output, rtol=1e-5, atol=1e-5)
