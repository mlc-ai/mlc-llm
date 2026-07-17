import pytest

from mlc_llm.model.qwen2 import qwen2_loader
from mlc_llm.model.qwen2.qwen2_model import QWen2Config, QWen2LMHeadModel
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support.runtime_lora import RuntimeLoRAConfig

pytestmark = [pytest.mark.unittest]


def _config() -> QWen2Config:
    return QWen2Config(
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        vocab_size=32,
        context_window_size=64,
        prefill_chunk_size=16,
        runtime_lora=RuntimeLoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=("down_proj", "gate_proj", "q_proj", "v_proj"),
        ),
    )


def test_qwen2_exports_separate_runtime_lora_parameters():
    config = _config()
    model = QWen2LMHeadModel(config)
    model.to("float16")
    _, exported_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    runtime_params = {
        name: param for name, param in exported_params if param.attrs.get("runtime_lora_param")
    }

    assert len(runtime_params) == 8
    assert (
        runtime_params["model.layers.0.self_attn.q_proj_lora.lora_a.weight"].attrs[
            "peft_source_suffix"
        ]
        == "model.layers.0.self_attn.q_proj.lora_A.weight"
    )
    assert (
        runtime_params["model.layers.0.mlp.down_proj_lora.lora_b.weight"].attrs[
            "peft_source_suffix"
        ]
        == "model.layers.0.mlp.down_proj.lora_B.weight"
    )
    assert (
        runtime_params["model.layers.0.mlp.down_proj_lora.lora_b.weight"].attrs[
            "runtime_lora_scale"
        ]
        == 2.0
    )


def test_qwen2_base_loader_does_not_claim_adapter_parameters():
    mapping = qwen2_loader.huggingface(_config(), QUANTIZATION["q0f16"])
    assert not any("_lora." in name for name in mapping.param_map)
    assert "model.layers.0.self_attn.c_attn.weight" in mapping.param_map
