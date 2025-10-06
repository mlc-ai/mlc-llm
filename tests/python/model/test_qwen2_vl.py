# pylint: disable=invalid-name,missing-docstring
import pytest
import numpy as np

from tvm import relax
from tvm.relax.frontend.nn import Tensor

from mlc_llm.model import MODEL_PRESETS, MODELS
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.quantization.group_quantization import GroupQuantizeLinear

from mlc_llm.model.qwen2_vl.qwen2_vl_config import QWen2VLConfig, QWen2VLVisionConfig
from mlc_llm.model.qwen2_vl.qwen2_vl_image import (
    QWen2VLVisionTransformer,
    QWen2VLImagePreprocessor,
    QWen2VLVisionAttention,
)
from mlc_llm.nn import RopeMode, precompute_rope_cache

def test_vision_transformer():
    """Test the vision transformer components independently."""
    # Create a basic vision config
    vision_config = QWen2VLVisionConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=8,
        patch_size=14,
        merge_size=2,
        image_size=448,
    )
    
    # Test image preprocessor
    image_processor = QWen2VLImagePreprocessor()
    test_image = np.random.randint(0, 255, (1, 224, 224, 3), dtype="uint8")
    processed_image = image_processor(Tensor(test_image))
    assert processed_image.shape[0] == 1  # batch size
    assert processed_image.shape[1] == 3  # channels
    assert processed_image.shape[2] % (vision_config.patch_size * vision_config.merge_size) == 0
    assert processed_image.shape[3] % (vision_config.patch_size * vision_config.merge_size) == 0

    # Test vision transformer
    vision_model = QWen2VLVisionTransformer(vision_config)
    vision_output = vision_model(processed_image)
    
    # Check output shape (should be B, N, 4*hidden_size due to patch merging)
    expected_seq_len = (processed_image.shape[2] // vision_config.patch_size // vision_config.merge_size) * \
                      (processed_image.shape[3] // vision_config.patch_size // vision_config.merge_size)
    assert vision_output.shape == (1, expected_seq_len, vision_config.hidden_size * 4)

def test_m_rope_implementation():
    """Test the M-ROPE implementation in the vision transformer."""
    # Create a basic vision config with specific M-ROPE parameters
    vision_config = QWen2VLVisionConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=8,
        patch_size=14,
        merge_size=2,
        image_size=448,
        num_rope_scales=4,  # 4 different frequency scales
        rope_theta=10000,
        max_patches=256,  # Maximum sequence length
    )
    
    # Create attention module with M-ROPE
    hidden_size = vision_config.hidden_size * 4  # After patch merging
    attention = QWen2VLVisionAttention(vision_config, hidden_size)
    
    # Verify rope cache creation for each scale
    assert len(attention.rope_cache) == vision_config.num_rope_scales
    
    # Check that each scale has a different frequency
    for scale_idx in range(vision_config.num_rope_scales):
        scale_key = f"scale_{scale_idx}"
        assert scale_key in attention.rope_cache
        
        # Verify the scale factor is applied correctly (1.0, 0.5, 0.25, 0.125)
        expected_scale = 1.0 / (2 ** scale_idx)
        
        # Create a reference rope cache with the expected scale
        reference_cache = precompute_rope_cache(
            dim=attention.head_dim,
            num_heads=attention.num_attention_heads,
            max_seq_len=vision_config.max_patches,
            rope_mode=RopeMode.NORMAL,
            rope_scale=expected_scale,
            rope_theta=vision_config.rope_theta,
        )
        
        # Compare the first few values to verify scaling
        # The rope cache contains cos and sin values that should differ by scale
        assert np.allclose(
            attention.rope_cache[scale_key]["cos"][0, 0].numpy(),
            reference_cache["cos"][0, 0].numpy(),
            rtol=1e-5
        )
    
    # Test forward pass with M-ROPE
    batch_size = 2
    seq_len = 64
    test_input = np.random.randn(batch_size, seq_len, hidden_size).astype("float32")
    output = attention(Tensor(test_input))
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_size)

@pytest.mark.parametrize("model_name", ["qwen2_vl"])
def test_qwen2_vl_creation(model_name: str):
    """Test the creation of the full Qwen2 VL model."""
    model_info = MODELS["qwen2_vl"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)

@pytest.mark.parametrize("model_name", ["qwen2_vl"])
@pytest.mark.parametrize(
    "quant_name",
    ["q3f16_1", "q4f16_1", "q4f32_1"],
)
def test_qwen2_vl_group_quantization(model_name: str, quant_name: str):
    """Test group quantization of Qwen2 VL."""
    model_info = MODELS["qwen2_vl"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model, quant_map = model_info.quantize["group-quant"](config, QUANTIZATION[quant_name])

    # Check vision model quantization
    assert "vision_model.embeddings.patch_embed.weight" in quant_map.param_map
    assert isinstance(
        model.vision_model.embeddings.patch_embed,  # type: ignore[attr-defined]
        GroupQuantizeLinear,
    )

    # Check vision projection quantization
    assert "vision_projection.linear_1.weight" in quant_map.param_map
    assert isinstance(
        model.vision_projection.linear_1,  # type: ignore[attr-defined]
        GroupQuantizeLinear,
    )

    # Check vision transformer layers
    for i in range(config.vision_config.num_hidden_layers):
        # Check attention weights
        assert f"vision_model.layers.{i}.attention.q_proj.weight" in quant_map.param_map
        assert isinstance(
            model.vision_model.layers[i].attention.q_proj,  # type: ignore[attr-defined]
            GroupQuantizeLinear,
        )
        
        # Check MLP weights
        assert f"vision_model.layers.{i}.mlp.fc1.weight" in quant_map.param_map
        assert isinstance(
            model.vision_model.layers[i].mlp.fc1,  # type: ignore[attr-defined]
            GroupQuantizeLinear,
        )

    # Check language model quantization (similar to qwen2 tests)
    for i in range(config.num_hidden_layers):
        assert f"language_model.layers.{i}.self_attn.c_attn.weight" in quant_map.param_map
        assert isinstance(
            model.language_model.layers[i].self_attn.c_attn,  # type: ignore[attr-defined]
            GroupQuantizeLinear,
        )

@pytest.mark.parametrize("model_name", ["qwen2_vl"])
@pytest.mark.parametrize(
    "quant_name",
    ["q0"],
)
def test_qwen2_vl_no_quantization(model_name: str, quant_name: str):
    """Test no-quantization mode of Qwen2 VL."""
    model_info = MODELS["qwen2_vl"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    _, quant_map = model_info.quantize["no-quant"](config, QUANTIZATION[quant_name])
    assert len(quant_map.param_map) == 0
    assert len(quant_map.map_func) == 0

if __name__ == "__main__":
    test_vision_transformer()
    test_m_rope_implementation()
    test_qwen2_vl_creation("qwen2_vl")
    test_qwen2_vl_group_quantization("qwen2_vl", "q4f16_1")
    test_qwen2_vl_no_quantization("qwen2_vl", "q0") 