# pylint: disable=invalid-name,missing-docstring
"""Unit tests for SigLIP Vision Encoder."""

from tvm.relax.frontend.nn import spec as nn_spec

from mlc_llm.model.vision import SigLIPVisionConfig, SigLIPVisionModel


SMALL_SIGLIP_CONFIG = {
    "hidden_size": 64,
    "image_size": 28,
    "intermediate_size": 128,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "patch_size": 14,
    "num_channels": 3,
    "layer_norm_eps": 1e-6,
}


def test_siglip_creation():
    """Test SigLIP Vision Model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from dict
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters include all expected components
    """
    config = SigLIPVisionConfig.from_dict(SMALL_SIGLIP_CONFIG)
    model = SigLIPVisionModel(config)
    image_size = config.image_size
    mod_spec = nn_spec.ModuleSpec.from_raw(
        {
            "forward": {
                "pixel_values": nn_spec.Tensor(
                    [1, config.num_channels, image_size, image_size], "float32"
                ),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
        },
        model,
    )
    mod, named_params = model.export_tvm(spec=mod_spec)

    assert mod is not None
    assert len(named_params) > 0

    param_names = [name for name, _ in named_params]

    # Verify key components exist in parameters
    has_patch_embed = any("patch_embedding" in n for n in param_names)
    has_position_embed = any("position_embedding" in n for n in param_names)
    has_encoder_layers = any("encoder.layers" in n for n in param_names)
    has_post_layernorm = any("post_layernorm" in n for n in param_names)

    assert has_patch_embed, "Should have patch_embedding parameters"
    assert has_position_embed, "Should have position_embedding parameters"
    assert has_encoder_layers, "Should have encoder layer parameters"
    assert has_post_layernorm, "Should have post_layernorm parameters"

    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


if __name__ == "__main__":
    test_siglip_creation()
