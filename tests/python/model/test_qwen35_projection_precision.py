"""Opt-in projection accumulation for Qwen3.5 text models."""

import numpy as np
import pytest
from tvm.relax.frontend import nn

from mlc_llm.model.qwen35.qwen35_model import Qwen35Config, Qwen35LMHeadModel


def config(**overrides):
    """Small hybrid configuration with one full-attention layer."""
    return Qwen35Config(
        hidden_size=128,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=4,
        head_dim=64,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=256,
        context_window_size=128,
        **overrides,
    )


@pytest.mark.parametrize("dtype", [None, "float32"])
def test_precision_round_trip(dtype):
    original = config(projection_out_dtype=dtype)
    assert Qwen35Config.from_dict(original.asdict()).projection_out_dtype == dtype
    nested = Qwen35Config.from_dict({"text_config": original.asdict()})
    assert nested.projection_out_dtype == dtype
    model = Qwen35LMHeadModel(original)
    model.to("float16")
    projections = [model.lm_head]
    for layer in model.model.layers:
        projections.extend([layer.mlp.gate_up_proj, layer.mlp.down_proj])
        attention = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
        projections.extend(
            value for value in vars(attention).values() if isinstance(value, nn.Linear)
        )
    assert projections
    assert all(module.out_dtype == dtype for module in projections)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", True, 32])
def test_invalid_precision(dtype):
    with pytest.raises(ValueError, match="projection_out_dtype"):
        config(projection_out_dtype=dtype)


def test_projection_retains_small_terms():
    torch = pytest.importorskip("torch")

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            model = Qwen35LMHeadModel(config(projection_out_dtype="float32"))
            self.projection = model.model.layers[3].self_attn.c_attn
            self.to("float16")

        def forward(self, x):
            return self.projection(x).astype(x.dtype)

    model = Projection()
    model.projection.weight.data = np.ones(model.projection.weight.shape, dtype="float16")
    inputs = np.tile(np.array([2048, 1, -2048, 1], dtype="float16"), 32)[None]
    compiled = model.jit(spec={"forward": {"x": nn.spec.Tensor((1, 128), "float16")}})
    actual = compiled["forward"](torch.tensor(inputs))
    np.testing.assert_array_equal(
        actual, np.full((1, model.projection.weight.shape[0]), 64, dtype="float16")
    )


@pytest.mark.parametrize("dtype", [None, "float32"])
def test_quantized_projection_preserves_accumulation_and_bias(dtype):
    from mlc_llm.quantization import QUANTIZATION
    from mlc_llm.quantization.group_quantization import GroupQuantizeLinear

    model = Qwen35LMHeadModel(config(projection_out_dtype=dtype, attention_bias=True))
    model.to("float16")
    source = model.model.layers[3].self_attn.c_attn
    quantized = GroupQuantizeLinear.from_linear(source, QUANTIZATION["q4f16_1"])
    assert quantized.out_dtype == dtype
    assert source.bias.dtype == quantized.bias.dtype == (dtype or "float16")


@pytest.mark.parametrize("dtype", [None, "float32"])
def test_hybrid_export_preserves_activation_and_parameter_dtypes(dtype):
    model = Qwen35LMHeadModel(config(projection_out_dtype=dtype))
    model.to("float16")
    module, parameters = model.export_tvm(spec=model.get_default_spec())
    assert module["batch_prefill"].ret_ty.fields[0].dtype == "float32"
    assert all(
        parameter.dtype == "float16" for name, parameter in parameters if name.endswith("weight")
    )
    script = module.script()
    assert ('out_dtype="float32"' in script) == (dtype == "float32")
