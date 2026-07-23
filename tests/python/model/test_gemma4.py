# pylint: disable=invalid-name,missing-docstring
"""Unit tests for the Gemma 4 text model architecture."""

import pytest

from mlc_llm.model import MODEL_PRESETS, MODELS


def test_gemma4_model_registered():
    """Gemma 4 must appear in the MODELS registry, with both loader sources wired up."""
    assert "gemma4" in MODELS, "gemma4 should be registered in MODELS"
    model_info = MODELS["gemma4"]
    assert model_info.name == "gemma4"
    # huggingface-{torch,safetensor} are the loader sources used by
    # ``mlc_llm convert_weight``.
    assert "huggingface-torch" in model_info.source
    assert "huggingface-safetensor" in model_info.source


def test_gemma4_config_validation():
    """The E2B-it preset must round-trip through Gemma4Config and expose the
    fields the runtime relies on."""
    model_info = MODELS["gemma4"]
    config = model_info.config.from_dict(MODEL_PRESETS["gemma4_e2b_it"])

    # Top-level model_config invariants.
    assert config.vocab_size == 262144

    # Text-config invariants the loader / runtime depend on.
    text = config.text_config
    assert text.hidden_size == 1536
    assert text.intermediate_size == 6144
    assert text.num_hidden_layers == 35
    assert text.num_attention_heads == 8
    assert text.num_key_value_heads == 1
    assert text.head_dim == 256
    assert text.sliding_window_size == 512
    assert text.hidden_size_per_layer_input == 256
    assert text.num_kv_shared_layers == 20
    assert text.use_double_wide_mlp is True
    # Hybrid sliding/full attention pattern: 35 layers, 4-sliding : 1-full.
    assert text.layer_types is not None
    assert len(text.layer_types) == 35
    assert text.layer_types.count("full_attention") == 7
    assert text.layer_types.count("sliding_attention") == 28
    # RoPE bases come from rope_parameters.{sliding,full}_attention.rope_theta.
    assert text.position_embedding_base == 10000.0
    assert text.global_position_embedding_base == 1000000.0
    # Logit softcapping must survive parsing.
    assert text.final_logit_softcapping == 30.0


def test_gemma4_config_rejects_double_wide_mlp_without_shared_kv():
    """``use_double_wide_mlp=True`` without ``num_kv_shared_layers`` must raise.

    Without this guard the layer-type discriminator silently falls through to
    the wrong intermediate_size and the model compiles but produces garbage.
    """
    bad_preset = dict(MODEL_PRESETS["gemma4_e2b_it"])
    bad_text = dict(bad_preset["text_config"])
    bad_text["num_kv_shared_layers"] = 0
    bad_preset["text_config"] = bad_text
    model_info = MODELS["gemma4"]
    with pytest.raises(ValueError, match="num_kv_shared_layers"):
        model_info.config.from_dict(bad_preset)


def test_gemma4_creation_and_export_to_ir():
    """Smoke test: instantiate the Gemma 4 nn.Module and export it to TVM IR.

    This exercises:
    - the embedding setup, including the per-layer split-scaled embedding
    - the 35-layer hybrid sliding/full attention stack
    - the per-layer input gate / projection / norm
    - the proportional / default RoPE branches
    - logit softcapping in the final ``get_logits``

    No compile, no runtime, no GPU. Just confirms the module shape is valid
    and parameter extraction works.
    """
    model_info = MODELS["gemma4"]
    config = model_info.config.from_dict(MODEL_PRESETS["gemma4_e2b_it"])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore[arg-type]
    )

    assert mod is not None
    assert len(named_params) > 0

    # The exported IR should contain the standard MLC entry points for chat.
    fn_names = {gv.name_hint for gv in mod.get_global_vars()}
    for required in ("embed", "prefill", "decode", "create_paged_kv_cache"):
        assert required in fn_names, f"missing required entry point: {required}"


if __name__ == "__main__":
    test_gemma4_model_registered()
    test_gemma4_config_validation()
    test_gemma4_config_rejects_double_wide_mlp_without_shared_kv()
    test_gemma4_creation_and_export_to_ir()
    print("OK")
