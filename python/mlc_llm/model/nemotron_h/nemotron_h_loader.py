"""
Weight loader for NemotronH (HuggingFace -> MLC-LLM).

HF weight naming:
  model.layers.{i}.norm.weight
  model.layers.{i}.mixer.in_proj.weight          (mamba layers)
  model.layers.{i}.mixer.conv1d.weight           (mamba layers) [conv_dim, 1, kernel]
  model.layers.{i}.mixer.conv1d.bias             (mamba layers)
  model.layers.{i}.mixer.dt_bias
  model.layers.{i}.mixer.A_log
  model.layers.{i}.mixer.D
  model.layers.{i}.mixer.norm.weight             (gated rmsnorm inside mamba)
  model.layers.{i}.mixer.out_proj.weight
  model.layers.{i}.mixer.q_proj.weight           (attention layers)
  model.layers.{i}.mixer.k_proj.weight
  model.layers.{i}.mixer.v_proj.weight
  model.layers.{i}.mixer.o_proj.weight
  model.layers.{i}.mixer.gate.weight             (moe layers)
  model.layers.{i}.mixer.gate.e_score_correction_bias
  model.layers.{i}.mixer.experts.up_proj         (moe) [n_experts, mid, hidden]
  model.layers.{i}.mixer.experts.down_proj       (moe) [n_experts, hidden, mid]
  model.layers.{i}.mixer.shared_experts.up_proj.weight
  model.layers.{i}.mixer.shared_experts.down_proj.weight
  model.norm_f.weight
  model.embeddings.weight
  lm_head.weight

MLC-LLM weight naming (mirrors our model definition):
  model.layers.{i}.norm.weight
  model.layers.{i}.mixer.in_proj.weight
  model.layers.{i}.mixer.conv1d_weight           (reshaped: [conv_dim, kernel])
  model.layers.{i}.mixer.conv1d_bias
  ... etc
"""

import functools


from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .nemotron_h_model import NemotronHConfig, NemotronHForCausalLM


def huggingface(
    model_config: NemotronHConfig,
    quantization: Quantization,
) -> ExternMapping:
    """Build HF -> MLC-LLM weight mapping for NemotronH."""
    model = NemotronHForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    mapping = ExternMapping()

    def _add(mlc_name: str, hf_names, transform_fn):
        mapping.add_mapping(mlc_name, hf_names, transform_fn)

    def _pass(dtype):
        return functools.partial(lambda x, d: x.astype(d), d=dtype)

    layers_block_type = model_config.layers_block_type

    for i, block_type in enumerate(layers_block_type):
        prefix_mlc = f"model.layers.{i}"
        prefix_hf = f"backbone.layers.{i}"

        # Layer norm (all block types)
        n = f"{prefix_mlc}.norm.weight"
        _add(n, [f"{prefix_hf}.norm.weight"], _pass(named_parameters[n].dtype))

        if block_type == "mamba":
            _hf = f"{prefix_hf}.mixer"
            _mlc = f"{prefix_mlc}.mixer"

            for w in ["in_proj.weight", "dt_bias", "A_log", "D", "out_proj.weight", "norm.weight"]:
                n = f"{_mlc}.{w}"
                _add(n, [f"{_hf}.{w}"], _pass(named_parameters[n].dtype))

            # conv1d: HF stores as [conv_dim, 1, kernel] -> MLC stores as [conv_dim, kernel]
            n = f"{_mlc}.conv1d_weight"
            _add(
                n,
                [f"{_hf}.conv1d.weight"],
                functools.partial(
                    lambda x, d: x.squeeze(1).astype(d),
                    d=named_parameters[n].dtype,
                ),
            )

            if f"{_mlc}.conv1d_bias" in named_parameters:
                n = f"{_mlc}.conv1d_bias"
                _add(n, [f"{_hf}.conv1d.bias"], _pass(named_parameters[n].dtype))

        elif block_type == "mlp":
            _hf = f"{prefix_hf}.mixer"
            _mlc = f"{prefix_mlc}.mixer"

            for w in ["up_proj.weight", "down_proj.weight"]:
                n = f"{_mlc}.{w}"
                _add(n, [f"{_hf}.{w}"], _pass(named_parameters[n].dtype))

        elif block_type == "attention":
            _hf = f"{prefix_hf}.mixer"
            _mlc = f"{prefix_mlc}.mixer"

            for w in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]:
                n = f"{_mlc}.{w}"
                _add(n, [f"{_hf}.{w}"], _pass(named_parameters[n].dtype))

        elif block_type == "moe":
            _hf = f"{prefix_hf}.mixer"
            _mlc = f"{prefix_mlc}.mixer"

            # Router gate
            n = f"{_mlc}.gate.weight"
            _add(n, [f"{_hf}.gate.weight"], _pass(named_parameters[n].dtype))

            n = f"{_mlc}.e_score_correction_bias"
            _add(n, [f"{_hf}.gate.e_score_correction_bias"], _pass(named_parameters[n].dtype))

            # Experts: HF stores as [n_experts, out, in] — MixtralExperts expects same layout
            # experts_up: HF up_proj [n_experts, moe_mid, hidden]
            n = f"{_mlc}.experts_up.weight"
            _add(n, [f"{_hf}.experts.up_proj"], _pass(named_parameters[n].dtype))

            # experts_down: HF down_proj [n_experts, hidden, moe_mid]
            n = f"{_mlc}.experts_down.weight"
            _add(n, [f"{_hf}.experts.down_proj"], _pass(named_parameters[n].dtype))

            # Shared expert
            for w in ["up_proj.weight", "down_proj.weight"]:
                n = f"{_mlc}.shared_expert.{w}"
                _add(n, [f"{_hf}.shared_experts.{w}"], _pass(named_parameters[n].dtype))

    # Final norm
    n = "model.norm.weight"
    _add(n, ["backbone.norm_f.weight"], _pass(named_parameters[n].dtype))

    # Embeddings
    n = "model.embed_tokens.weight"
    _add(n, ["backbone.embeddings.weight"], _pass(named_parameters[n].dtype))

    # LM head - NemotronH uses backbone.embeddings.weight (tied)
    if not model_config.tie_word_embeddings:
        n = "lm_head.weight"
        _add(n, ["lm_head.weight"], _pass(named_parameters[n].dtype))

    # Catch any remaining params not yet mapped (fallback 1:1)
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                _pass(mlc_param.dtype),
            )

    return mapping
