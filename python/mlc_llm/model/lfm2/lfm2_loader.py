"""Weight mapping from Hugging Face to MLC for LFM2.

Q, K and V of the attention layers are fused into qkv_proj, and the feed-forward gate (w1)
and up (w3) projections into gate_up_proj; w2 becomes down_proj. Every other weight keeps
its Hugging Face name.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .lfm2_model import Lfm2Config, Lfm2ForCausalLM


def huggingface(model_config: Lfm2Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Lfm2Config
        The configuration of the LFM2 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Lfm2ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    def concat(*arrays, dtype):
        return np.concatenate(arrays, axis=0).astype(dtype)

    def cast(x, dtype):
        return x.astype(dtype)

    for i, layer_type in enumerate(model_config.layer_types):
        prefix = f"model.layers.{i}"
        if layer_type == "full_attention":
            mlc_name = f"{prefix}.self_attn.qkv_proj.weight"
            mapping.add_mapping(
                mlc_name,
                [f"{prefix}.self_attn.{p}_proj.weight" for p in "qkv"],
                functools.partial(concat, dtype=named_parameters[mlc_name].dtype),
            )
        mlc_name = f"{prefix}.feed_forward.gate_up_proj.weight"
        mapping.add_mapping(
            mlc_name,
            [f"{prefix}.feed_forward.w1.weight", f"{prefix}.feed_forward.w3.weight"],
            functools.partial(concat, dtype=named_parameters[mlc_name].dtype),
        )
        mlc_name = f"{prefix}.feed_forward.down_proj.weight"
        mapping.add_mapping(
            mlc_name,
            [f"{prefix}.feed_forward.w2.weight"],
            functools.partial(cast, dtype=named_parameters[mlc_name].dtype),
        )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(cast, dtype=mlc_param.dtype),
            )
    return mapping
