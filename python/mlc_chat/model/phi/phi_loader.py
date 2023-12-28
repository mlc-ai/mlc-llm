"""
This file specifies how MLC's Phi parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
import functools

from mlc_chat.loader import ExternMapping
from mlc_chat.quantization import Quantization

from .phi_model import PhiConfig, PhiForCausalLM


def huggingface(model_config: PhiConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : PhiConfig
        The configuration of the Phi model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = PhiForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=W0632:unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    mapping.add_mapping(
        "transformer.embd.weight",
        ["transformer.embd.wte.weight"],
        functools.partial(
            lambda x, dtype: x.astype(dtype),
            dtype=named_parameters["transformer.embd.weight"].dtype,
        ),
    )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping
