"""HF checkpoint mapping for Gemma 4 large text decoders."""

import functools
from types import SimpleNamespace

import numpy as np

from mlc_llm.loader import ExternMapping

from .gemma4_large_model import Gemma4LargeForCausalLM
from .gemma4_loader import _mark_vision_weights_unused


def _cast(value, dtype):
    return value.astype(dtype)


def _concat(gate, up, dtype):
    return np.concatenate([gate, up], axis=0).astype(dtype)


def huggingface(config, quantization):
    model = Gemma4LargeForCausalLM(config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, parameter in params:
        source = "model." + name
        if ".experts." in name:
            source = source.removesuffix(".weight")
        if ".mlp.gate_up_proj.weight" in name:
            mapping.add_mapping(
                name,
                [
                    source.replace("gate_up_proj", "gate_proj"),
                    source.replace("gate_up_proj", "up_proj"),
                ],
                functools.partial(_concat, dtype=parameter.dtype),
            )
        else:
            mapping.add_mapping(name, [source], functools.partial(_cast, dtype=parameter.dtype))
    _mark_vision_weights_unused(
        mapping, SimpleNamespace(vision_config=config.kwargs.get("vision_config", {}))
    )
    return mapping
