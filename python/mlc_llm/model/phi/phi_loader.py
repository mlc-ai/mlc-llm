"""
This file specifies how MLC's Phi parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .phi_model import Phi1Config, PhiConfig, PhiForCausalLM


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

    def _add(mlc_name, hf_name):
        mapping.add_mapping(
            mlc_name,
            [hf_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    if model_config.model_type == "mixformer-sequential":
        _add("transformer.embd.weight", "layers.0.wte.weight")
        prefix = "transformer.h"
        for i in range(model_config.n_layer):
            _add(f"{prefix}.{i}.ln.weight", f"layers.{i + 1}.ln.weight")
            _add(f"{prefix}.{i}.ln.bias", f"layers.{i + 1}.ln.bias")
            _add(f"{prefix}.{i}.mixer.Wqkv.weight", f"layers.{i + 1}.mixer.Wqkv.weight")
            _add(f"{prefix}.{i}.mixer.Wqkv.bias", f"layers.{i + 1}.mixer.Wqkv.bias")
            _add(f"{prefix}.{i}.mixer.out_proj.weight", f"layers.{i + 1}.mixer.out_proj.weight")
            _add(f"{prefix}.{i}.mixer.out_proj.bias", f"layers.{i + 1}.mixer.out_proj.bias")
            _add(f"{prefix}.{i}.mlp.fc1.weight", f"layers.{i + 1}.mlp.fc1.weight")
            _add(f"{prefix}.{i}.mlp.fc1.bias", f"layers.{i + 1}.mlp.fc1.bias")
            _add(f"{prefix}.{i}.mlp.fc2.weight", f"layers.{i + 1}.mlp.fc2.weight")
            _add(f"{prefix}.{i}.mlp.fc2.bias", f"layers.{i + 1}.mlp.fc2.bias")
            mapping.add_unused(f"layers.{i + 1}.mixer.rotary_emb.inv_freq")
        prefix = f"layers.{model_config.n_layer + 1}"
        _add("lm_head.ln.weight", f"{prefix}.ln.weight")
        _add("lm_head.ln.bias", f"{prefix}.ln.bias")
        _add("lm_head.linear.weight", f"{prefix}.linear.weight")
        _add("lm_head.linear.bias", f"{prefix}.linear.bias")

    elif model_config.model_type == "phi-msft":
        _add("transformer.embd.weight", "transformer.embd.wte.weight")
        for mlc_name, _ in named_parameters.items():
            if mlc_name not in mapping.param_map:
                _add(mlc_name, mlc_name)
    return mapping


def phi1_huggingface(model_config: Phi1Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of Phi-1/Phi-1.5 HuggingFace PyTorch parameters.

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

    def _add(mlc_name, hf_name):
        mapping.add_mapping(
            mlc_name,
            [hf_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    def _concat_add(mlc_name, hf_names):
        mapping.add_mapping(
            mlc_name,
            hf_names,
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    _add("lm_head.linear.weight", "lm_head.weight")
    _add("lm_head.linear.bias", "lm_head.bias")
    _add("lm_head.ln.weight", "model.final_layernorm.weight")
    _add("lm_head.ln.bias", "model.final_layernorm.bias")
    _add("transformer.embd.weight", "model.embed_tokens.weight")

    prefix = "transformer.h"
    hf_prefix = "model.layers"
    for i in range(model_config.num_hidden_layers):
        _add(f"{prefix}.{i}.ln.weight", f"{hf_prefix}.{i}.input_layernorm.weight")
        _add(f"{prefix}.{i}.ln.bias", f"{hf_prefix}.{i}.input_layernorm.bias")
        _concat_add(
            f"{prefix}.{i}.mixer.Wqkv.weight",
            [
                f"{hf_prefix}.{i}.self_attn.q_proj.weight",
                f"{hf_prefix}.{i}.self_attn.k_proj.weight",
                f"{hf_prefix}.{i}.self_attn.v_proj.weight",
            ],
        )
        _concat_add(
            f"{prefix}.{i}.mixer.Wqkv.bias",
            [
                f"{hf_prefix}.{i}.self_attn.q_proj.bias",
                f"{hf_prefix}.{i}.self_attn.k_proj.bias",
                f"{hf_prefix}.{i}.self_attn.v_proj.bias",
            ],
        )
        _add(f"{prefix}.{i}.mixer.out_proj.weight", f"{hf_prefix}.{i}.self_attn.dense.weight")
        _add(f"{prefix}.{i}.mixer.out_proj.bias", f"{hf_prefix}.{i}.self_attn.dense.bias")
        _add(f"{prefix}.{i}.mlp.fc1.weight", f"{hf_prefix}.{i}.mlp.fc1.weight")
        _add(f"{prefix}.{i}.mlp.fc1.bias", f"{hf_prefix}.{i}.mlp.fc1.bias")
        _add(f"{prefix}.{i}.mlp.fc2.weight", f"{hf_prefix}.{i}.mlp.fc2.weight")
        _add(f"{prefix}.{i}.mlp.fc2.bias", f"{hf_prefix}.{i}.mlp.fc2.bias")
        mapping.add_unused(f"{hf_prefix}.{i}.mixer.rotary_emb.inv_freq")

    return mapping
