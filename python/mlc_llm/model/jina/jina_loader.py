"""
This file specifies how MLC's BERT parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .jina_model import JinaConfig, JinaModel

MAPPING = [
    ("embeddings.word_embeddings.lora_a", "roberta.embeddings.word_embeddings.parametrizations.weight.0.lora_A"),
    ("embeddings.word_embeddings.lora_b", "roberta.embeddings.word_embeddings.parametrizations.weight.0.lora_B"),
    ("embeddings.word_embeddings.weight", "roberta.embeddings.word_embeddings.parametrizations.weight.original"),
    ("embeddings.token_type_embeddings.lora_a", "roberta.embeddings.token_type_embeddings.parametrizations.weight.0.lora_A"),
    ("embeddings.token_type_embeddings.lora_b", "roberta.embeddings.token_type_embeddings.parametrizations.weight.0.lora_B"),
    ("embeddings.token_type_embeddings.weight", "roberta.embeddings.token_type_embeddings.parametrizations.weight.original"),
    ("embeddings.layer_norm.weight", "roberta.emb_ln.weight"),
    ("embeddings.layer_norm.bias", "roberta.emb_ln.bias"),
    # ("pooler.dense.bias", "roberta.pooler.dense.bias"),
    # ("pooler.dense.lora_a", "roberta.pooler.dense.parametrizations.weight.0.lora_A"),
    # ("pooler.dense.lora_b", "roberta.pooler.dense.parametrizations.weight.0.lora_B"),
    # ("pooler.dense.weight", "roberta.pooler.dense.parametrizations.weight.original"),
]

for i in range(24):
    MAPPING += [
        (f"encoder.layers.{i}.mha.qkv.bias", f"roberta.encoder.layers.{i}.mixer.Wqkv.bias"),
        (f"encoder.layers.{i}.mha.qkv.lora_a", f"roberta.encoder.layers.{i}.mixer.Wqkv.parametrizations.weight.0.lora_A"),
        (f"encoder.layers.{i}.mha.qkv.lora_b", f"roberta.encoder.layers.{i}.mixer.Wqkv.parametrizations.weight.0.lora_B"),
        (f"encoder.layers.{i}.mha.qkv.weight", f"roberta.encoder.layers.{i}.mixer.Wqkv.parametrizations.weight.original"),
        (f"encoder.layers.{i}.mha.out_proj.bias", f"roberta.encoder.layers.{i}.mixer.out_proj.bias"),
        (f"encoder.layers.{i}.mha.out_proj.lora_a", f"roberta.encoder.layers.{i}.mixer.out_proj.parametrizations.weight.0.lora_A"),
        (f"encoder.layers.{i}.mha.out_proj.lora_b", f"roberta.encoder.layers.{i}.mixer.out_proj.parametrizations.weight.0.lora_B"),
        (f"encoder.layers.{i}.mha.out_proj.weight", f"roberta.encoder.layers.{i}.mixer.out_proj.parametrizations.weight.original"),
        (f"encoder.layers.{i}.layer_norm1.weight", f"roberta.encoder.layers.{i}.norm1.weight"),
        (f"encoder.layers.{i}.layer_norm1.bias", f"roberta.encoder.layers.{i}.norm1.bias"),
        (f"encoder.layers.{i}.mlp.fc1.bias", f"roberta.encoder.layers.{i}.mlp.fc1.bias"),
        (f"encoder.layers.{i}.mlp.fc1.lora_a", f"roberta.encoder.layers.{i}.mlp.fc1.parametrizations.weight.0.lora_A"),
        (f"encoder.layers.{i}.mlp.fc1.lora_b", f"roberta.encoder.layers.{i}.mlp.fc1.parametrizations.weight.0.lora_B"),
        (f"encoder.layers.{i}.mlp.fc1.weight", f"roberta.encoder.layers.{i}.mlp.fc1.parametrizations.weight.original"),
        (f"encoder.layers.{i}.mlp.fc2.bias", f"roberta.encoder.layers.{i}.mlp.fc2.bias"),
        (f"encoder.layers.{i}.mlp.fc2.lora_a", f"roberta.encoder.layers.{i}.mlp.fc2.parametrizations.weight.0.lora_A"),
        (f"encoder.layers.{i}.mlp.fc2.lora_b", f"roberta.encoder.layers.{i}.mlp.fc2.parametrizations.weight.0.lora_B"),
        (f"encoder.layers.{i}.mlp.fc2.weight", f"roberta.encoder.layers.{i}.mlp.fc2.parametrizations.weight.original"),
        (f"encoder.layers.{i}.layer_norm2.weight", f"roberta.encoder.layers.{i}.norm2.weight"),
        (f"encoder.layers.{i}.layer_norm2.bias", f"roberta.encoder.layers.{i}.norm2.bias"),
    ]



def huggingface(model_config: JinaConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : BertConfig
        The configuration of the BERT model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = JinaModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for mlc_name, origin_name in MAPPING:
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [origin_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=mlc_param.dtype,
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
