"""Standard HuggingFace loader mapping helpers."""

from __future__ import annotations

import functools
from typing import Callable, Iterable, Optional, Sequence, Type

import numpy as np
from tvm.relax.frontend import nn  # type: ignore[import]

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

NameTransform = Callable[[str], str]
ExportSpecGetter = Callable[[nn.Module], object]


def _default_export_spec(model: nn.Module) -> object:
    return model.get_default_spec()


def make_standard_hf_loader(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    model_cls: Type[nn.Module],
    layer_prefix: str = "model.layers",
    qkv_names: Sequence[str] = ("q_proj", "k_proj", "v_proj"),
    qkv_concat_axis: int = 0,
    qkv_target_name: str = "qkv_proj",
    add_qkv_bias: bool = False,
    qkv_bias_optional: bool = False,
    gate_up_names: Sequence[str] = ("gate_proj", "up_proj"),
    gate_up_concat_axis: int = 0,
    gate_up_target_name: str = "gate_up_proj",
    include_qkv: bool = True,
    include_gate_up: bool = True,
    add_unused: Optional[Iterable[str]] = None,
    hf_prefix: str = "model.",
    name_transform: Optional[NameTransform] = None,
    export_spec_getter: Optional[ExportSpecGetter] = None,
    num_layers_getter: Optional[Callable[[object], int]] = None,
) -> Callable[[object, Quantization], ExternMapping]:
    """Create a standard loader for HuggingFace weights.

    This handles the common QKV concatenation, gate+up concatenation, optional
    QKV bias mapping, and passes through remaining parameters 1:1.
    """

    if not qkv_names:
        include_qkv = False
    if not gate_up_names:
        include_gate_up = False
    if not include_qkv:
        qkv_names = ()
    if not include_gate_up:
        gate_up_names = ()

    def _default_name_transform(name: str) -> str:
        # When hf_prefix is empty, strip the "model." prefix so models that
        # expose bare top-level weights (no "model." namespace) still load.
        if hf_prefix == "":
            return name[6:] if name.startswith("model.") else name
        return name

    name_transform_fn = name_transform or _default_name_transform
    spec_getter = export_spec_getter or _default_export_spec
    unused_names = tuple(add_unused or ())

    def huggingface(  # pylint: disable=too-many-locals,too-many-branches
        model_config: object,
        quantization: Quantization,
    ) -> ExternMapping:
        model = model_cls(model_config)
        if quantization is not None:
            model.to(quantization.model_dtype)
        _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
            spec=spec_getter(model),
            allow_extern=True,
        )
        named_parameters = dict(_named_params)
        mapping = ExternMapping()

        if include_qkv or include_gate_up or unused_names:
            if num_layers_getter is None:
                num_layers = model_config.num_hidden_layers  # type: ignore[attr-defined]
            else:
                num_layers = num_layers_getter(model_config)

            for i in range(num_layers):
                attn = f"{layer_prefix}.{i}.self_attn"
                if include_qkv:
                    mlc_qkv_name = f"{attn}.{qkv_target_name}.weight"
                    mlc_param = named_parameters[mlc_qkv_name]
                    mapping.add_mapping(
                        mlc_qkv_name,
                        [name_transform_fn(f"{attn}.{name}.weight") for name in qkv_names],
                        functools.partial(
                            lambda q, k, v, dtype: np.concatenate(
                                [q, k, v], axis=qkv_concat_axis
                            ).astype(dtype),
                            dtype=mlc_param.dtype,
                        ),
                    )

                    if add_qkv_bias:
                        mlc_bias_name = f"{attn}.{qkv_target_name}.bias"
                        if (not qkv_bias_optional) or mlc_bias_name in named_parameters:
                            mlc_param = named_parameters[mlc_bias_name]
                            mapping.add_mapping(
                                mlc_bias_name,
                                [name_transform_fn(f"{attn}.{name}.bias") for name in qkv_names],
                                functools.partial(
                                    lambda q, k, v, dtype: np.concatenate(
                                        [q, k, v], axis=qkv_concat_axis
                                    ).astype(dtype),
                                    dtype=mlc_param.dtype,
                                ),
                            )

                if include_gate_up:
                    mlp = f"{layer_prefix}.{i}.mlp"
                    mlc_gate_up_name = f"{mlp}.{gate_up_target_name}.weight"
                    if gate_up_names:
                        mlc_param = named_parameters[mlc_gate_up_name]
                        mapping.add_mapping(
                            mlc_gate_up_name,
                            [name_transform_fn(f"{mlp}.{name}.weight") for name in gate_up_names],
                            functools.partial(
                                lambda gate, up, dtype: np.concatenate(
                                    [gate, up], axis=gate_up_concat_axis
                                ).astype(dtype),
                                dtype=mlc_param.dtype,
                            ),
                        )

                for unused_name in unused_names:
                    mapping.add_unused(name_transform_fn(f"{attn}.{unused_name}"))

        for mlc_name, mlc_param in named_parameters.items():
            if mlc_name not in mapping.param_map:
                mapping.add_mapping(
                    mlc_name,
                    [name_transform_fn(mlc_name)],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )

        return mapping

    return huggingface
