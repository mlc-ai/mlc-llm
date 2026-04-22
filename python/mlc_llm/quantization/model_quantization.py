"""Quantization factory utilities for model quantization."""

from typing import Any, Callable, Dict, Optional, Tuple, Type  # noqa: UP035

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping

from .awq_quantization import AWQQuantize
from .block_scale_quantization import BlockScaleQuantize
from .ft_quantization import FTQuantize
from .group_quantization import GroupQuantize
from .no_quantization import NoQuantize
from .per_tensor_quantization import PerTensorQuantize
from .quantization import Quantization

FuncQuantization = Callable[[Any, Quantization], Tuple[nn.Module, QuantizeMapping]]  # noqa: UP006


def make_quantization_functions(
    model_cls: Type[nn.Module],  # noqa: UP006
    *,
    model_ctor: Optional[Callable[[Any], nn.Module]] = None,
    supports_group_quant: bool = True,
    supports_ft_quant: bool = True,
    supports_awq: bool = False,
    awq_unsupported_message: Optional[str] = None,
    supports_per_tensor: bool = False,
    supports_block_scale: bool = False,
    set_tensor_parallel_shards: bool = True,
    per_tensor_use_shards: bool = True,
) -> Dict[str, FuncQuantization]:  # noqa: UP006
    """Create standard quantization function implementations for a model class."""

    def _create_model(model_config: Any) -> nn.Module:
        if model_ctor is not None:
            return model_ctor(model_config)
        return model_cls(model_config)

    def _no_quant(model_config: Any, quantization: NoQuantize) -> Tuple[nn.Module, QuantizeMapping]:  # noqa: UP006
        model = _create_model(model_config)
        model.to(quantization.model_dtype)
        return model, QuantizeMapping({}, {})

    def _group_quant(
        model_config: Any,
        quantization: GroupQuantize,
    ) -> Tuple[nn.Module, QuantizeMapping]:  # noqa: UP006
        model = _create_model(model_config)
        model.to(quantization.model_dtype)
        quant_map = QuantizeMapping({}, {})
        if set_tensor_parallel_shards:
            if not hasattr(model_config, "tensor_parallel_shards"):
                raise AttributeError(
                    "model_config is missing required "
                    "attribute 'tensor_parallel_shards' for group quantization"
                )
            quantization.tensor_parallel_shards = getattr(model_config, "tensor_parallel_shards")
        model = quantization.quantize_model(
            model,
            quant_map,
            "",
        )
        return model, quant_map

    def _ft_quant(model_config: Any, quantization: FTQuantize) -> Tuple[nn.Module, QuantizeMapping]:  # noqa: UP006
        model = _create_model(model_config)
        model.to(quantization.model_dtype)
        quant_map = QuantizeMapping({}, {})
        model = quantization.quantize_model(
            model,
            quant_map,
            "",
        )
        return model, quant_map

    def _awq_quant(
        model_config: Any, quantization: AWQQuantize
    ) -> Tuple[nn.Module, QuantizeMapping]:  # noqa: UP006
        if awq_unsupported_message is not None:
            raise NotImplementedError(awq_unsupported_message)
        model = _create_model(model_config)
        model.to(quantization.model_dtype)
        quant_map = QuantizeMapping({}, {})
        model = quantization.quantize_model(
            model,
            quant_map,
            "",
        )
        return model, quant_map

    def _per_tensor_quant(
        model_config: Any,
        quantization: PerTensorQuantize,
    ) -> Tuple[nn.Module, QuantizeMapping]:  # noqa: UP006
        model = _create_model(model_config)
        model.to(quantization.model_dtype)
        quant_map = QuantizeMapping({}, {})
        kwargs = {}
        if per_tensor_use_shards:
            if not hasattr(model_config, "tensor_parallel_shards"):
                raise AttributeError(
                    "model_config is missing required attribute "
                    "'tensor_parallel_shards' for per-tensor quantization"
                )
            kwargs["tensor_parallel_shards"] = getattr(model_config, "tensor_parallel_shards")
        model = quantization.quantize_model(
            model,
            quant_map,
            "",
            **kwargs,
        )
        return model, quant_map

    def _block_scale_quant(
        model_config: Any,
        quantization: BlockScaleQuantize,
    ) -> Tuple[nn.Module, QuantizeMapping]:  # noqa: UP006
        model = _create_model(model_config)
        model.to(quantization.model_dtype)
        quant_map = QuantizeMapping({}, {})
        model = quantization.quantize_model(model, quant_map, "")
        return model, quant_map

    quantize_fns: Dict[str, FuncQuantization] = {"no-quant": _no_quant}  # noqa: UP006
    if supports_group_quant:
        quantize_fns["group-quant"] = _group_quant
    if supports_ft_quant:
        quantize_fns["ft-quant"] = _ft_quant
    if supports_awq:
        quantize_fns["awq"] = _awq_quant
    if supports_per_tensor:
        quantize_fns["per-tensor-quant"] = _per_tensor_quant
    if supports_block_scale:
        quantize_fns["block-scale-quant"] = _block_scale_quant
    return quantize_fns


def make_awq_quant(
    model_cls: Type[nn.Module],  # noqa: UP006
) -> Callable[[Any, AWQQuantize], Tuple[nn.Module, QuantizeMapping]]:  # noqa: UP006
    """Create a standard AWQ quantization function for loaders."""

    def awq_quant(
        model_config: Any, quantization: AWQQuantize
    ) -> Tuple[nn.Module, QuantizeMapping]:  # noqa: UP006
        model = model_cls(model_config)
        model.to(quantization.model_dtype)
        quant_map = QuantizeMapping({}, {})
        model = quantization.quantize_model(
            model,
            quant_map,
            "",
        )
        return model, quant_map

    return awq_quant
