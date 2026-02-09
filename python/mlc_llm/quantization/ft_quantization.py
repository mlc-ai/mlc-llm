"""The FasterTransformer quantization config"""

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple

import tvm
from tvm import DataType, DataTypeCode, IRModule, relax, te, tir
from tvm.relax.frontend import nn
from tvm.runtime import Tensor
from tvm.s_tir import dlight as dl
from tvm.target import Target

from ..loader import QuantizeMapping
from ..op import faster_transformer_dequantize_gemm
from ..support import logging
from ..support.auto_target import detect_cuda_arch_list
from ..support.style import bold
from .group_quantization import (
    GroupQuantize,
    GroupQuantizeEmbedding,
    GroupQuantizeLinear,
)
from .utils import is_final_fc, is_moe_gate

logger = logging.getLogger(__name__)


@dataclass
class FTQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for FasterTransformer quantization"""

    name: str
    kind: str
    quantize_dtype: Literal["int4", "int8"]
    storage_dtype: Literal["int8"]
    model_dtype: Literal["float16"]
    group_size: Optional[int] = None

    num_elem_per_storage: int = 0
    max_int_value: int = 0

    def fallback_group_quantize(self) -> GroupQuantize:
        """
        The fallback group quantization config for other parameters.

        Returns
        ------
        quantize: GroupQuantize
            The group quantization config to fallback.
        """
        return GroupQuantize(
            name=self.name,
            kind="group-quant",
            group_size=32,  # hardcoded to 32 as only supporting int4 quantization
            quantize_dtype=self.quantize_dtype,
            storage_dtype="uint32",
            model_dtype=self.model_dtype,
            linear_weight_layout="NK",
        )

    def __post_init__(self):
        assert self.kind == "ft-quant"
        quantize_dtype = DataType(self.quantize_dtype)
        storage_dtype = DataType(self.storage_dtype)
        assert self.quantize_dtype in ["int4", "int8"]
        assert storage_dtype.type_code == DataTypeCode.INT
        assert self.model_dtype == "float16"
        assert self.group_size in [None, 64, 128]
        if storage_dtype.bits < quantize_dtype.bits:
            raise ValueError("Storage unit should be greater or equal to quantized element")

        self.num_elem_per_storage = storage_dtype.bits // quantize_dtype.bits
        self.max_int_value = (2 ** (quantize_dtype.bits - 1)) - 1
        self._quantize_func_cache = {}

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with FasterTransformer quantization

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """

        class _Mutator(nn.Mutator):
            def __init__(self, config: FTQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """
                The visiting method for FasterTransformer quantization of nn.Module nodes.

                Parameters
                ----------
                name : str
                    The name of the current node.

                node : nn.Module
                    The current node of nn.Module to mutate.

                Returns
                ------
                ret_node: Any
                    The new node to replace current node.
                """
                if isinstance(node, nn.Linear):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [
                        f"{name}.q_weight",
                        f"{name}.q_scale",
                    ]
                    if (
                        # pylint: disable=too-many-boolean-expressions
                        is_final_fc(name)
                        or node.out_dtype == "float32"
                        or (self.config.quantize_dtype == "int4" and node.out_features % 8 != 0)
                        or (self.config.quantize_dtype == "int8" and node.out_features % 4 != 0)
                    ):
                        # Under any of the conditions we fall back to GroupQuantize
                        # For `is_final_fc()` see https://github.com/mlc-ai/mlc-llm/issues/1723
                        # If simply skipping lm_head quantization degrades performance
                        # Other requirements are from CUTLASS
                        logger.info(
                            'Fallback to GroupQuantize for nn.Linear: "%s", '
                            + "weight.shape: %s, out_dtype: %s",
                            bold(name),
                            node.weight.shape,
                            node.out_dtype,
                        )
                        group_quantize = self.config.fallback_group_quantize()
                        self.quant_map.map_func[weight_name] = group_quantize.quantize_weight
                        return GroupQuantizeLinear.from_linear(node, group_quantize)
                    if not is_moe_gate(name, node):
                        self.quant_map.map_func[weight_name] = self.config.quantize_weight
                        return FTQuantizeLinear.from_linear(node, self.config)
                if isinstance(node, nn.Embedding):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [
                        f"{name}.q_weight",
                        f"{name}.q_scale",
                    ]
                    group_quantize = self.config.fallback_group_quantize()
                    self.quant_map.map_func[weight_name] = group_quantize.quantize_weight
                    return GroupQuantizeEmbedding.from_embedding(node, group_quantize)
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def quantize_weight(self, weight: Tensor) -> List[Tensor]:
        """
        Quantize weight with FasterTransformer quantization

        Parameters
        ----------
        weight : Tensor
            The original weight.

        Returns
        ------
        ret: List[Tensor]
            The list of FasterTransformer quantized weights.
        """
        assert tvm.get_global_func("relax.ext.cutlass", True), (
            "Cutlass should be enabled in TVM runtime to quantize weight, "
            "but not enabled in current TVM runtime environment. "
            "To enable Cutlass in TVM runtime, please `set(USE_CUTLASS ON)` "
            "in config.cmake when compiling TVM from source"
        )
        assert len(weight.shape) == 2
        device = weight.device
        device_type = device._DEVICE_TYPE_TO_NAME[  # pylint: disable=protected-access
            device.dlpack_device_type()
        ]
        if device_type == "cuda":
            target = Target.current()
            if target is None:
                target = Target.from_device(device)
            with target:

                def _create_quantize_func() -> IRModule:
                    bb = relax.BlockBuilder()  # pylint: disable=invalid-name
                    weight_var = relax.Var(
                        "weight", relax.TensorStructInfo(weight.shape, weight.dtype)
                    )
                    with bb.function(name="main", params=[weight_var]):
                        with bb.dataflow():
                            lv0 = bb.emit_te(
                                self._quantize, weight_var
                            )  # pylint: disable=invalid-name
                            lv1 = bb.normalize(lv0[0])
                            lv2 = bb.emit(
                                relax.call_pure_packed(
                                    "cutlass.ft_preprocess_weight",
                                    lv1,
                                    detect_cuda_arch_list(target=target)[0],
                                    DataType(self.quantize_dtype).bits == 4,
                                    sinfo_args=lv1.struct_info,
                                )
                            )
                            gv = bb.emit_output(
                                relax.Tuple([lv2, lv0[1]])
                            )  # pylint: disable=invalid-name
                        bb.emit_func_output(gv)
                    return bb.finalize()

                def _compile_quantize_func(mod: IRModule) -> Callable:
                    mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    )(mod)
                    ex = relax.build(mod, target=target)
                    vm = relax.VirtualMachine(ex, device)  # pylint: disable=invalid-name
                    return vm["main"]

                key = str(
                    (
                        int(weight.shape[0]),
                        int(weight.shape[1]),
                        weight.dtype,
                        device_type,
                    )
                )
                quantize_func = self._quantize_func_cache.get(key, None)
                if quantize_func is None:
                    logger.info("Compiling quantize function for key: %s", key)
                    quantize_func = _compile_quantize_func(_create_quantize_func())
                    self._quantize_func_cache[key] = quantize_func
                data = quantize_func(weight)
                return data
        else:
            raise NotImplementedError(f"Device type {device_type} is not supported")

    def _quantize(  # pylint: disable=too-many-locals
        self,
        weight: te.Tensor,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """FasterTransformer quantization for weight tensor, defined in tensor expression."""
        assert len(weight.shape) == 2
        n, k = weight.shape

        cur_group_size = k if not self.group_size else self.group_size
        scale_shape = (tir.ceildiv(k, cur_group_size), n)
        r = te.reduce_axis((0, cur_group_size), name="r")

        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda j, i: te.max(
                tir.if_then_else(
                    j * cur_group_size + r < k,
                    te.abs(weight[i, j * cur_group_size + r]),
                    te.min_value(self.model_dtype),
                ),
                axis=r,
            ),
            name="max_abs_value",
        )
        max_int = tir.const(self.max_int_value, self.model_dtype)
        scale = te.compute(
            scale_shape,
            lambda i, j: max_abs[i, j].astype(self.model_dtype) / max_int,
            name="scale",
        )
        # compute scaled weight
        quantize_dtype = DataType(self.quantize_dtype)
        bin_mask = tir.const((1 << quantize_dtype.bits) - 1, self.storage_dtype)
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda i, j: (
                tir.min(
                    tir.max(
                        tir.round(weight[i, j] / scale[j // cur_group_size, i]),
                        -max_int - 1,
                    ),
                    max_int,
                ).astype(self.storage_dtype)
                & bin_mask
            ),
        )

        quantized_weight_shape = (k, tir.ceildiv(n, self.num_elem_per_storage))
        r = te.reduce_axis((0, self.num_elem_per_storage), name="r")  # pylint: disable=invalid-name
        quantized_weight = te.compute(
            shape=quantized_weight_shape,
            fcompute=lambda j, i: tir.sum(
                tir.if_then_else(
                    i * self.num_elem_per_storage + r < n,
                    scaled_weight[i * self.num_elem_per_storage + r, j]
                    << (
                        r.astype(self.storage_dtype)
                        * tir.const(quantize_dtype.bits, self.storage_dtype)
                    ),
                    tir.const(0, self.storage_dtype),
                ),
                axis=r,
            ),
            name="weight",
        )

        return quantized_weight, scale


class FTQuantizeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module with FasterTransformer quantization"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        config: FTQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        cur_group_size = in_features if not config.group_size else config.group_size
        self.q_weight = nn.Parameter(
            (in_features, tir.ceildiv(out_features, config.num_elem_per_storage)),
            config.storage_dtype,
        )
        self.q_scale = nn.Parameter(
            (tir.ceildiv(in_features, cur_group_size), out_features), config.model_dtype
        )
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @staticmethod
    def from_linear(src: nn.Linear, config: FTQuantize) -> "FTQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a FasterTransformer quantized FTQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : FTQuantize
            The FasterTransformer quantization config.

        Returns
        -------
        ret : FTQuantizeLinear
            The FasterTransformer quantized FTQuantizeLinear layer.
        """
        quantized_linear = FTQuantizeLinear(
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
            bias=getattr(src, "bias", None) is not None,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """
        Forward method for FasterTransformer quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the FasterTransformer quantized linear layer.
        """
        return faster_transformer_dequantize_gemm(
            x, self.q_weight, self.q_scale, self.bias, group_size=self.config.group_size
        )

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        self.q_weight.to(dtype=dtype)
        self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init
