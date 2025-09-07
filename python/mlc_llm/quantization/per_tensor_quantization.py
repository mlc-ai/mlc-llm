"""The per-tensor quantization config"""

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union

import numpy as np
from tvm import DataType, DataTypeCode, IRModule, nd, relax, te, tir, topi
from tvm.relax.frontend import nn
from tvm.runtime import NDArray

from mlc_llm.loader import QuantizeMapping
from mlc_llm.nn import MixtralExperts
from mlc_llm.op import cutlass, extern
from mlc_llm.support import logging

from .utils import (
    apply_sharding,
    compile_quantize_func,
    convert_uint_packed_fp8_to_float,
    is_final_fc,
    is_moe_gate,
    pack_weight,
)

logger = logging.getLogger(__name__)


@dataclass
class PerTensorQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for per-tensor quantization"""

    name: str
    kind: str
    activation_dtype: Literal["float8_e4m3fn", "float8_e5m2"]
    weight_dtype: Literal["float8_e4m3fn", "float8_e5m2"]
    storage_dtype: Literal["uint32", "float8_e4m3fn", "float8_e5m2"]
    model_dtype: Literal["float16"]
    quantize_embedding: bool = True
    quantize_final_fc: bool = True
    quantize_linear: bool = True

    num_elem_per_storage: int = 0
    max_int_value: int = 0
    use_scale: bool = True
    # The calibration mode for quantization. If set to "inference", the model is built for
    # inference. This should be used after calibration is done.
    # If set to "max", the model is built for calibration that computes the scale using max value of
    # the activations.
    calibration_mode: Literal["inference", "max"] = "inference"
    tensor_parallel_shards: int = 1

    def __post_init__(self):
        assert self.kind == "per-tensor-quant"
        self.num_elem_per_storage = (
            DataType(self.storage_dtype).bits // DataType(self.weight_dtype).bits
        )
        self.max_int_value = int(tir.max_value(self.weight_dtype).value)
        self._quantize_func_cache = {}

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
        tensor_parallel_shards: int,
    ) -> nn.Module:
        """
        Quantize model with per-tensor quantization

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        tensor_parallel_shards : int
            The number of tensor parallel shards.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """

        self.tensor_parallel_shards = tensor_parallel_shards

        class _Mutator(nn.Mutator):
            def __init__(self, config: PerTensorQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """
                The visiting method for per-tensor quantization of nn.Module nodes.

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
                weight_name = f"{name}.weight"
                param_names = (
                    [f"{name}.q_weight", f"{name}.q_scale"]
                    if self.config.use_scale
                    else [
                        f"{name}.q_weight",
                    ]
                )
                if (
                    isinstance(node, nn.Linear)
                    and self.config.quantize_linear
                    and (not is_final_fc(name) or self.config.quantize_final_fc)
                    and not is_moe_gate(name, node)
                ):
                    self.quant_map.param_map[weight_name] = param_names
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    op = PerTensorQuantizeLinear.from_linear(node, self.config, name)
                elif isinstance(node, nn.Embedding) and self.config.quantize_embedding:
                    self.quant_map.param_map[weight_name] = param_names
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    op = PerTensorQuantizeEmbedding.from_embedding(node, self.config)
                elif isinstance(node, MixtralExperts):
                    self.quant_map.param_map[weight_name] = param_names
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    op = PerTensorQuantizeMixtralExperts.from_mixtral_experts(
                        node, self.config, name
                    )
                else:
                    return self.visit(name, node)

                if hasattr(op, "q_calibration_scale") and op.q_calibration_scale:
                    # update quant_map for calibration scale
                    param_name = f"{name}.q_calibration_scale"
                    old_map_func = self.quant_map.map_func[weight_name]

                    def map_func(*args, **kwargs):
                        # placeholder for calibration scale, the actual value will be set after
                        # calibration.
                        scale = nd.empty(
                            shape=op.q_calibration_scale.shape, dtype=op.q_calibration_scale.dtype
                        )
                        return [*old_map_func(*args, **kwargs), scale]

                    self.quant_map.param_map[weight_name].append(param_name)
                    self.quant_map.map_func[weight_name] = map_func
                return op

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def quantize_weight(self, weight) -> List[NDArray]:
        """
        Quantize weight with per-tensor quantization.

        Parameters
        ----------
        weight : NDArray
            The weight to quantize.

        Returns
        -------
        ret : List[NDArray]
            The quantized weight and the scale if use_scale is True.
        """
        device = weight.device
        device_type = device.DEVICE_TYPE_TO_NAME[device.device_type]

        def _create_quantize_func() -> IRModule:
            if DataType(self.weight_dtype).type_code in [
                DataTypeCode.Float8E4M3FN,
                DataTypeCode.Float8E5M2,
            ]:
                quantize_func = functools.partial(
                    self.quantize_float8,
                    quantize_dtype=self.weight_dtype,
                    storage_dtype=self.storage_dtype,
                )
            else:
                assert NotImplementedError()

            class Quantizer(nn.Module):
                """Quantizer module for per-tensor quantization."""

                def main(self, weight: nn.Tensor):  # pylint: disable=missing-function-docstring
                    return quantize_func(weight)

            mod = Quantizer()
            mod, _ = mod.export_tvm(  # pylint: disable=unbalanced-tuple-unpacking
                spec={"main": {"weight": nn.spec.Tensor(weight.shape, weight.dtype)}}
            )
            return mod

        key = f"({weight.shape}, {weight.dtype}, {device_type}"
        quantize_func = self._quantize_func_cache.get(key, None)
        if quantize_func is None:
            logger.info("Compiling quantize function for key: %s", key)
            quantize_func = compile_quantize_func(_create_quantize_func(), device)
            self._quantize_func_cache[key] = quantize_func
        return quantize_func(weight)

    def quantize_float8(  # pylint: disable=too-many-locals
        self,
        tensor: nn.Tensor,
        quantize_dtype: str,
        storage_dtype: str,
    ) -> Union[Tuple[nn.Tensor], Tuple[nn.Tensor, nn.Tensor]]:
        """Per-tensor quantization for weight tensor, defined in tensor expression."""

        if self.use_scale:
            # min_scaling_factor taken from TRT-LLM
            def _compute_scale(x: te.Tensor) -> te.Tensor:
                max_abs = topi.max(topi.abs(x))
                min_scaling_factor = tir.const(1.0 / (self.max_int_value * 512.0), self.model_dtype)
                scale = topi.maximum(
                    max_abs.astype(self.model_dtype) / self.max_int_value, min_scaling_factor
                ).astype("float32")
                scale = topi.expand_dims(scale, axis=0)
                return scale

            scale = nn.tensor_expr_op(_compute_scale, "compute_scale", args=[tensor])
        else:
            scale = None

        def _compute_quantized_tensor(weight: te.Tensor, scale: Optional[te.Tensor]) -> te.Tensor:
            elem_storage_dtype = (
                f"uint{DataType(quantize_dtype).bits}"
                if DataType(storage_dtype).type_code == DataTypeCode.UINT
                else quantize_dtype
            )
            scaled_tensor = te.compute(
                shape=weight.shape,
                fcompute=lambda *idx: tir.Cast(
                    self.storage_dtype,
                    tir.reinterpret(
                        elem_storage_dtype,
                        tir.Cast(
                            quantize_dtype,
                            weight(*idx) / scale(0) if scale is not None else weight(*idx),
                        ),
                    ),
                ),
            )

            if quantize_dtype == self.storage_dtype:
                return scaled_tensor

            packed_weight = pack_weight(
                scaled_tensor,
                axis=-1,
                num_elem_per_storage=self.num_elem_per_storage,
                weight_dtype=self.weight_dtype,
                storage_dtype=self.storage_dtype,
            )

            return packed_weight

        quantized_tensor = nn.tensor_expr_op(
            _compute_quantized_tensor, "compute_quantized_tensor", args=[tensor, scale]
        )

        if self.use_scale:
            return quantized_tensor, scale
        return (quantized_tensor,)

    def _dequantize(
        self,
        q_weight: te.Tensor,
        scale: Optional[te.Tensor] = None,
        out_shape: Optional[Sequence[tir.PrimExpr]] = None,
    ) -> te.Tensor:
        if self.use_scale:
            assert scale is not None
        if DataType(self.weight_dtype).type_code in [
            DataTypeCode.Float8E4M3FN,
            DataTypeCode.Float8E5M2,
        ]:
            return self.dequantize_float8(q_weight, scale, self.weight_dtype, out_shape)
        raise NotImplementedError()

    def dequantize_float8(
        self,
        q_tensor: te.Tensor,
        scale: Optional[te.Tensor],
        quantize_dtype: str,
        out_shape: Optional[Sequence[tir.PrimExpr]] = None,
    ) -> te.Tensor:
        """Dequantize a fp8 tensor (input or weight) to higher-precision float."""
        if quantize_dtype != self.storage_dtype:
            dequantized_tensor = convert_uint_packed_fp8_to_float(
                q_tensor,
                self.num_elem_per_storage,
                self.storage_dtype,
                self.model_dtype,
                quantize_dtype,
                axis=-1,
                out_shape=out_shape,
            )
        else:
            dequantized_tensor = q_tensor.astype(self.model_dtype)
        if scale is not None:
            dequantized_tensor = dequantized_tensor * scale.astype(dequantized_tensor.dtype)
        return dequantized_tensor


class PerTensorQuantizeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module with per-tensor quantization."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: Union[int, tir.Var],
        config: PerTensorQuantize,
        name: str,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype or config.model_dtype
        self.config = config
        self.name = name
        self.q_weight = nn.Parameter(
            (out_features, tir.ceildiv(in_features, config.num_elem_per_storage)),
            config.storage_dtype,
        )
        self.q_calibration_scale = None
        if config.use_scale:
            self.q_scale = nn.Parameter((1,), "float32")
            if config.calibration_mode == "inference":
                self.q_calibration_scale = nn.Parameter((1,), "float32")
        else:
            self.q_scale = None
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, src: nn.Linear, config: PerTensorQuantize, name: str
    ) -> "PerTensorQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a per-tensor quantized PerTensorQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : PerTensorQuantize
            The per-tensor quantization config.

        name: str
            The name of the layer.

        Returns
        -------
        ret : PerTensorQuantizeLinear
            The per-tensor quantized PerTensorQuantizeLinear layer.
        """
        out_features, in_features = src.weight.shape
        quantized_linear = cls(
            in_features=in_features,
            out_features=out_features,
            config=config,
            name=name,
            bias=getattr(src, "bias", None) is not None,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_linear.q_weight)
            # scale doesn't need to be sharded since it's the same for all shards
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """
        Forward method for per-tensor quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the per-tensor quantized linear layer.
        """
        # Note: Use calibration scale when calibration is enabled
        if self.config.calibration_mode == "inference":
            if self.q_calibration_scale:
                x /= self.q_calibration_scale.astype(x.dtype)
            x_q = x.astype(self.config.activation_dtype)
            x_scale = self.q_calibration_scale
        elif self.config.calibration_mode == "max":
            _, x_scale = self.config.quantize_float8(  # type: ignore
                x,
                quantize_dtype=self.config.activation_dtype,
                storage_dtype=self.config.storage_dtype,
            )
            if self.config.tensor_parallel_shards > 1:
                x_scale = nn.ccl_allreduce(x_scale, "max")
            x_scale = nn.extern(
                "mlc_llm.calibration_observer",
                [f"{self.name}.q_calibration_scale", "max", x_scale],
                out=nn.Tensor.placeholder(x_scale.shape, x_scale.dtype),
            )
            x_q = (x / x_scale.astype(x.dtype)).astype(self.config.activation_dtype)
            x = x_q.astype(self.config.model_dtype) * x_scale.astype(self.config.model_dtype)
        else:
            raise ValueError(f"Unknown calibration mode: {self.config.calibration_mode}")

        if (
            self.config.weight_dtype == self.config.storage_dtype
            and self.config.calibration_mode == "inference"
        ):
            if (
                extern.get_store().cutlass_gemm
                and functools.reduce(lambda x, y: x * y, x_q.shape[:-1]) != 1
            ):
                # Dispatch to cutlass kernel for gemm when cutlass is available.
                scale = (
                    x_scale * self.q_scale
                    if self.config.use_scale
                    else nn.wrap_nested(
                        relax.Constant(nd.array(np.array([1.0]).astype("float32"))), "scale"
                    )
                )
                return cutlass.fp8_gemm(
                    x_q, self.q_weight, scale, self.config.weight_dtype, self.config.model_dtype
                )
            x = nn.op.matmul(x_q, nn.permute_dims(self.q_weight), out_dtype="float32")
            if self.config.use_scale:
                scale = x_scale * self.q_scale
                x = x * scale
            x = x.astype(self.out_dtype)
        else:
            w = nn.op.tensor_expr_op(
                lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                    weight,
                    scale,
                    out_shape=[
                        (
                            tir.IntImm("int64", self.out_features)
                            if isinstance(self.out_features, int)
                            else weight.shape[0]
                        ),
                        tir.IntImm("int64", self.in_features),
                    ],
                ),
                "dequantize",
                args=[self.q_weight, self.q_scale],
            )
            x = nn.op.matmul(x, nn.permute_dims(w), out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        self.q_weight.to(dtype=dtype)
        if self.q_scale:
            self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


class PerTensorQuantizeEmbedding(nn.Module):
    """An nn.Embedding module with group quantization"""

    def __init__(self, num: Union[int, tir.Var], dim: int, config: PerTensorQuantize):
        self.num = num
        self.dim = dim
        self.config = config
        self.q_weight = nn.Parameter(
            (num, tir.ceildiv(dim, config.num_elem_per_storage)), config.storage_dtype
        )
        if self.config.use_scale:
            self.q_scale = nn.Parameter((1,), "float32")
        else:
            self.q_scale = None

    @staticmethod
    def from_embedding(
        embedding: nn.Embedding, config: PerTensorQuantize
    ) -> "PerTensorQuantizeEmbedding":
        """
        Converts a non-quantized nn.Embedding to a per-tensor quantized PerTensorQuantizeEmbedding

        Parameters
        ----------
        linear : nn.Embedding
            The non-quantized nn.Embedding.

        config : PerTensorQuantize
            The per-tensor quantization config.

        Returns
        -------
        ret : PerTensorQuantizeEmbedding
            The per-tensor quantized embedding layer.
        """
        num, dim = embedding.weight.shape
        return PerTensorQuantizeEmbedding(num, dim, config)

    def forward(self, x: nn.Tensor):  # pylint: disable=invalid-name
        """
        Forward method for per-tensor quantized embedding layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the embedding layer.
        """
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                out_shape=[
                    tir.IntImm("int64", self.num) if isinstance(self.num, int) else weight.shape[0],
                    tir.IntImm("int64", self.dim),
                ],
            ),
            "dequantize",
            args=[self.q_weight, self.q_scale],
        )
        if x.ndim == 1:
            return nn.op.take(w, x, axis=0)
        return nn.op.reshape(
            nn.op.take(w, nn.op.reshape(x, shape=[-1]), axis=0),
            shape=[*x.shape, self.dim],
        )

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which dequantizes the weight
        and multiplies it with the input tensor.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the lm_head layer.
        """
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                out_shape=[
                    tir.IntImm("int64", self.num) if isinstance(self.num, int) else weight.shape[0],
                    tir.IntImm("int64", self.dim),
                ],
            ),
            "dequantize",
            args=[self.q_weight, self.q_scale],
        )
        w = nn.op.permute_dims(w)
        return nn.op.matmul(x, w, out_dtype="float32")


class PerTensorQuantizeMixtralExperts(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An MixtralExperts module with group quantization"""

    _IMPL: Dict[str, Type["PerTensorQuantizeMixtralExperts"]] = {}

    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        config: PerTensorQuantize,
        name: str,
    ):  # pylint: disable=too-many-arguments
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.name = name
        self.q_weight = nn.Parameter(
            (
                num_local_experts,
                out_features,
                tir.ceildiv(in_features, config.num_elem_per_storage),
            ),
            config.storage_dtype,
        )
        self.q_calibration_scale = None
        if config.use_scale:
            self.q_scale = nn.Parameter((1,), "float32")
            if config.calibration_mode == "inference":
                self.q_calibration_scale = nn.Parameter((1,), "float32")
        else:
            self.q_scale = None

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts",
        config: PerTensorQuantize,
        name: str,
    ) -> "PerTensorQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a per-tensor quantized
        PerTensorQuantizeMixtralExperts

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        config : PerTensorQuantize
            The per-tensor quantization config

        name: str
            The name of the layer.

        Returns
        -------
        ret : PerTensorQuantizeMixtralExperts
            The per-tensor quantized MixtralExperts layer
        """
        if DataType(config.weight_dtype).type_code in [
            DataTypeCode.Float8E4M3FN,
            DataTypeCode.Float8E5M2,
        ]:
            return PerTensorQuantizeMixtralExperts._IMPL["fp8"].from_mixtral_experts(
                src, config, name
            )
        raise NotImplementedError()

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """Forward method for per-tensor quantized mistral experts.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        indptr: nn.Tensor
            The indptr tensor

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the per-tensor quantized mistral experts layer.
        """
        raise NotImplementedError()
