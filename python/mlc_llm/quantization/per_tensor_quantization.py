"""The per-tensor quantization config"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from tvm import DataType, DataTypeCode, IRModule
from tvm import dlight as dl
from tvm import relax, te, tir, topi
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.target import Target

from mlc_llm.loader import QuantizeMapping
from mlc_llm.nn import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp

from .utils import (
    is_final_fc,
    convert_uint_packed_fp8_to_float,
    compile_quantize_func,
    apply_sharding,
)

logger = logging.getLogger(__name__)


@dataclass
class PerTensorQuantize:
    name: str
    kind: str
    activation_dtype: Literal["e4m3_float8", "e5m2_float8", "float16"]
    weight_dtype: Literal["e4m3_float8", "e5m2_float8", "float16"]
    storage_dtype: Literal["uint32"]
    model_dtype: Literal["float16"]
    quantize_embedding: bool = True
    quantize_linear: bool = True
    quantize_final_fc: bool = True

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    max_int_value: int = 0
    no_scale: bool = False

    def __post_init__(self):
        assert self.kind == "per-tensor-quant"
        self.num_elem_per_storage = (
            DataType(self.storage_dtype).bits // DataType(self.weight_dtype).bits
        )
        self.max_int_value = int(tir.max_value(self.weight_dtype).value)
        self._quantize_func_cache = {}

    def quantize_model(
        self, model: nn.Module, quant_map: QuantizeMapping, name_prefix: str
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

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """

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
                    if not self.config.no_scale
                    else [
                        f"{name}.q_weight",
                    ]
                )
                if (
                    isinstance(node, nn.Linear)
                    and self.config.quantize_linear
                    and (not is_final_fc(name) or self.config.quantize_final_fc)
                ):
                    self.quant_map.param_map[weight_name] = param_names
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return PerTensorQuantizeLinear.from_linear(node, self.config)
                if isinstance(node, nn.Embedding) and self.config.quantize_embedding:
                    self.quant_map.param_map[weight_name] = param_names
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return PerTensorQuantizeEmbedding.from_embedding(node, self.config)
                if isinstance(node, MixtralExperts):
                    self.quant_map.param_map[weight_name] = param_names
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    op = PerTensorQuantizeMixtralExperts.from_mixtral_experts(node, self.config)
                    self.quant_map = op.add_calibration_params(self.quant_map, name)
                    return op
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def quantize_weight(self, weight) -> Union[Tuple[NDArray, NDArray], NDArray]:
        """
        Quantize weight with per-tensor quantization.

        Parameters
        ----------
        weight : NDArray
            The weight to quantize.

        Returns
        -------
        ret : Union[Tuple[NDArray, NDArray], NDArray]
            The quantized weight and scale if output_transpose is True, otherwise the quantized weight.
        """
        device = weight.device
        device_type = device.MASK2STR[device.device_type]

        def _create_quantize_func() -> IRModule:
            if DataType(self.weight_dtype).type_code in [
                DataTypeCode.E4M3Float,
                DataTypeCode.E5M2Float,
            ]:
                quantize_func = self._quantize_float8
            else:
                assert NotImplementedError()

            class Quantizer(nn.Module):
                def main(self, weight: nn.Tensor):
                    return quantize_func(weight)

            mod = Quantizer()
            mod, _ = mod.export_tvm(
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

    def _quantize_float8(self, weight: nn.Tensor):
        shape = weight.shape
        quantize_dtype = DataType(self.weight_dtype)

        if self.no_scale:
            # TODO(csullivan, vinx13): Ensure scheduling is applied after lowering relax ops.
            # Currently, dlight scheduling is applied before the R.cast is lowered.
            # scaled_weight = weight.astype(self.weight_dtype)
            scaled_weight = nn.tensor_expr_op(
                lambda scaled_weight: te.compute(
                    shape=weight.shape,
                    fcompute=lambda *idx: scaled_weight(*idx).astype(self.weight_dtype),
                    name="cast",
                ),
                "cast_weight",
                args=[weight],
            )
        else:
            from .fp8_quantization import quantize

            scaled_weight, scale = quantize(
                weight,
                quantize_dtype=quantize_dtype,
                kind="fp8-max",
                max_int_value=self.max_int_value,
            )

        if self.num_elem_per_storage == 1:
            quantized_weight = nn.tensor_expr_op(
                lambda scaled_weight: te.compute(
                    shape=scaled_weight.shape,
                    fcompute=lambda *idx: tir.reinterpret(self.storage_dtype, scaled_weight(*idx)),
                    name="quantized_weight",
                ),
                "quantized_weight",
                args=[scaled_weight],
            )
        else:
            axis = -1
            k = shape[axis]
            r = te.reduce_axis(
                (0, self.num_elem_per_storage), name="r"
            )  # pylint: disable=invalid-name
            quantized_weight_shape = (
                *weight.shape[:axis],
                tir.ceildiv(weight.shape[axis], self.num_elem_per_storage),
            )
            quantized_weight = nn.tensor_expr_op(
                lambda scaled_weight: te.compute(
                    shape=quantized_weight_shape,
                    fcompute=lambda *idx: tir.sum(
                        tir.if_then_else(
                            idx[axis] * self.num_elem_per_storage + r < k,
                            tir.reinterpret(
                                "uint8",
                                scaled_weight(
                                    *idx[:axis], idx[axis] * self.num_elem_per_storage + r
                                ),
                            ).astype(self.storage_dtype)
                            << (r * quantize_dtype.bits),
                            0,
                        ),
                        axis=r,
                    ),
                    name="quantized_weight",
                ),
                "quantized_weight",
                args=[scaled_weight],
            )

        if self.no_scale:
            return (quantized_weight,)
        return quantized_weight, scale

    def _quantize_float16(self, weight: nn.Tensor):
        shape = weight.shape
        return (weight,)

    def _dequantize(
        self,
        q_weight: te.Tensor,
        scale: Optional[te.Tensor] = None,
        out_shape: Optional[List[tir.PrimExpr]] = None,
    ) -> te.Tensor:
        if not self.no_scale:
            assert scale is not None
        if DataType(self.weight_dtype).type_code in [
            DataTypeCode.E4M3Float,
            DataTypeCode.E5M2Float,
        ]:
            return self._dequantize_float8(q_weight, scale, out_shape)
        raise NotImplementedError()

    def _dequantize_float8(
        self,
        q_weight: te.Tensor,
        scale: Optional[te.Tensor] = None,
        out_shape: Optional[List[tir.PrimExpr]] = None,
    ) -> te.Tensor:
        if out_shape is None:
            out_shape = (*q_weight.shape[:-1], q_weight.shape[-1] * self.num_elem_per_storage)

        if self.num_elem_per_storage == 1:
            weight = te.compute(
                shape=out_shape,
                fcompute=lambda *idx: tir.reinterpret(self.weight_dtype, q_weight(*idx)).astype(
                    self.model_dtype
                ),
                name="dequantize_weight",
            )
        else:
            weight = convert_uint_packed_fp8_to_float(
                q_weight,
                DataType(self.weight_dtype).bits,
                self.num_elem_per_storage,
                self.storage_dtype,
                self.model_dtype,
                self.weight_dtype,
                axis=-1,
                out_shape=out_shape,
            )

        if not self.no_scale:
            weight = weight * scale
        return weight


class PerTensorQuantizeLinear(nn.Module):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: Union[int, tir.Var],
        config: PerTensorQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        """
        Converts a non-quantized nn.Linear to a per-tensor quantized PerTensorQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : PerTensorQuantize
            The per-tensor quantization config.

        Returns
        -------
        ret: PerTensorQuantizeLinear
            The per-tensor quantized linear layer.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        self.q_weight = nn.Parameter(
            (out_features, tir.ceildiv(in_features, config.num_elem_per_storage)),
            config.storage_dtype,
        )
        if not config.no_scale:
            self.q_scale = nn.Parameter((1,), config.model_dtype)
        else:
            self.q_scale = None
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, src: nn.Linear, config: PerTensorQuantize) -> "PerTensorQuantizeLinear":
        # For dynamic shape, src.out_features is `"name"`; src.weight.shape[0] is `tir.Var("name")`
        out_features, in_features = src.weight.shape
        quantized_linear = cls(
            in_features=in_features,
            out_features=out_features,
            config=config,
            bias=getattr(src, "bias", None) is not None,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs

        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_linear.q_weight)
            apply_sharding(
                tp.ShardScalar(name=shard.name), f"{shard.name}_q_scale", quantized_linear.q_scale
            )
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """
        Forward method for group quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the group quantized linear layer.
        """
        assert DataType(self.config.weight_dtype).type_code in [
            DataTypeCode.E4M3Float,
            DataTypeCode.E5M2Float,
        ]
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                out_shape=[
                    tir.IntImm("int64", self.out_features),
                    tir.IntImm("int64", self.in_features),
                ],
            ),
            "dequantize",
            args=[self.q_weight, self.q_scale],
        )
        w = nn.op.permute_dims(w)
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
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
        if not self.config.no_scale:
            self.q_scale = nn.Parameter((1,), config.model_dtype)
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
        Forward method for group quantized embedding layer.

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

    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        config: PerTensorQuantize,
    ):  # pylint: disable=too-many-arguments
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.q_weight = nn.Parameter(
            (
                num_local_experts,
                out_features,
                tir.ceildiv(in_features, config.num_elem_per_storage),
            ),
            config.storage_dtype,
        )
        if not config.no_scale:
            self.q_scale = nn.Parameter((1,), config.model_dtype)
        else:
            self.q_scale = None

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts",
        config: PerTensorQuantize,
    ) -> "PerTensorQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a per-tensor quantized PerTensorQuantizeMixtralExperts

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        config : PerTensorQuantize
            The per-tensor quantization config

        Returns
        -------
        ret : PerTensorQuantizeMixtralExperts
            The per-tensor quantized MixtralExperts layer
        """
        if DataType(config.weight_dtype).type_code in [
            DataTypeCode.E4M3Float,
            DataTypeCode.E5M2Float,
        ]:
            from .fp8_quantization import MixtralExpertsFP8

            # TODO(wuwei): refactor this interface to only pass the conifg
            quantized_mixtral_experts = MixtralExpertsFP8.from_mixtral_experts(
                src,
                config,
                config.activation_dtype,
                config.weight_dtype,
                runtime="max" if "calibration" not in config.name else "max-calibration",
            )
            quantized_mixtral_experts.no_scale = config.no_scale
        else:
            raise NotImplementedError()
        return quantized_mixtral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """Forward method for group quantized mistral experts.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        indptr: nn.Tensor
            The indptr tensor

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the group quantized mistral experts layer.
        """
        raise NotImplementedError()
