""" Quantization techniques for FP8 """
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from tvm import DataType, DataTypeCode, IRModule
from tvm import dlight as dl
from tvm import relax, te, tir, topi
from tvm import nd
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.script import tir as T
from tvm.target import Target

from mlc_llm.loader import QuantizeMapping
from mlc_llm.nn import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp

from . import group_quantization as gq
from . import per_tensor_quantization as ptq
from .utils import apply_sharding


def quantize(
    x: nn.Tensor,
    quantize_dtype: str,
    kind="fp8-max",
    scale_tensor: Optional[nn.Tensor] = None,
    compute_scale_only=False,
    name: str = "quantize",
    **kwargs,
) -> Tuple[nn.Tensor, ...]:
    """
    Quantizes the input tensor to a specified lower-precision datatype using different quantization schemes.

    This function supports quantization schemes such as 'fp8-max', where each element in the tensor is scaled and
    quantized to a target datatype that uses fewer bits than the original datatype. The fp8-max range scheme
    scales the tensor values based on the maximum value in the tensor to utilize the full range of the target datatype.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor to be quantized.

    quantize_dtype : DataType
        The target datatype for quantization.

    kind : str, optional
        The kind of quantization scheme to use.

    name : str, optional
        A name hint for the operation.

    **kwargs : dict
        Additional keyword arguments for quantization parameters. For 'fp8-max', 'max_int_value' must be provided,
        which defines the maximum integer value that can be represented in the target datatype.

    Returns
    -------
    result : Tuple[nn.Tensor, ...]
        A list of tensors from the qunatization,
        Usually the quantized tensor, and parameter tensors like scale and zero point

    """
    if kind == "fp8-max":
        # quant: Tensor(dtype="e4m3_float8") = (x / scale); scale: float16 = max(x) / fp8_max_int_value);
        assert (
            "max_int_value" in kwargs
        ), "'max_int_value' must be provided when using fp8-max quantization"
        assert len(kwargs) == 1, f"Unknown additional kwargs: {kwargs}"

        assert (
            compute_scale_only == False or scale_tensor == None
        ), "fp8-max calibration: Cannot provide a scale and request scale computation"

        def _compute_scale(max_abs: te.Tensor) -> te.Tensor:
            max_int = tir.const(kwargs["max_int_value"], x.dtype)
            min_scaling_factor = tir.const(1.0 / (kwargs["max_int_value"] * 512.0), x.dtype)

            scale = te.compute(
                (1,),
                lambda *idx: te.max(
                    max_abs(*idx).astype(x.dtype) / max_int,
                    min_scaling_factor,
                ),
                name="scale",
            )
            return scale

        def _quantize(tensor: te.Tensor, scale: te.Tensor) -> te.Tensor:
            scaled_act = te.compute(
                shape=tensor.shape,
                fcompute=lambda *idx: tir.Cast(
                    quantize_dtype,
                    tensor(*idx) / scale[0],
                ),
            )
            return scaled_act

        def _fused_compute_scale_and_quantize(
            tensor: te.Tensor,
            max_abs: te.Tensor,
            out_shape: Optional[List[tir.PrimExpr]] = None,
        ):
            scale = _compute_scale(max_abs)
            scaled_act = _quantize(tensor, scale)
            return scaled_act, scale

        if compute_scale_only or scale_tensor == None:
            max_abs = nn.op.extern(
                "tvm.contrib.cuda.reduce_max_abs",
                [x],
                nn.Tensor.placeholder((1,), x.dtype),
            )

        if compute_scale_only:
            scale = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda max_tensor: _compute_scale(  # pylint: disable=protected-access
                    max_tensor,
                ),
                name_hint="compute_scale",
                args=[max_abs],
            )
            return scale
        elif scale_tensor == None:
            quant, scale = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda tensor, max_tensor: _fused_compute_scale_and_quantize(  # pylint: disable=protected-access
                    tensor,
                    max_tensor,
                ),
                name_hint="compute_scale_and_quantize",
                args=[x, max_abs],
            )
            return quant, scale
        else:
            quant = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda tensor, scale_t: _quantize(  # pylint: disable=protected-access
                    tensor,
                    scale_t,
                ),
                name_hint="quantize",
                args=[x, scale_tensor],
            )
            return quant
    else:
        raise ValueError("Unknown quantization kind")


def dequantize(
    quant: nn.Tensor,
    scale: nn.Tensor,
    zero: nn.Tensor = None,
    kind="fp8-max",
    name="dequantize",
    **kwargs,
) -> nn.Tensor:
    """
    Dequantizes the input tensor from a specified lower-precision datatype back to a higher-precision datatype.

    This function supports dequantization schemes such as 'fp8-max', where each element in the quantized tensor
    is converted back to a higher-precision format using the provided scale. The 'fp8-max' scheme specifically
    reverses the scaling applied during quantization, without utilizing a zero-point adjustment.

    Parameters
    ----------
    quant : nn.Tensor
        The quantized tensor to be dequantized.

    scale : nn.Tensor
        The scale used during quantization.
        original higher-precision format.

    zero : nn.Tensor, optional
        The zero-point used during quantization.

    kind : str, optional
        The kind of dequantization scheme to use.

    name : str, optional
        A name hint for the operation.

    **kwargs : dict
        Additional keyword arguments for dequantization parameters.

    Returns
    -------
    nn.Tensor
        The dequantized tensor.

    """
    if kind == "fp8-max":
        # dequant: Tensor(dtype="float16") = (quant * scale); scale precompute by quantization
        assert zero == None, "FP8 max range quantization does not utilzie a zero point"
        return quant * scale
    else:
        raise ValueError("Unknown quantization kind")


def inplace_maximum(scale: nn.Tensor, param: nn.Tensor):
    @T.prim_func
    def max_update(
        scale_local: T.Buffer(scale.shape, scale.dtype),
        scale_global: T.Buffer(param.shape, param.dtype),
        # TODO(csullivan): consider using nn.op.tensor_ir_inplace_op
        out_scale: T.Buffer(param.shape, param.dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # TODO(csullivan): use index expansion
        intermediate = T.alloc_buffer(scale_global.shape, dtype=scale_global.dtype)
        for i in range(scale_global.shape[0]):
            with T.block("read"):
                vi = T.axis.remap("S", [i])
                T.reads(scale_global[vi])
                T.writes(intermediate[vi])
                intermediate[vi] = scale_global[vi]
        for i in range(scale_local.shape[0]):
            with T.block("max_update"):
                vi = T.axis.remap("S", [i])
                T.reads(scale_local[vi], intermediate[vi])
                T.writes(scale_global[vi], out_scale[vi])
                scale_global[vi] = T.if_then_else(
                    scale_local[vi] > intermediate[vi], scale_local[vi], intermediate[vi]
                )
                out_scale[vi] = scale_local[vi]

    return nn.op.tensor_ir_op(
        max_update,
        name_hint="inplace_maximum",
        args=[scale, param],
        out=nn.Tensor.placeholder(scale.shape, scale.dtype),
    )


nn.op.quantize = quantize
nn.op.dequantize = dequantize
nn.op.maximum_inplace = inplace_maximum


class MixtralExpertsFP8(
    ptq.PerTensorQuantizeMixtralExperts
):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        weight_config: ptq.PerTensorQuantize,
        activation_dtype: str = None,
        weight_dtype: str = None,
        runtime: str = "cast",
        tensor_parallel_shards=1,
    ):  # pylint: disable=too-many-arguments
        super().__init__(num_local_experts, in_features, out_features, weight_config)
        self.activation_dtype = activation_dtype
        self.weight_dtype = weight_dtype
        self.runtime = runtime
        self.weight_config = weight_config
        self.max_int_value = {
            key: self.get_max_int(dtype)
            for key, dtype in zip(["x", "w"], [self.activation_dtype, self.weight_dtype])
        }

        # TODO(csullivan): Delete this as it is no longer necessary
        self.tensor_parallel_shards = tensor_parallel_shards

        if "max" in self.runtime:
            self.q_calibration_scale = nn.Parameter((1,), weight_config.model_dtype)

    @staticmethod
    def get_max_int(dtype):
        if dtype == "e4m3_float8":
            return 448
        elif dtype == "e5m2_float8":
            return 57344
        elif dtype == "float16":
            return 65504
        else:
            raise NotImplementedError()

    def add_calibration_params(self, quant_map: QuantizeMapping, layer_name: str):
        scale_name = f"{layer_name}.q_calibration_scale"

        def alloc_scale():
            return nd.empty(
                shape=self.q_calibration_scale.shape, dtype=self.q_calibration_scale.dtype
            )

        quant_map.map_func[scale_name] = alloc_scale
        return quant_map

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts",
        weight_config: ptq.PerTensorQuantize,
    ) -> "MixtralExpertsFP8":
        """
        Converts a non-quantized MixtralExperts to a per-tensor quantized MixtralExperts.

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        weight_config : GroupQuantize
            The group quantization weight_config.

        Returns
        -------
        ret : MixtralExpertsFP8
            The per-tensor quantized MixtralExperts.
        """

        quantized_mistral_experts = MixtralExpertsFP8(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            weight_config=weight_config,
            activation_dtype=weight_config.activation_dtype,
            weight_dtype=weight_config.weight_dtype,
            runtime="max" if "calibration" not in weight_config.name else "max-calibration",
            tensor_parallel_shards=src.tensor_parallel_shards,
        )

        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_mistral_experts.q_weight)
            apply_sharding(
                tp.ShardScalar(name=shard.name),
                f"{shard.name}_q_scale",
                quantized_mistral_experts.q_scale,
            )
            if "max" in quantized_mistral_experts.runtime:
                apply_sharding(
                    tp.ShardScalar(name=shard.name),
                    f"{shard.name}_q_calibration_scale",
                    quantized_mistral_experts.q_calibration_scale,
                )

        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        if self.runtime == "max-calibration":
            x, local_scale = nn.op.quantize(
                x,
                quantize_dtype=self.activation_dtype,
                kind="fp8-max",
                max_int_value=self.max_int_value["x"],
            )

            local_scale = nn.op.maximum_inplace(local_scale, self.q_calibration_scale)
            # Calibration done in fp16 mma
            x = nn.op.astype(x, dtype="float16")
            if DataType(self.weight_dtype).type_code in [
                DataTypeCode.E4M3Float,
                DataTypeCode.E5M2Float,
            ]:
                dequant_func = self.config._dequantize_float8
                # The below computes w = cast(reinterpret_cast(self.q_weight, float8_e4m3), float16) * self.q_scale
                w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                    lambda weight, scale: dequant_func(  # pylint: disable=protected-access
                        weight,
                        scale,
                        out_shape=(self.num_local_experts, self.out_features, self.in_features),
                    ),
                    name_hint="dequantize_weight",
                    args=[self.q_weight, self.q_scale],
                )
            elif self.weight_dtype == "float16":
                w = self.q_weight

        elif self.runtime == "max":
            local_scale = self.q_calibration_scale
            x = x / local_scale
            x = nn.op.astype(x, dtype=self.activation_dtype)
            w = self.q_weight
        elif self.runtime == "cast":
            x = nn.op.astype(x, dtype=self.activation_dtype)
            w = self.q_weight
        else:
            raise NotImplementedError(
                f"Only max and cast runtimes are supported for FP8 activations, {self.runtime} is unsupported."
            )

        workspace = nn.op.wrap_nested(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr((4096 * 1024,)),
                dtype="uint8",
                runtime_device_index=0,
            ),
            "relax.alloc_tensor",
        )

        batch_size, in_features = x.shape
        num_local_experts, out_features, _ = self.q_weight.shape

        if self.runtime == "max-calibration":
            func = "cutlass.group_gemm_scale_fp16_sm90"
        else:
            a_format = self.activation_dtype.split("_")[0]
            w_format = self.weight_dtype.split("_")[0]
            func = f"cutlass.group_gemm_{a_format}_{w_format}_fp16"

        if self.runtime == "cast":
            func = func + "_host_scale"
            total_scale = 1.0
        else:
            if self.runtime != "max-calibration" and self.weight_dtype == "e4m3_float8":
                # for calibration, q_scale is already used to dequantize the weights
                total_scale = local_scale * self.q_scale
            else:
                total_scale = local_scale
            total_scale = nn.op.astype(total_scale, dtype="float32")

        return nn.op.extern(
            func,
            [
                x,
                w,
                indptr,
                workspace,
                total_scale,
            ],
            out=nn.Tensor.placeholder(
                (batch_size, out_features),
                dtype=self.weight_config.model_dtype,
            ),
        )


# TODO(csullivan): Refactor Linear and MixtralExperts to shared base with common code
class PTQLinearFP8(ptq.PerTensorQuantizeLinear):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        in_features,
        out_features,
        weight_config: ptq.PerTensorQuantize,
        activation_dtype: str = None,
        weight_dtype: str = None,
        bias: bool = True,
        out_dtype: Optional[str] = None,
        runtime: str = "cast",
    ):  # pylint: disable=too-many-arguments
        super().__init__(in_features, out_features, weight_config, bias, out_dtype)
        self.activation_dtype = activation_dtype
        self.weight_dtype = weight_dtype
        self.runtime = runtime
        self.weight_config = weight_config
        self.max_int_value = {
            key: self.get_max_int(dtype)
            for key, dtype in zip(["x", "w"], [self.activation_dtype, self.weight_dtype])
        }

        if "max" in self.runtime:
            self.q_calibration_scale = nn.Parameter((1,), weight_config.model_dtype)

    @staticmethod
    def get_max_int(dtype):
        if dtype == "e4m3_float8":
            return 448
        elif dtype == "e5m2_float8":
            return 57344
        elif dtype == "float16":
            return 65504
        else:
            raise NotImplementedError()

    def add_calibration_params(self, quant_map: QuantizeMapping, layer_name: str):
        scale_name = f"{layer_name}.q_calibration_scale"

        def alloc_scale():
            return nd.empty(
                shape=self.q_calibration_scale.shape, dtype=self.q_calibration_scale.dtype
            )

        quant_map.map_func[scale_name] = alloc_scale
        return quant_map

    @staticmethod
    def from_linear(
        src: nn.Linear,
        weight_config: ptq.PerTensorQuantize,
    ) -> "PTQLinearFP8":
        """
        Converts a non-quantized Linear to a per-tensor quantized FP8 Linear.

        Parameters
        ----------
        src : nn.Linear
            The non-quantized Linear layer

        weight_config : GroupQuantize
            The group quantization weight_config.

        Returns
        -------
        ret : PTQLinearFP8
            The per-tensor quantized Linear layer in FP8 precision.
        """
        out_features, in_features = src.weight.shape
        quantized_linear = PTQLinearFP8(
            in_features=in_features,
            out_features=out_features,
            weight_config=weight_config,
            activation_dtype=weight_config.activation_dtype,
            weight_dtype=weight_config.weight_dtype,
            bias=getattr(src, "bias", None) is not None,
            out_dtype=src.out_dtype,
            runtime="max" if "calibration" not in weight_config.name else "max-calibration",
        )

        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_linear.q_weight)
            apply_sharding(
                tp.ShardScalar(name=shard.name),
                f"{shard.name}_q_scale",
                quantized_linear.q_scale,
            )
            # TODO(csullivan) lm head is not sharded. calibration logic needs to change i think
            if "max" in quantized_linear.runtime:
                apply_sharding(
                    tp.ShardScalar(name=shard.name),
                    f"{shard.name}_q_calibration_scale",
                    quantized_linear.q_calibration_scale,
                )

        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        if self.runtime == "max-calibration":
            x, local_scale = nn.op.quantize(
                x,
                quantize_dtype=self.activation_dtype,
                kind="fp8-max",
                max_int_value=self.max_int_value["x"],
            )

            local_scale = nn.op.maximum_inplace(local_scale, self.q_calibration_scale)
            # Calibration done in fp16 mma so convert x and w back to fp16
            x = nn.op.astype(x, dtype="float16")

            if DataType(self.weight_dtype).type_code in [
                DataTypeCode.E4M3Float,
                DataTypeCode.E5M2Float,
            ]:
                dequant_func = self.config._dequantize_float8
                # The below computes w = cast(reinterpret_cast(self.q_weight, float8_e4m3), float16) * self.q_scale
                w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                    lambda weight, scale: dequant_func(  # pylint: disable=protected-access
                        weight,
                        scale,
                        out_shape=(self.out_features, self.in_features),
                    ),
                    name_hint="dequantize_weight",
                    args=[self.q_weight, self.q_scale],
                )
            elif self.weight_dtype == "float16":
                w = self.q_weight
        elif self.runtime == "max":
            local_scale = self.q_calibration_scale
            x = x / local_scale
            x = nn.op.astype(x, dtype=self.activation_dtype)
            w = self.q_weight
        elif self.runtime == "cast":
            x = nn.op.astype(x, dtype=self.activation_dtype)
            w = self.q_weight
        else:
            raise NotImplementedError(
                f"Only max and cast runtimes are supported for FP8 activations, {self.runtime} is unsupported."
            )
        w = nn.op.permute_dims(w)
        x = nn.op.matmul(x, w, out_dtype="float32")

        if self.runtime == "cast":
            total_scale = 1.0
        else:
            if self.runtime == "max" and self.weight_dtype == "e4m3_float8":
                total_scale = local_scale * self.q_scale
            else:
                # for calibration, q_scale is already used to dequantize the weights
                total_scale = local_scale
            total_scale = nn.op.astype(total_scale, dtype="float32")

        x *= total_scale

        x = nn.op.astype(x, self.weight_config.model_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x
