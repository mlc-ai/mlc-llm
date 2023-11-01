"""The group quantization config"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from tvm import DataType, DataTypeCode, device
from tvm import dlight as dl
from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.target import Target

from ..parameter import QuantizeMapping


@dataclass
class GroupQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for group quantization"""

    name: str
    kind: str
    group_size: int
    quantize_dtype: str  # "int3", "int4", "int8"
    storage_dtype: str  # "uint32"
    model_dtype: str  # "float16", "float32"

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    max_int_value: int = 0

    def __post_init__(self):
        assert self.kind == "group-quant"
        quantize_dtype = DataType(self.quantize_dtype)
        storage_dtype = DataType(self.storage_dtype)
        model_dtype = DataType(self.model_dtype)
        assert quantize_dtype.type_code == DataTypeCode.INT
        assert storage_dtype.type_code == DataTypeCode.UINT
        assert model_dtype.type_code == DataTypeCode.FLOAT
        if storage_dtype.bits < quantize_dtype.bits:
            raise ValueError("Storage unit should be greater or equal to quantized element")

        self.num_elem_per_storage = storage_dtype.bits // quantize_dtype.bits
        if self.group_size % self.num_elem_per_storage != 0:
            raise ValueError("Group size should be divisible by numbers of elements per storage")
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage
        self.max_int_value = (2 ** (quantize_dtype.bits - 1)) - 1

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """Quantize model with group quantization"""

        class _Mutator(nn.Mutator):
            def __init__(self, config: GroupQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                if isinstance(node, nn.Linear):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    # self.quant_map.map_func[weight_name] = self.config.quantize
                    return GroupQuantizeLinear.from_linear(node, self.config)
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def _dequantize(
        self,
        weight: te.Tensor,
        scale: te.Tensor,
        out_shape: Optional[List[tir.PrimExpr]] = None,
    ):
        quantize_dtype = DataType(self.quantize_dtype)
        storage_dtype = DataType(self.storage_dtype)
        tir_bin_mask = tir.const((2**quantize_dtype.bits) - 1, self.storage_dtype)
        tir_max_int = tir.const(self.max_int_value, self.model_dtype)
        dequantized_weight = te.compute(
            shape=[weight.shape[0], weight.shape[1] * self.num_elem_per_storage]
            if out_shape is None
            else out_shape,
            fcompute=lambda i, j: tir.multiply(
                tir.subtract(
                    tir.bitwise_and(
                        tir.shift_right(
                            weight[i, j // self.num_elem_per_storage],
                            (j % self.num_elem_per_storage) * storage_dtype.bits,
                        ),
                        tir_bin_mask,
                    ),
                    tir_max_int,
                ),
                scale[i, j // self.group_size],
            ),
        )
        return dequantized_weight

    def quantize_weight(self, weight: NDArray) -> List[NDArray]:
        """Quantize weight with group quantization"""
        assert weight.dtype == self.model_dtype
        assert len(weight.shape) == 2
        bb = relax.BlockBuilder()
        weight_var = relax.Var("weight", relax.TensorStructInfo(weight.shape, self.model_dtype))
        with bb.function(name="quantize", params=[weight_var]):
            with bb.dataflow():
                lv = bb.emit_te(self._quantize, weight_var)
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        mod = bb.get()
        with Target("cuda"):
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback()
            )(mod)
        ex = relax.build(mod, "cuda")
        dev = device("cuda", 0)
        vm = relax.VirtualMachine(ex, dev)
        return vm["quantize"](weight)

    def _quantize(  # pylint: disable=too-many-locals
        self, weight: te.Tensor
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""
        assert len(weight.shape) == 2
        n, k = weight.shape  # pylint: disable=invalid-name
        quantize_dtype = DataType(self.quantize_dtype)
        # compute scale per group
        r = te.reduce_axis((0, self.group_size), name="r")  # pylint: disable=invalid-name
        num_group = tir.ceildiv(k, self.group_size)
        scale_shape = (n, num_group)
        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda i, j: te.max(
                te.abs(weight[i, j * self.group_size + r]),
                where=j * self.group_size + r < k,
                axis=r,
            ),
            name="max_abs_value",
        )
        scale = te.compute(
            scale_shape,
            lambda i, j: max_abs[i, j] / tir.const(self.max_int_value, self.model_dtype),
            name="scale",
        )

        # compute scaled weight
        tir_max_int = tir.const(self.max_int_value, self.model_dtype)
        tir_zero = tir.const(0, self.model_dtype)
        tir_max_int_2 = tir.const(self.max_int_value * 2, self.model_dtype)
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda i, j: tir.min(
                tir.max(
                    tir.round(weight[i, j] / scale[i, j // self.group_size] + tir_max_int),
                    tir_zero,
                ),
                tir_max_int_2,
            ).astype(self.storage_dtype),
        )

        # compute quantized weight per storage
        r = te.reduce_axis((0, self.num_elem_per_storage), name="r")
        num_storage = self.num_storage_per_group * num_group
        quantized_weight_shape = (n, num_storage)
        quantized_weight = te.compute(
            shape=quantized_weight_shape,
            fcompute=lambda i, j: tir.sum(
                scaled_weight[i, j * self.num_elem_per_storage + r] << (r * quantize_dtype.bits),
                axis=r,
                where=j * self.num_elem_per_storage + r < k,
            ),
            name="weight",
        )
        return quantized_weight, scale


class GroupQuantizeLinear(nn.Module):
    """An nn.Linear module with group quantization"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        config: GroupQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        n_group = tir.ceildiv(in_features, config.group_size)
        self.weight = nn.Parameter(
            (out_features, n_group * config.num_elem_per_storage),
            config.storage_dtype,
        )
        self.scale = nn.Parameter((out_features, n_group), config.model_dtype)
        if bias:
            self.bias = nn.Parameter((out_features,), config.model_dtype)
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear, config: GroupQuantize):
        """Converts a non-quantized nn.Linear to a quantized GroupQuantizeLinear"""
        return GroupQuantizeLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            config=config,
            bias=getattr(linear, "bias", None) is not None,
            out_dtype=linear.out_dtype,
        )

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name,missing-docstring
        w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                [
                    tir.IntImm("int64", self.out_features),
                    tir.IntImm("int64", self.in_features),
                ],
            ),
            name_hint="decode",
            args=[self.weight, self.scale],
        )
        w = nn.op.permute_dims(w)  # pylint: disable=invalid-name
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x
