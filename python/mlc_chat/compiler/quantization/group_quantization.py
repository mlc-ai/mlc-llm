"""The group quantization config"""
from typing import Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from tvm.relax.frontend import nn
from tvm import DataType, te, tir, DataTypeCode, relax, dlight as dl, device
from tvm.target import Target
from ..parameter import QuantizeMapping
from tvm.runtime import NDArray


@dataclass
class GroupQuantizeConfig:
    group_size: int
    quantize_dtype: DataType  # "int3", "int4", "int8"
    storage_dtype: DataType  # "uint32"
    weight_dtype: DataType  # "float16", "float32"

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    max_int_value: int = 0

    def __post_init__(self):
        assert (
            isinstance(self.weight_dtype, DataType)
            and self.weight_dtype.type_code == DataTypeCode.FLOAT
        )
        assert (
            isinstance(self.quantize_dtype, DataType)
            and self.quantize_dtype.type_code == DataTypeCode.INT
        )
        assert (
            isinstance(self.storage_dtype, DataType)
            and self.storage_dtype.type_code == DataTypeCode.UINT
        )
        self.num_elem_per_storage = self.storage_dtype.bits // self.quantize_dtype.bits
        assert (
            self.num_elem_per_storage > 0
        ), "Storage unit should have more bits than single quantized elemtent"
        assert (
            self.group_size % self.num_elem_per_storage == 0
        ), "Group size should be divisible by numbers of elements per storage"
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage
        self.max_int_value = (1 << (self.quantize_dtype.bits - 1)) - 1

    def apply(self, mod: nn.Module, quant_map: QuantizeMapping, name_prefix: str) -> nn.Module:
        mutator = GroupQuantizeMutator(self, quant_map)
        mod = mutator.visit(name_prefix, mod)
        return mod

    def quantize(self, weight: NDArray) -> List[NDArray]:
        assert weight.dtype == str(self.weight_dtype)
        assert len(weight.shape) == 2
        bb = relax.BlockBuilder()
        weight_var = relax.Var("weight", relax.TensorStructInfo(weight.shape, self.weight_dtype))
        with bb.function(name="quantize", params=[weight_var]):
            with bb.dataflow():
                lv = bb.emit_te(self._quantize, weight_var)
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        mod = bb.get()
        with Target("cuda"):
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        ex = relax.build(mod, "cuda")
        dev = device("cuda", 0)
        vm = relax.VirtualMachine(ex, dev)
        return vm["quantize"](weight)

    def _quantize(self, weight: te.Tensor) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""
        assert len(weight.shape) == 2
        n, k = weight.shape
        # compute scale per group
        r = te.reduce_axis((0, self.group_size), name="r")
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
            lambda i, j: max_abs[i, j] / tir.const(self.max_int_value, str(self.weight_dtype)),
            name="scale",
        )

        # compute scaled weight
        tir_max_int = tir.const(self.max_int_value, str(self.weight_dtype))
        tir_zero = tir.const(0, str(self.weight_dtype))
        tir_max_int_2 = tir.const(self.max_int_value * 2, str(self.weight_dtype))
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
                scaled_weight[i, j * self.num_elem_per_storage + r]
                << (r * self.quantize_dtype.bits),
                axis=r,
                where=j * self.num_elem_per_storage + r < k,
            ),
            name="weight",
        )
        return quantized_weight, scale

    def _dequantize(
        self, weight: te.Tensor, scale: te.Tensor, out_shape: Optional[List[tir.PrimExpr]] = None
    ):
        tir_bin_mask = tir.const((1 << self.quantize_dtype.bits) - 1, str(self.storage_dtype))
        tir_max_int = tir.const(self.max_int_value, str(self.weight_dtype))
        dequantized_weight = te.compute(
            shape=[weight.shape[0], weight.shape[1] * self.num_elem_per_storage]
            if out_shape is None
            else out_shape,
            fcompute=lambda i, j: tir.multiply(
                tir.subtract(
                    tir.bitwise_and(
                        tir.shift_right(
                            weight[i, j // self.num_elem_per_storage],
                            (j % self.num_elem_per_storage) * self.storage_dtype.bits,
                        ),
                        tir_bin_mask,
                    ),
                    tir_max_int,
                ),
                scale[i, j // self.group_size],
            ),
        )
        return dequantized_weight


class GroupQuantizeLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: GroupQuantizeConfig,
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
            (
                out_features,
                n_group * config.num_elem_per_storage,
            ),
            config.storage_dtype,
        )
        self.scale = nn.Parameter((out_features, n_group), config.weight_dtype)
        if bias:
            self.bias = nn.Parameter((out_features,), config.weight_dtype)
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear, config: GroupQuantizeConfig):
        return GroupQuantizeLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            config=config,
            bias=linear.bias,
            out_dtype=linear.out_dtype,
        )

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(
                weight,
                scale,
                self.config,
                [tir.IntImm("int64", self.out_features), tir.IntImm("int64", self.in_features)],
            ),
            name_hint="group_dequantize",
            args=[self.weight, self.scale],
        )
        w = nn.op.permute_dims(w)
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x


class GroupQuantizeMutator(nn.Mutator):
    def __init__(self, config: GroupQuantizeConfig, quant_map: QuantizeMapping) -> None:
        super().__init__()
        self.config = config
        self.quant_map = quant_map

    def visit_module(self, name: str, node: nn.Module) -> Any:
        if isinstance(node, nn.Linear):
            weight_name = f"{name}.weight"
            self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
            self.quant_map.map_func[weight_name] = self.config.quantize
            return GroupQuantizeLinear.from_linear(node, self.config)
        return self.visit(name, node)
