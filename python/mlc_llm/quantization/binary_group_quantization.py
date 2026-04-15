"""Binary group quantization: 1-bit weights with per-group FP16 scales.

Each weight is stored as a single bit (0 or 1) packed into uint32.
Dequantization: w = (2 * bit - 1) * scale
where each group of `group_size` weights shares one FP16 scale factor.

Designed for natively-trained 1-bit models such as PrismML Bonsai.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Literal, Optional, Tuple, Union

from tvm import IRModule, relax, te, tirx, topi
from tvm.relax.frontend import nn
from tvm.runtime import Tensor

from mlc_llm.loader import QuantizeMapping
from mlc_llm.support import logging

from .utils import (
    apply_sharding,
    compile_quantize_func,
    is_final_fc,
    is_moe_gate,
    pack_weight,
)

logger = logging.getLogger(__name__)


@dataclass
class BinaryGroupQuantize:
    """Configuration for binary (1-bit) group quantization."""

    name: str
    kind: str
    group_size: int
    storage_dtype: Literal["uint32"]
    model_dtype: Literal["float16", "float32"]
    linear_weight_layout: Literal["KN", "NK"]
    quantize_embedding: bool = True
    quantize_final_fc: bool = True

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    tensor_parallel_shards: int = 0

    def __post_init__(self):
        assert self.kind == "binary-group-quant"
        assert self.group_size % 32 == 0, "Group size must be divisible by 32"
        self.num_elem_per_storage = 32  # 32 single-bit weights per uint32
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage
        self.linear_quant_axis = 0 if self.linear_weight_layout == "KN" else 1
        self._quantize_func_cache = {}

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """Quantize model with binary group quantization."""

        class _Mutator(nn.Mutator):
            def __init__(self, config: "BinaryGroupQuantize", quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                if getattr(node, "no_quantization", False):
                    return node
                if (
                    isinstance(node, nn.Linear)
                    and (not is_final_fc(name) or self.config.quantize_final_fc)
                    and not is_moe_gate(name, node)
                ):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [
                        f"{name}.q_weight",
                        f"{name}.q_scale",
                    ]
                    self.quant_map.map_func[weight_name] = partial(
                        self.config.quantize_weight,
                        output_transpose=self.config.linear_weight_layout == "KN",
                    )
                    return BinaryGroupQuantizeLinear.from_linear(node, self.config)
                if isinstance(node, nn.Embedding) and self.config.quantize_embedding:
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [
                        f"{name}.q_weight",
                        f"{name}.q_scale",
                    ]
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return BinaryGroupQuantizeEmbedding.from_embedding(node, self.config)
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def _dequantize(
        self,
        weight: te.Tensor,
        scale: te.Tensor,
        axis: int,
        out_shape: Optional[List[tirx.PrimExpr]] = None,
    ):
        """Binary dequantization: extract 1-bit from uint32, apply per-group scale.

        For each weight: w = (2 * bit - 1) * scale
        where bit is 0 or 1 extracted from the packed uint32.
        """
        tir_one = tirx.const(1, self.storage_dtype)
        tir_two_f = tirx.const(2.0, self.model_dtype)
        tir_one_f = tirx.const(1.0, self.model_dtype)

        if out_shape is None:
            out_shape = list(weight.shape)
            out_shape[axis] *= self.num_elem_per_storage
        axis = axis if axis >= 0 else len(out_shape) + axis

        # The name="dequantize" is required for TIR GPU reduction schedules
        return te.compute(
            shape=out_shape,
            fcompute=lambda *idx: tirx.multiply(
                tirx.subtract(
                    tirx.multiply(
                        tirx.bitwise_and(
                            tirx.shift_right(
                                weight(
                                    *idx[:axis],
                                    idx[axis] // self.num_elem_per_storage,
                                    *idx[axis + 1 :],
                                ),
                                (idx[axis] % self.num_elem_per_storage).astype(self.storage_dtype),
                            ),
                            tir_one,
                        ).astype(self.model_dtype),
                        tir_two_f,
                    ),
                    tir_one_f,
                ),
                scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1 :]),
            ),
            name="dequantize",
        )

    def _quantize(
        self,
        weight: te.Tensor,
        axis: int = -1,
        output_transpose: bool = False,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Binary group quantization for weight tensor, defined in tensor expression.

        1. Compute per-group scale = max(|w|) in each group
        2. Binarize: bit = 1 if w >= 0 else 0
        3. Pack 32 bits into each uint32
        """
        shape = weight.shape
        axis = axis if axis >= 0 else len(shape) + axis
        k = shape[axis]
        # compute scale per group
        r = te.reduce_axis((0, self.group_size), name="r")
        num_group = tirx.ceildiv(k, self.group_size)
        scale_shape = (*shape[:axis], num_group, *shape[axis + 1 :])
        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda *idx: te.max(
                tirx.if_then_else(
                    idx[axis] * self.group_size + r < k,
                    te.abs(
                        weight(
                            *idx[:axis],
                            idx[axis] * self.group_size + r,
                            *idx[axis + 1 :],
                        )
                    ),
                    te.min_value(self.model_dtype),
                ),
                axis=r,
            ),
            name="max_abs_value",
        )
        scale = te.compute(
            scale_shape,
            lambda *idx: max_abs(*idx).astype(self.model_dtype),
            name="scale",
        )
        # binarize: 1 if w >= 0 else 0, cast to storage dtype for packing
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda *idx: tirx.if_then_else(
                weight(*idx) >= tirx.const(0, self.model_dtype),
                tirx.const(1, self.storage_dtype),
                tirx.const(0, self.storage_dtype),
            ),
        )
        # pack 32 bits into each uint32 using the standard pack_weight utility
        num_storage = self.num_storage_per_group * num_group
        quantized_weight_shape = (*shape[:axis], num_storage, *shape[axis + 1 :])
        quantized_weight = pack_weight(
            scaled_weight,
            axis=axis,
            num_elem_per_storage=self.num_elem_per_storage,
            weight_dtype="int1",
            storage_dtype=self.storage_dtype,
            out_shape=quantized_weight_shape,
        )
        if output_transpose:
            if len(quantized_weight.shape) != 2 or len(scale.shape) != 2:
                raise ValueError(
                    "Does not support transpose output quantized weight with ndim != 2"
                )
            quantized_weight = topi.transpose(quantized_weight)
            scale = topi.transpose(scale)
        return quantized_weight, scale

    def quantize_weight(
        self, weight: Tensor, axis: int = -1, output_transpose: bool = False
    ) -> List[Tensor]:
        """Quantize weight with binary group quantization.

        Parameters
        ----------
        weight : Tensor
            The original weight.
        axis : int
            The group axis.
        output_transpose : bool
            Whether to transpose the output quantized weight.

        Returns
        -------
        ret : List[Tensor]
            The list of [q_weight, q_scale] tensors.
        """
        device = weight.device
        device_type = device._DEVICE_TYPE_TO_NAME[  # pylint: disable=protected-access
            device.dlpack_device_type()
        ]
        axis = axis if axis >= 0 else len(weight.shape) + axis

        def _create_quantize_func() -> IRModule:
            bb = relax.BlockBuilder()
            weight_var = relax.Var("weight", relax.TensorStructInfo(weight.shape, weight.dtype))
            with bb.function(name="main", params=[weight_var]):
                with bb.dataflow():
                    lv = bb.emit_te(self._quantize, weight_var, axis, output_transpose)
                    gv = bb.emit_output(lv)
                bb.emit_func_output(gv)
            return bb.finalize()

        key = (
            f"({weight.shape}, {weight.dtype}, {device_type}, "
            f"axis={axis}, output_transpose={output_transpose})"
        )
        quantize_func = self._quantize_func_cache.get(key, None)
        if quantize_func is None:
            logger.info("Compiling quantize function for key: %s", key)
            quantize_func = compile_quantize_func(_create_quantize_func(), device=device)
            self._quantize_func_cache[key] = quantize_func
        return quantize_func(weight)


class BinaryGroupQuantizeLinear(nn.Module):
    """An nn.Linear module with binary group quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: Union[int, tirx.Var],
        config: BinaryGroupQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        num_group = tirx.ceildiv(in_features, config.group_size)
        num_shards = config.tensor_parallel_shards
        if num_shards > 1 and (in_features * num_shards // config.group_size) % num_shards != 0:
            raise ValueError(
                f"The linear dimension {in_features * num_shards} has "
                f"{in_features * num_shards // config.group_size} groups under group size "
                f"{config.group_size}. The groups cannot be evenly distributed on "
                f"{num_shards} GPUs.\n"
                "Possible solutions: reduce number of GPUs, or use quantization with smaller "
                "group size."
            )
        if config.linear_weight_layout == "KN":
            self.q_weight = nn.Parameter(
                (config.num_storage_per_group * num_group, out_features),
                config.storage_dtype,
            )
            self.q_scale = nn.Parameter((num_group, out_features), config.model_dtype)
        else:
            self.q_weight = nn.Parameter(
                (out_features, config.num_storage_per_group * num_group),
                config.storage_dtype,
            )
            self.q_scale = nn.Parameter((out_features, num_group), config.model_dtype)
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @staticmethod
    def from_linear(src: nn.Linear, config: BinaryGroupQuantize) -> "BinaryGroupQuantizeLinear":
        """Convert a non-quantized nn.Linear to BinaryGroupQuantizeLinear."""
        out_features, in_features = src.weight.shape
        quantized_linear = BinaryGroupQuantizeLinear(
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
            apply_sharding(shard, f"{shard.name}_q_scale", quantized_linear.q_scale)
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        """Forward method for binary group quantized linear layer."""
        # Use unique name_hint to avoid runtime scope tracking conflicts
        # while still starting with "dequantize" for compiler pass detection
        unique_name = f"dequantize_{id(self)}"
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(
                weight,
                scale,
                axis=self.config.linear_quant_axis,
                out_shape=(
                    [
                        (
                            tirx.IntImm("int64", self.out_features)
                            if isinstance(self.out_features, int)
                            else weight.shape[0]
                        ),
                        tirx.IntImm("int64", self.in_features),
                    ]
                    if self.config.linear_weight_layout == "NK"
                    else [
                        tirx.IntImm("int64", self.in_features),
                        (
                            tirx.IntImm("int64", self.out_features)
                            if isinstance(self.out_features, int)
                            else weight.shape[1]
                        ),
                    ]
                ),
            ),
            name_hint=unique_name,
            args=[self.q_weight, self.q_scale],
        )
        if self.config.linear_weight_layout == "NK":
            w = nn.op.permute_dims(w)
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        """Override to() to avoid converting bias when out_dtype is set."""
        self.q_weight.to(dtype=dtype)
        self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype


class BinaryGroupQuantizeEmbedding(nn.Module):
    """An nn.Embedding module with binary group quantization."""

    def __init__(self, num: Union[int, tirx.Var], dim: int, config: BinaryGroupQuantize):
        self.num = num
        self.dim = dim
        self.config = config
        num_group = tirx.ceildiv(dim, config.group_size)
        self.q_weight = nn.Parameter(
            (num, config.num_storage_per_group * num_group), config.storage_dtype
        )
        self.q_scale = nn.Parameter((num, num_group), config.model_dtype)

    @staticmethod
    def from_embedding(
        embedding: nn.Embedding, config: BinaryGroupQuantize
    ) -> "BinaryGroupQuantizeEmbedding":
        """Convert a non-quantized nn.Embedding to BinaryGroupQuantizeEmbedding."""
        num, dim = embedding.weight.shape
        return BinaryGroupQuantizeEmbedding(num, dim, config)

    def forward(self, x: nn.Tensor):
        """Forward method for binary group quantized embedding layer."""
        # Use unique name_hint to avoid runtime scope tracking conflicts
        unique_name = f"dequantize_{id(self)}"
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(
                weight,
                scale,
                axis=-1,
                out_shape=[
                    (
                        tirx.IntImm("int64", self.num)
                        if isinstance(self.num, int)
                        else weight.shape[0]
                    ),
                    tirx.IntImm("int64", self.dim),
                ],
            ),
            name_hint=unique_name,
            args=[self.q_weight, self.q_scale],
        )
        if x.ndim == 1:
            return nn.op.take(w, x, axis=0)
        return nn.op.reshape(
            nn.op.take(w, nn.op.reshape(x, shape=[-1]), axis=0),
            shape=[*x.shape, self.dim],
        )

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which dequantizes the weight and multiplies with input."""
        # Use unique name_hint to avoid runtime scope tracking conflicts
        unique_name = f"dequantize_{id(self)}"
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(
                weight,
                scale,
                axis=-1,
                out_shape=[
                    (
                        tirx.IntImm("int64", self.num)
                        if isinstance(self.num, int)
                        else weight.shape[0]
                    ),
                    tirx.IntImm("int64", self.dim),
                ],
            ),
            name_hint=unique_name,
            args=[self.q_weight, self.q_scale],
        )
        w = nn.op.permute_dims(w)
        return nn.op.matmul(x, w, out_dtype="float32")
