"""The group quantization config"""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from tvm import DataType, DataTypeCode, IRModule
from tvm import dlight as dl
from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.target import Target

from mlc_chat.loader import QuantizeMapping
from mlc_chat.nn import MixtralExperts
from mlc_chat.support import logging
from mlc_chat.support import tensor_parallel as tp

from .utils import convert_uint_to_float

logger = logging.getLogger(__name__)


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
        self._quantize_func_cache = {}

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with group quantization

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
            def __init__(self, config: GroupQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """
                The visiting method for group quantization of nn.Module nodes.

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
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return GroupQuantizeLinear.from_linear(node, self.config)
                if isinstance(node, nn.Embedding):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return GroupQuantizeEmbedding.from_embedding(node, self.config)
                if isinstance(node, MixtralExperts):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return GroupQuantizeMixtralExperts.from_mixtral_experts(node, self.config)
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
        tir_max_int = tir.const(self.max_int_value, self.model_dtype)
        float_weight = convert_uint_to_float(
            weight,
            DataType(self.quantize_dtype).bits,
            self.num_elem_per_storage,
            self.storage_dtype,
            self.model_dtype,
            out_shape,
        )
        return te.compute(
            shape=[*weight.shape[:-1], weight.shape[-1] * self.num_elem_per_storage]
            if out_shape is None
            else out_shape,
            fcompute=lambda *idx: tir.multiply(
                tir.subtract(
                    float_weight[idx[:-1] + (idx[-1],)],
                    tir_max_int,
                ),
                scale[idx[:-1] + (idx[-1] // self.group_size,)],
            ),
            name="dequantize",
        )

    def quantize_weight(self, weight: NDArray) -> List[NDArray]:
        """
        Quantize weight with group quantization

        Parameters
        ----------
        weight : NDArray
            The original weight.

        Returns
        ------
        ret: List[NDArray]
            The list of group quantized weights.
        """
        device = weight.device
        device_type = device.MASK2STR[device.device_type]

        def _create_quantize_func() -> IRModule:
            bb = relax.BlockBuilder()  # pylint: disable=invalid-name
            weight_var = relax.Var("weight", relax.TensorStructInfo(weight.shape, weight.dtype))
            with bb.function(name="main", params=[weight_var]):
                with bb.dataflow():
                    lv = bb.emit_te(self._quantize, weight_var)  # pylint: disable=invalid-name
                    gv = bb.emit_output(lv)  # pylint: disable=invalid-name
                bb.emit_func_output(gv)
            return bb.finalize()

        def _compile_quantize_func(mod: IRModule) -> Callable:
            if device_type in ["cuda", "rocm", "metal", "vulkan"]:
                target = Target.current()
                if target is None:
                    target = Target.from_device(device)
                with target:
                    mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    )(mod)
            elif device_type == "cpu":
                target = "llvm"
                mod = relax.transform.LegalizeOps()(mod)
            else:
                raise NotImplementedError(f"Device type {device_type} is not supported")
            ex = relax.build(mod, target=target)
            vm = relax.VirtualMachine(ex, device)  # pylint: disable=invalid-name
            return vm["main"]

        key = str(
            (
                *(int(weight.shape[i]) for i in range(len(weight.shape) - 1)),
                int(weight.shape[-1]),
                weight.dtype,
                device_type,
            )
        )
        quantize_func = self._quantize_func_cache.get(key, None)
        if quantize_func is None:
            logger.info("Compiling quantize function for key: %s", key)
            quantize_func = _compile_quantize_func(_create_quantize_func())
            self._quantize_func_cache[key] = quantize_func
        return quantize_func(weight)

    def _quantize(  # pylint: disable=too-many-locals
        self,
        weight: te.Tensor,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""
        max_int = tir.const(self.max_int_value, self.model_dtype)
        shape = weight.shape  # pylint: disable=invalid-name
        k = shape[-1]
        quantize_dtype = DataType(self.quantize_dtype)
        # compute scale per group
        r = te.reduce_axis((0, self.group_size), name="r")  # pylint: disable=invalid-name
        num_group = tir.ceildiv(k, self.group_size)
        scale_shape = (*shape[:-1], num_group)
        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda *idx: te.max(
                tir.if_then_else(
                    idx[-1] * self.group_size + r < k,
                    te.abs(weight[idx[:-1] + (idx[-1] * self.group_size + r,)]),
                    te.min_value(self.model_dtype),
                ),
                axis=r,
            ),
            name="max_abs_value",
        )
        scale = te.compute(
            scale_shape,
            lambda *idx: max_abs[idx[:-1] + (idx[-1],)].astype(self.model_dtype) / max_int,
            name="scale",
        )
        # compute scaled weight
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda *idx: tir.min(
                tir.max(
                    tir.round(
                        weight[idx[:-1] + (idx[-1],)]
                        / scale[idx[:-1] + (idx[-1] // self.group_size,)]
                        + max_int
                    ),
                    tir.const(0, self.model_dtype),
                ),
                max_int * 2,
            ).astype(self.storage_dtype),
        )
        # compute quantized weight per storage
        r = te.reduce_axis((0, self.num_elem_per_storage), name="r")  # pylint: disable=invalid-name
        num_storage = self.num_storage_per_group * num_group
        quantized_weight_shape = (*shape[:-1], num_storage)
        quantized_weight = te.compute(
            shape=quantized_weight_shape,
            fcompute=lambda *idx: tir.sum(
                tir.if_then_else(
                    idx[-1] * self.num_elem_per_storage + r < k,
                    scaled_weight[idx[:-1] + (idx[-1] * self.num_elem_per_storage + r,)]
                    << (r * quantize_dtype.bits),
                    0,
                ),
                axis=r,
            ),
            name="weight",
        )
        return quantized_weight, scale


class GroupQuantizeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
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
        num_group = tir.ceildiv(in_features, config.group_size)
        self.q_weight = nn.Parameter(
            (out_features, config.num_storage_per_group * num_group), config.storage_dtype
        )
        self.q_scale = nn.Parameter((out_features, num_group), config.model_dtype)
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @staticmethod
    def from_linear(src: nn.Linear, config: GroupQuantize) -> "GroupQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a group quantized GroupQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeLinear
            The group quantized GroupQuantizeLinear layer.
        """
        quantized_linear = GroupQuantizeLinear(
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
            bias=getattr(src, "bias", None) is not None,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            _apply_sharding(shard, f"{shard.name}_q_weight", quantized_linear.q_weight)
            _apply_sharding(shard, f"{shard.name}_q_scale", quantized_linear.q_scale)
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
        w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                [tir.IntImm("int64", self.out_features), tir.IntImm("int64", self.in_features)],
            ),
            name_hint="dequantize",
            args=[self.q_weight, self.q_scale],
        )
        w = nn.op.permute_dims(w)  # pylint: disable=invalid-name
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
        self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


class GroupQuantizeEmbedding(nn.Module):
    """An nn.Embedding module with group quantization"""

    def __init__(self, num: int, dim: int, config: GroupQuantize):
        self.num = num
        self.dim = dim
        self.config = config
        num_group = tir.ceildiv(dim, config.group_size)
        self.q_weight = nn.Parameter(
            (num, config.num_storage_per_group * num_group), config.storage_dtype
        )
        self.q_scale = nn.Parameter((num, num_group), config.model_dtype)

    @staticmethod
    def from_embedding(embedding: nn.Embedding, config: GroupQuantize) -> "GroupQuantizeEmbedding":
        """
        Converts a non-quantized nn.Embedding to a group quantized GroupQuantizeEmbedding

        Parameters
        ----------
        linear : nn.Embedding
            The non-quantized nn.Embedding.

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeEmbedding
            The group quantized GroupQuantizeEmbedding layer.
        """
        num, dim = embedding.weight.shape
        return GroupQuantizeEmbedding(num, dim, config)

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
        w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                [tir.IntImm("int64", self.num), tir.IntImm("int64", self.dim)],
            ),
            name_hint="dequantize",
            args=[self.q_weight, self.q_scale],
        )
        if x.ndim == 1:
            return nn.op.take(w, x, axis=0)
        return nn.op.reshape(
            nn.op.take(w, nn.op.reshape(x, shape=[-1]), axis=0),
            shape=[*x.shape, self.dim],
        )


class GroupQuantizeMixtralExperts(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An MixtralExperts module with group quantization"""

    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        config: GroupQuantize,
    ):  # pylint: disable=too-many-arguments
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        num_group = tir.ceildiv(in_features, config.group_size)
        self.q_weight = nn.Parameter(
            (num_local_experts, out_features, config.num_storage_per_group * num_group),
            config.storage_dtype,
        )
        self.q_scale = nn.Parameter(
            (num_local_experts, out_features, num_group), config.model_dtype
        )
        self.quantize_dtype = config.quantize_dtype
        self.group_size = config.group_size
        self.dtype = config.model_dtype

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts", config: GroupQuantize
    ) -> "GroupQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a group quantized GroupQuantizeMixtralExperts

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeMixtralExperts
            The group quantized GroupQuantizeMixtralExperts layer.
        """
        quantized_mistral_experts = GroupQuantizeMixtralExperts(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
        )
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            _apply_sharding(shard, f"{shard.name}_q_weight", quantized_mistral_experts.q_weight)
            _apply_sharding(shard, f"{shard.name}_q_scale", quantized_mistral_experts.q_scale)
        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """Forward method for group quantized mistral experts.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        indptr: nn.Tensor
            The indptr tensor

        single_batch_decode: bool
            Whether to use single-batch decode

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the group quantized mistral experts layer.
        """
        from mlc_chat.op import moe_matmul  # pylint: disable=import-outside-toplevel

        assert x.ndim == 2
        if indptr.ndim == 2:  # single-batch
            assert indptr.shape[0] == 1
            return moe_matmul.dequantize_gemv(
                x,
                self.q_weight,
                self.q_scale,
                indptr,
                quantize_dtype=self.quantize_dtype,
                group_size=self.group_size,
            )
        assert indptr.ndim == 1
        return moe_matmul.dequantize_group_gemm(
            x,
            self.q_weight,
            self.q_scale,
            indptr,
            quantize_dtype=self.quantize_dtype,
            group_size=self.group_size,
        )


def _apply_sharding(shard, name: str, weight: nn.Parameter):
    if isinstance(shard, tp.ShardSingleDim):
        weight.attrs["shard_strategy"] = tp.ShardSingleDim(
            name=name,
            dim=shard.dim,
            segs=shard.segs,
        )
    else:
        raise NotImplementedError(f"Unknowing sharding strategy: {shard}")
