"""The block-scale quantization config"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import tvm
from tvm import DataType, DataTypeCode, te, tir
from tvm.relax.frontend import nn
from tvm.script import tir as T

from mlc_llm.loader import QuantizeMapping
from mlc_llm.nn import MixtralExperts
from mlc_llm.op import cutlass, extern, moe_matmul, triton
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp

from .utils import apply_sharding, is_final_fc, is_moe_gate

logger = logging.getLogger(__name__)


@dataclass
class BlockScaleQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for block-scale quantization"""

    name: str
    kind: str = "block-scale"
    weight_dtype: Literal["float8_e4m3fn", "float8_e5m2"] = "float8_e4m3fn"
    model_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    quantize_linear: bool = True
    weight_block_size: Optional[Tuple[int, int]] = None
    use_activation_scale: bool = False

    def __post_init__(self):
        assert self.kind == "block-scale-quant"
        weight_dtype = DataType(self.weight_dtype)
        model_dtype = DataType(self.model_dtype)
        assert weight_dtype.type_code in [
            DataTypeCode.Float8E4M3FN,
            DataTypeCode.Float8E5M2,
        ]
        assert model_dtype.type_code in [
            DataTypeCode.FLOAT,
            DataTypeCode.BFLOAT,
        ]

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """Quantize model with block-scale quantization

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

        weight_block_size = model.weight_block_size

        class _Mutator(nn.Mutator):
            def __init__(self, config: BlockScaleQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """The visiting method for block-scale quantization of nn.Module nodes.

                Parameters
                ----------
                name : str
                    The name of the current node.

                node : nn.Module
                    The current node of nn.Module to mutate.

                Returns
                ------
                ret : Any
                """
                if getattr(node, "no_quantization", False):
                    return node

                if hasattr(node, "w_uk"):
                    assert hasattr(node, "w_uv")
                    assert node.block_size == weight_block_size
                    if (
                        node.qk_nope_head_dim % node.block_size[0] != 0
                        or node.v_head_dim % node.block_size[1] != 0
                    ):
                        raise ValueError(
                            "Invalid DeepSeek model config: "
                            "qk_nope_head_dim must be multiple of weight_block_size[0], and "
                            "v_head_dim must be multiple of weight_block_size[1]. "
                            f"However, qk_nope_head_dim is {node.qk_nope_head_dim}, "
                            f"v_head_dim is {node.v_head_dim}, "
                            f"weight_block_size is {node.block_size}."
                        )
                    w_uk_shard_strategy = node.w_uk.attrs.get("shard_strategy", None)
                    w_uv_shard_strategy = node.w_uv.attrs.get("shard_strategy", None)
                    node.w_uk = nn.Parameter(
                        (node.num_heads, node.kv_lora_rank, node.qk_nope_head_dim),
                        self.config.weight_dtype,
                    )
                    node.w_uv = nn.Parameter(
                        (node.num_heads, node.v_head_dim, node.kv_lora_rank),
                        self.config.weight_dtype,
                    )
                    node.w_uk_scale_inv = nn.Parameter(
                        (
                            node.num_heads,
                            node.kv_lora_rank // node.block_size[1],
                            node.qk_nope_head_dim // node.block_size[0],
                        ),
                        "float32",
                    )
                    node.w_uv_scale_inv = nn.Parameter(
                        (
                            node.num_heads,
                            node.v_head_dim // node.block_size[0],
                            node.kv_lora_rank // node.block_size[1],
                        ),
                        "float32",
                    )
                    if w_uk_shard_strategy is not None:
                        assert w_uk_shard_strategy.segs is None
                        apply_sharding(w_uk_shard_strategy, w_uk_shard_strategy.name, node.w_uk)
                        apply_sharding(
                            w_uk_shard_strategy,
                            f"{w_uk_shard_strategy.name}_scale_inv",
                            node.w_uk_scale_inv,
                        )
                    if w_uv_shard_strategy is not None:
                        assert w_uv_shard_strategy.segs is None
                        apply_sharding(w_uv_shard_strategy, w_uv_shard_strategy.name, node.w_uv)
                        apply_sharding(
                            w_uv_shard_strategy,
                            f"{w_uv_shard_strategy.name}_scale_inv",
                            node.w_uv_scale_inv,
                        )

                if (
                    isinstance(node, nn.Linear)
                    and not is_final_fc(name)
                    and not is_moe_gate(name, node)
                ):
                    if self.config.use_activation_scale:
                        return BlockScaleQuantizeLinearStaticActivation.from_linear(
                            node, self.config, weight_block_size
                        )
                    return BlockScaleQuantizeLinear.from_linear(
                        node, self.config, weight_block_size
                    )
                if isinstance(node, MixtralExperts):
                    return BlockScaleQuantizeMixtralExperts.from_mixtral_experts(
                        node, self.config, weight_block_size
                    )
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        self.weight_block_size = weight_block_size
        return model


class BlockScaleQuantizeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Block-scale quantization for Linear"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        weight_dtype: Literal["float8_e4m3fn", "float8_e5m2"],
        block_size: Tuple[int, int],
        bias: bool = True,
        dtype: Optional[str] = None,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.weight = nn.Parameter((out_features, in_features), weight_dtype)
        self.weight_scale_inv = nn.Parameter(
            (
                (out_features + block_size[0] - 1) // block_size[0],
                (in_features + block_size[1] - 1) // block_size[1],
            ),
            "float32",
        )
        self.weight_dtype = weight_dtype
        self.block_size = block_size
        if bias:
            self.bias = nn.Parameter((out_features,), dtype if out_dtype is None else out_dtype)
        else:
            self.bias = None

    @staticmethod
    def from_linear(
        src: nn.Linear, config: BlockScaleQuantize, weight_block_size: Optional[Tuple[int, int]]
    ) -> "BlockScaleQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a block-scale quantized BlockScaleQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : BlockScaleQuantize
            The block-scale quantization config.

        weight_block_size : Optional[Tuple[int, int]]
            The weight block size.

        Returns
        -------
        ret : BlockScaleQuantizeLinear
            The block-scale quantized BlockScaleQuantizeLinear.
        """
        assert weight_block_size is not None
        out_features, in_features = src.weight.shape
        quantized_linear = BlockScaleQuantizeLinear(
            in_features=in_features,
            out_features=out_features,
            weight_dtype=config.weight_dtype,
            block_size=weight_block_size,
            bias=getattr(src, "bias", None) is not None,
            dtype=config.model_dtype,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, shard.name, quantized_linear.weight)
            if isinstance(shard, tp.ShardSingleDim) and shard.segs is not None:
                shard.segs = [x // weight_block_size[shard.dim] for x in shard.segs]
            apply_sharding(shard, f"{shard.name}_scale_inv", quantized_linear.weight_scale_inv)
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        """Forward pass of the block-scale quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor.
        """
        m = 1
        for i in range(x.ndim - 1):
            m *= x.shape[i]
        if m == 1:
            x_shape = x.shape
            return dequantize_float8_groupwise_scaled_gemv(
                x.reshape(1, x.shape[-1]),
                self.weight,
                self.weight_scale_inv,
                self.block_size,
                self.out_dtype if self.out_dtype is not None else x.dtype,
            ).reshape(*x_shape[:-1], -1)

        shape_supported_by_cutlass = (  # pylint: disable=unused-variable
            self.weight.shape[0] % 128 == 0 and self.weight.shape[1] % 128 == 0
        )
        # Todo: check "shape supported by cutlass" for Hopper  # pylint: disable=fixme
        if (
            extern.get_store().cutlass_gemm
            and tvm.get_global_func(
                "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn", allow_missing=True
            )
            is not None
        ):
            x_fp8, x_scale = rowwise_group_quant_fp8(
                x, self.block_size[1], self.weight_dtype, transpose_scale=True
            )
            x = cutlass.fp8_groupwise_scaled_gemm(
                x_fp8,
                x_scale,
                self.weight,
                self.weight_scale_inv,
                self.block_size,
                self.out_dtype if self.out_dtype is not None else x.dtype,
            )
        else:
            x_fp8, x_scale = rowwise_group_quant_fp8(
                x, self.block_size[1], self.weight_dtype, transpose_scale=False
            )
            x = triton.fp8_groupwise_scaled_gemm(
                x_fp8,
                x_scale,
                self.weight,
                self.weight_scale_inv,
                self.block_size,
                self.out_dtype if self.out_dtype is not None else x.dtype,
            )
        if self.bias is not None:
            x = x + self.bias
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


class BlockScaleQuantizeLinearStaticActivation(BlockScaleQuantizeLinear):
    """Block-scale quantization for static activation FP8."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        weight_dtype: Literal["float8_e4m3fn", "float8_e5m2"],
        block_size: Tuple[int, int],
        bias: bool = True,
        dtype: Optional[str] = None,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            weight_dtype=weight_dtype,
            block_size=block_size,
            bias=bias,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        num_in_groups = (in_features + block_size[1] - 1) // block_size[1]
        self.activation_scale = nn.Parameter((num_in_groups,), "float32")

    @staticmethod
    def from_linear(
        src: nn.Linear, config: BlockScaleQuantize, weight_block_size: Optional[Tuple[int, int]]
    ) -> "BlockScaleQuantizeLinearStaticActivation":
        """
        Convert a non-quantized nn.Linear to a block-scale quantized BlockScaleQuantizeLinearStaticActivation.
        
        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : BlockScaleQuantize
            The block-scale quantization config.
        
        weight_block_size : Optional[Tuple[int, int]]
            The weight block size.
            
        Returns
        -------
        ret : BlockScaleQuantizeLinearStaticActivation
            The block-scale quantized BlockScaleQuantizeLinearStaticActivation
        """
        assert weight_block_size is not None
        out_features, in_features = src.weight.shape
        quantized_linear = BlockScaleQuantizeLinearStaticActivation(
            in_features=in_features,
            out_features=out_features,
            weight_dtype=config.weight_dtype,
            block_size=weight_block_size,
            bias=getattr(src, "bias", None) is not None,
            dtype=config.model_dtype,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, shard.name, quantized_linear.weight)
            if isinstance(shard, tp.ShardSingleDim) and shard.segs is not None:
                shard.segs = [x // weight_block_size[shard.dim] for x in shard.segs]
            apply_sharding(shard, f"{shard.name}_scale_inv", quantized_linear.weight_scale_inv)
            apply_sharding(shard, f"{shard.name}_activation_scale", quantized_linear.activation_scale)
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        x_fp8 = static_activation_group_quant_fp8(
            x,
            self.activation_scale,
            self.block_size[1],
            self.weight_dtype,
        )
        x_scale_triton = broadcast_activation_scale(
            x,
            self.activation_scale,
            self.block_size[1],
            transpose=False,
        )
        shape_supported_by_cutlass = (
            self.weight.shape[0] % 128 == 0 and self.weight.shape[1] % 128 == 0
        )
        if (
            extern.get_store().cutlass_gemm
            and shape_supported_by_cutlass
            and tvm.get_global_func(
                "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn", allow_missing=True
            )
            is not None
        ):
            x_scale = broadcast_activation_scale(
                x,
                self.activation_scale,
                self.block_size[1],
                transpose=True,
            )
            out = cutlass.fp8_groupwise_scaled_gemm(
                x_fp8,
                x_scale,
                self.weight,
                self.weight_scale_inv,
                self.block_size,
                self.out_dtype if self.out_dtype is not None else x.dtype,
            )
        else:
            out = triton.fp8_groupwise_scaled_gemm(
                x_fp8,
                x_scale_triton,
                self.weight,
                self.weight_scale_inv,
                self.block_size,
                self.out_dtype if self.out_dtype is not None else x.dtype,
            )
        if self.bias is not None:
            out = out + self.bias
        return out


class BlockScaleQuantizeMixtralExperts(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Block-scale quantization for MoE experts"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        weight_dtype: Literal["float8_e4m3fn", "float8_e5m2"],
        block_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((num_local_experts, out_features, in_features), weight_dtype)
        self.weight_scale_inv = nn.Parameter(
            (
                num_local_experts,
                (out_features + block_size[0] - 1) // block_size[0],
                (in_features + block_size[1] - 1) // block_size[1],
            ),
            "float32",
        )
        self.weight_dtype = weight_dtype
        self.block_size = block_size

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts",
        config: BlockScaleQuantize,
        weight_block_size: Optional[Tuple[int, int]],
    ) -> "BlockScaleQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a block-scale
        quantized BlockScaleQuantizeMixtralExperts

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        config : BlockScaleQuantize
            The block-scale quantization config.

        weight_block_size : Optional[Tuple[int, int]]
            The weight block size.

        Returns
        -------
        ret : BlockScaleQuantizeMixtralExperts
            The block-scale quantized BlockScaleQuantizeMixtralExperts layer.
        """
        assert weight_block_size is not None
        quantized_mistral_experts = BlockScaleQuantizeMixtralExperts(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            weight_dtype=config.weight_dtype,
            block_size=weight_block_size,
        )
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, shard.name, quantized_mistral_experts.weight)
            if isinstance(shard, tp.ShardSingleDim) and shard.segs is not None:
                shard.segs = [x // weight_block_size[shard.dim - 1] for x in shard.segs]
            apply_sharding(
                shard, f"{shard.name}_scale_inv", quantized_mistral_experts.weight_scale_inv
            )
        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:
        """Forward pass of the block-scale quantized MixtralExperts.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        indptr : nn.Tensor
            The indptr tensor of group gemm, with shape of [num_experts + 1,].

        Returns
        -------
        ret : nn.Tensor
            The output tensor.
        """
        if indptr.ndim == 2:
            # The input is for single token, which does not need group gemm
            # and can be specialized.
            expert_indices = indptr
            assert expert_indices.shape[0] == 1
            return moe_matmul.dequantize_block_scale_float8_gemv(
                x,
                self.weight,
                self.weight_scale_inv,
                expert_indices,
                self.block_size,
                x.dtype,
            )

        x_fp8, x_scale = rowwise_group_quant_fp8(
            x, self.block_size[1], self.weight_dtype, transpose_scale=False
        )
        if (
            extern.get_store().cutlass_gemm
            and tvm.get_global_func(
                "cutlass.groupwise_scaled_group_gemm_e4m3fn_e4m3fn", allow_missing=True
            )
            is not None
        ):
            x = cutlass.fp8_groupwise_scaled_group_gemm(
                x_fp8,
                x_scale,
                self.weight,
                self.weight_scale_inv,
                indptr,
                self.block_size,
                x.dtype,
            )
        else:
            x = triton.fp8_groupwise_scaled_group_gemm(
                x_fp8,
                x_scale,
                self.weight,
                self.weight_scale_inv,
                indptr,
                self.block_size,
                x.dtype,
            )
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


def rowwise_group_quant_fp8(  # pylint: disable=too-many-arguments
    x: nn.Tensor,
    group_size: int,
    dtype: Literal["float8_e4m3fn", "float8_e5m2"],
    transpose_scale: bool,
    eps: float = 1e-10,
    keep_first_batch_dim: bool = False,
) -> Tuple[nn.Tensor, nn.Tensor]:
    """Rowwise group quantization of fp8 tensor.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor.

    group_size : int
        The group size per row for quantization.

    transpose_scale : bool
        Whether return the transposed scales or not.

    Returns
    -------
    x_fp8 : nn.Tensor
        The quantized tensor.

    x_scale : nn.Tensor
        The scales of the quantized tensor.
        If transpose_scale is True, the shape is
        (*x.shape[:-2], ceildiv(x.shape[-1], group_size), x.shape[-2]).
        Otherwise, the shape is (*x.shape[:-1], ceildiv(x.shape[-1], group_size)).
    """
    assert x.ndim >= 2
    assert group_size > 0

    def quantize(x: te.Tensor):
        num_group = tir.ceildiv(x.shape[-1], group_size)
        max_abs_shape = (*x.shape[:-1], num_group)
        max_abs_reduce_axis = te.reduce_axis((0, group_size), name="r")
        scale_dtype = "float32"
        max_abs = te.compute(
            shape=max_abs_shape,
            fcompute=lambda *idx: te.max(
                tir.if_then_else(
                    idx[-1] * group_size + max_abs_reduce_axis < x.shape[-1],
                    tir.Max(
                        te.abs(
                            x(*idx[:-1], idx[-1] * group_size + max_abs_reduce_axis).astype(
                                scale_dtype
                            )
                        ),
                        eps,
                    ),
                    tir.min_value(scale_dtype),
                ),
                axis=max_abs_reduce_axis,
            ),
            name="max_abs",
        )
        assert dtype in ["float8_e4m3fn", "float8_e5m2"]
        fp8_max = 448.0 if dtype == "float8_e4m3fn" else 57344.0
        fp8_min = -fp8_max
        scale = te.compute(
            shape=max_abs_shape,
            fcompute=lambda *idx: max_abs(*idx) / tir.const(fp8_max, scale_dtype),
            name="scale",
        )
        x_quantized = te.compute(
            shape=x.shape,
            fcompute=lambda *idx: tir.max(
                tir.min(
                    x(*idx).astype(scale_dtype) / scale(*idx[:-1], idx[-1] // group_size), fp8_max
                ),
                fp8_min,
            ).astype(dtype),
            name="x_quantized",
        )
        if transpose_scale:
            if not keep_first_batch_dim:
                scale = te.compute(
                    shape=(num_group, *x.shape[:-1]),
                    fcompute=lambda *idx: scale(*idx[1:], idx[0]),
                    name="scale",
                )
            else:
                assert len(x.shape) > 2
                scale = te.compute(
                    shape=(x.shape[0], num_group, *x.shape[1:-1]),
                    fcompute=lambda *idx: scale(idx[0], *idx[2:], idx[1]),
                    name="scale",
                )
        return x_quantized, scale

    x_quantized, scale = nn.tensor_expr_op(quantize, name_hint="rowwise_group_quant_fp8", args=[x])
    return x_quantized, scale


def static_activation_group_quant_fp8(
    x: nn.Tensor,
    activation_scale: nn.Tensor,
    group_size: int,
    dtype: Literal["float8_e4m3fn", "float8_e5m2"],
) -> nn.Tensor:
    """Quantize activations with a pre-computed scale."""

    assert activation_scale.ndim == 1

    def quantize(x: te.Tensor, scale: te.Tensor):
        fp8_max = 448.0 if dtype == "float8_e4m3fn" else 57344.0
        fp8_min = -fp8_max

        def fcompute(*idx):
            group_idx = tir.indexdiv(idx[-1], group_size)
            return tir.max(
                tir.min(
                    x(*idx).astype("float32") / scale(group_idx),
                    fp8_max,
                ),
                fp8_min,
            ).astype(dtype)

        return te.compute(shape=x.shape, fcompute=fcompute, name="static_activation_group_fp8")

    quantized = nn.tensor_expr_op(
        quantize,
        name_hint="static_activation_group_fp8",
        args=[x, activation_scale],
    )
    return quantized


def broadcast_activation_scale(
    x: nn.Tensor,
    activation_scale: nn.Tensor,
    group_size: int,
    transpose: bool,
) -> nn.Tensor:
    """Broadcast stored activation scales."""

    reshape_shape = (1,) * (x.ndim - 1) + (activation_scale.shape[0],)
    scale = nn.op.reshape(activation_scale, reshape_shape)
    scale = nn.op.broadcast_to(scale, (*x.shape[:-1], activation_scale.shape[0]))
    if transpose:
        axes = list(range(scale.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        scale = nn.op.permute_dims(scale, axes=axes)
    return scale


def dequantize_float8_groupwise_scaled_gemv(
    x: nn.Tensor,
    w: nn.Tensor,
    w_scale: nn.Tensor,
    block_size: Tuple[int, int],
    out_dtype: str,
) -> nn.Tensor:
    """GEMV for FP8 groupwise scaled quantization.

    Parameters
    ----------
    x : Tensor
        The input tensor of shape (k,)

    w : Tensor
        The quantized weight tensor of shape (n, k)

    w_scale : Tensor
        The scale tensor of shape
        (n // block_size[0], k // block_size[1])

    block_size : Tuple[int, int]
        The block size of the weight tensor.

    out_dtype : str
        The output dtype of the GEMV computation.
    """
    assert x.ndim == 2
    assert w.ndim == 2
    assert w_scale.ndim == 2
    assert x.shape[0] == 1
    assert x.shape[1] == w.shape[1]
    _, k = x.shape
    n, _ = w.shape
    model_dtype = x.dtype
    quantize_dtype = w.dtype

    assert (n + block_size[0] - 1) // block_size[0] == w_scale.shape[0]
    assert (k + block_size[1] - 1) // block_size[1] == w_scale.shape[1]

    def _dequantize(w, s, i, j):
        return w[i, j].astype(model_dtype) * s[i // block_size[0], j // block_size[1]].astype(
            model_dtype
        )

    @T.prim_func(private=True)
    def _func(
        x: T.Buffer((1, k), model_dtype),  # type: ignore
        w: T.Buffer((n, k), quantize_dtype),  # type: ignore
        w_scale: T.Buffer(  # type: ignore
            ((n + block_size[0] - 1) // block_size[0], (k + block_size[1] - 1) // block_size[1]),
            "float32",
        ),
        o: T.Buffer((n,), out_dtype),  # type: ignore
    ):
        T.func_attr({"op_pattern": 4, "tir.noalias": True})  # kOutEWiseFusable
        y = T.alloc_buffer((n, k), model_dtype)
        for i1, i2 in T.grid(n, k):
            with T.block("dequantize"):
                i, j = T.axis.remap("SS", [i1, i2])
                y[i, j] = _dequantize(w, w_scale, i, j)
        for i1, i2 in T.grid(n, k):
            with T.block("gemv"):
                i, j = T.axis.remap("SR", [i1, i2])
                with T.init():
                    o[i] = T.cast(T.float16(0), out_dtype)
                o[i] += (x[0, j] * y[i, j]).astype(out_dtype)

    return nn.op.tensor_ir_op(
        _func,
        "moe_dequantize_gemv",
        args=[x, w, w_scale],
        out=nn.Tensor.placeholder([n], out_dtype),
    )
