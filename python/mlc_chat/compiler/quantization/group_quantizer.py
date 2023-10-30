"""A group quantizer for on the fly parameter quantization"""
# pylint: disable=too-few-public-methods

from typing import List, Tuple

from tvm import te, tir

from .quantization import QuantizeConfig


def te_quantize(
    weight: te.Tensor, config: QuantizeConfig
) -> Tuple[te.Tensor, te.Tensor, List[te.Tensor]]:
    """Group quantization for weight tensor, defined in tensor expression."""
    # pylint: disable=too-many-locals
    assert len(weight.shape) == 2
    n, m = weight.shape  # pylint: disable=invalid-name
    # compute scale per group
    r = te.reduce_axis((0, config.group_size), name="r")  # pylint: disable=invalid-name
    num_group = tir.ceildiv(m, config.group_size)
    scale_shape = (n, num_group)
    max_abs = te.compute(
        shape=scale_shape,
        fcompute=lambda i, j: te.max(
            tir.if_then_else(
                j * config.group_size + r < weight.shape[1],
                te.abs(weight[i, j * config.group_size + r]),
                tir.const(1e-4, config.weight_dtype),
            ),
            axis=r,
        ),
        name="max_abs_value",
    )
    scale = te.compute(
        (n, m),
        lambda i, j: max_abs[i, j] / tir.const(config.max_int_value, dtype=config.weight_dtype),
        name="scale",
    )

    # compute scaled weight
    tir_max_int = tir.const(config.max_int_value, config.weight_dtype)
    tir_zero = tir.const(0, config.weight_dtype)
    tir_max_int_2 = tir.const(config.max_int_value * 2, config.weight_dtype)
    scaled_weight = te.compute(
        shape=weight.shape,
        fcompute=lambda i, j: tir.min(
            tir.max(
                tir.round(weight[i, j] / scale[i, j // config.group_size] + tir_max_int),
                tir_zero,
            ),
            tir_max_int_2,
        ).astype(config.storage_dtype),
    )

    # compute quantized weight per storage
    r = te.reduce_axis((0, config.num_elem_per_storage), name="r")  # pylint: disable=invalid-name
    num_storage = config.num_storage_per_group * num_group
    quantized_weight_shape = (n, num_storage)
    quantized_weight = te.compute(
        shape=quantized_weight_shape,
        fcompute=lambda i, j: tir.sum(
            scaled_weight[i, j * config.num_elem_per_storage + r]
            << (r * config.quantize_dtype_bits),
            axis=r,
            where=j * config.num_elem_per_storage + r < m,
        ),
        name="weight",
    )
    return quantized_weight, scale, [max_abs, scaled_weight]
    # pylint: enable=too-many-locals
