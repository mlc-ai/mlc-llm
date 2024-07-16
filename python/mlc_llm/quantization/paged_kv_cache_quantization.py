"""Paged KV cache quantization config"""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments,too-many-locals
import enum
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from tvm import DataType, tir
from tvm.script import tir as T
from tvm.target import Target


class PagedKVCacheQuantization(enum.IntEnum):
    """The quantization scheme to apply to Paged KV cache.
    If it is none, quantization will not be applied to kv cache.
    Otherwise, different quantization schemes can be applied to kv cache.
    """

    KV_NO_QUANT = 0
    KV_GROUP_QUANT_INT_3 = 1
    KV_GROUP_QUANT_INT_4 = 2


@dataclass
class BaseKVConfig:  # pylint: disable=too-many-instance-attributes
    """Base configuration for key-value cache"""

    name: str
    kind: str
    head_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    model_dtype: DataType
    target: Target


@dataclass
class NoQuantizeKV(BaseKVConfig):
    """Configuration for key-value non-quantization"""

    num_storage: int = 0
    kv_storage_dtype: DataType = None

    def __post_init__(self):
        assert self.kind == "no-quant"
        assert str(self.model_dtype) in ["float16", "float32"]

        self.num_storage = self.head_dim
        self.kv_storage_dtype = self.model_dtype


@dataclass
class GroupQuantizeKV(BaseKVConfig):  # pylint: disable=too-many-instance-attributes
    """Configuration for key-value group quantization"""

    group_size: int
    kv_quantize_dtype: DataType

    max_int_value: int = 0
    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    num_groups: int = 0
    num_storage_weight: int = 0
    num_storage_scale: int = 0
    num_storage: int = 0
    kv_storage_dtype: DataType = None

    def __post_init__(self):
        assert self.kind == "group-quant"
        assert str(self.kv_quantize_dtype) in ["int3", "int4"]
        assert str(self.model_dtype) in ["float16", "float32"]

        self.kv_storage_dtype = {
            "float16": DataType("uint16"),
            "float32": DataType("uint32"),
        }[str(self.model_dtype)]

        self.max_int_value = (2 ** (self.kv_quantize_dtype.bits - 1)) - 1
        self.num_elem_per_storage = self.kv_storage_dtype.bits // self.kv_quantize_dtype.bits
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage
        self.num_groups = math.ceil(self.head_dim / self.group_size)
        self.num_storage_weight = self.num_storage_per_group * self.num_groups
        self.num_storage_scale = self.num_groups
        self.num_storage = self.num_storage_weight + self.num_storage_scale

        if self.kv_storage_dtype.bits < self.kv_quantize_dtype.bits:
            raise ValueError("Storage unit should be greater or equal to quantized element")
        if self.group_size % self.num_elem_per_storage != 0:
            raise ValueError("Group size should be divisible by numbers of elements per storage")

    def kv_cache_quantize_transpose_append(self):
        """
        Return the TIR function that appends new k/v data to PagedKVCache
        (fused w/ kv quantization).
        """

        # pylint: disable=line-too-long,invalid-name
        # fmt: off
        @T.prim_func
        def tir_kv_cache_transpose_append(
            var_pages: T.handle,
            var_k_data: T.handle,
            var_v_data: T.handle,
            var_position_map: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
            num_pages = T.int64()
            position_map_elem_offset = T.int32()
            pages = T.match_buffer(var_pages, (num_pages, 2, self.num_key_value_heads, 16, self.num_storage), self.kv_storage_dtype)
            k_data = T.match_buffer(var_k_data, (ntoken, self.num_key_value_heads, self.head_dim), self.model_dtype)
            v_data = T.match_buffer(var_v_data, (ntoken, self.num_key_value_heads, self.head_dim), self.model_dtype)
            position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)

            k_max_abs_value = T.alloc_buffer((ntoken, self.num_key_value_heads, self.num_groups), self.model_dtype)
            v_max_abs_value = T.alloc_buffer((ntoken, self.num_key_value_heads, self.num_groups), self.model_dtype)
            k_scale = T.alloc_buffer((ntoken, self.num_key_value_heads, self.num_groups), self.model_dtype)
            v_scale = T.alloc_buffer((ntoken, self.num_key_value_heads, self.num_groups), self.model_dtype)
            k_compute = T.alloc_buffer((ntoken, self.num_key_value_heads, self.num_storage_weight), self.kv_storage_dtype)
            v_compute = T.alloc_buffer((ntoken, self.num_key_value_heads, self.num_storage_weight), self.kv_storage_dtype)

            for i0, i1, i2, r in T.grid(ntoken, T.int64(self.num_key_value_heads), T.int64(self.num_groups), T.int64(self.group_size)):
                with T.block("k_max_abs_value"):
                    v_i0, v_i1, v_i2, v_r = T.axis.remap("SSSR", [i0, i1, i2, r])
                    T.reads(k_data[v_i0, v_i1, v_i2 * self.group_size + v_r])
                    T.writes(k_max_abs_value[v_i0, v_i1, v_i2])
                    with T.init():
                        k_max_abs_value[v_i0, v_i1, v_i2] = T.min_value(self.model_dtype)
                    k_max_abs_value[v_i0, v_i1, v_i2] = T.max(
                        k_max_abs_value[v_i0, v_i1, v_i2],
                        T.if_then_else(
                            v_i2 * self.group_size + v_r < self.head_dim,
                            T.fabs(k_data[v_i0, v_i1, v_i2 * self.group_size + v_r]),
                            T.min_value(self.model_dtype),
                        ),
                    )
            for i0, i1, i2, r in T.grid(ntoken, T.int64(self.num_key_value_heads), T.int64(self.num_groups), T.int64(self.group_size)):
                with T.block("v_max_abs_value"):
                    v_i0, v_i1, v_i2, v_r = T.axis.remap("SSSR", [i0, i1, i2, r])
                    T.reads(v_data[v_i0, v_i1, v_i2 * self.group_size + v_r])
                    T.writes(v_max_abs_value[v_i0, v_i1, v_i2])
                    with T.init():
                        v_max_abs_value[v_i0, v_i1, v_i2] = T.min_value(self.model_dtype)
                    v_max_abs_value[v_i0, v_i1, v_i2] = T.max(
                        v_max_abs_value[v_i0, v_i1, v_i2],
                        T.if_then_else(
                            v_i2 * self.group_size + v_r < self.head_dim,
                            T.fabs(v_data[v_i0, v_i1, v_i2 * self.group_size + v_r]),
                            T.min_value(self.model_dtype),
                        ),
                    )

            for i0, i1, i2 in T.grid(ntoken, T.int64(self.num_key_value_heads), T.int64(self.num_groups)):
                with T.block("scale"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(k_max_abs_value[v_i0, v_i1, v_i2], v_max_abs_value[v_i0, v_i1, v_i2])
                    T.writes(k_scale[v_i0, v_i1, v_i2], v_scale[v_i0, v_i1, v_i2])
                    k_scale[v_i0, v_i1, v_i2] = k_max_abs_value[v_i0, v_i1, v_i2] / self.max_int_value
                    v_scale[v_i0, v_i1, v_i2] = v_max_abs_value[v_i0, v_i1, v_i2] / self.max_int_value

            for i0, i1, i2, r in T.grid(ntoken, T.int64(self.num_key_value_heads), T.int64(self.num_storage_weight), T.int64(self.num_elem_per_storage)):
                with T.block("k_compute_pack"):
                    v_i0, v_i1, v_i2, v_r = T.axis.remap("SSSR", [i0, i1, i2, r])
                    T.reads(
                        k_data[v_i0, v_i1, v_i2 * self.num_elem_per_storage + v_r],
                        k_scale[v_i0, v_i1, v_i2 // self.num_storage_per_group],
                    )
                    T.writes(k_compute[v_i0, v_i1, v_i2])
                    with T.init():
                        k_compute[v_i0, v_i1, v_i2] = 0
                    k_compute[v_i0, v_i1, v_i2] = k_compute[v_i0, v_i1, v_i2] + T.if_then_else(
                        v_i2 * self.num_elem_per_storage + v_r < self.head_dim,
                        T.shift_left(
                            T.Cast(
                                self.kv_storage_dtype,
                                T.min(
                                    T.max(
                                        T.round(
                                            k_data[v_i0, v_i1, v_i2 * self.num_elem_per_storage + v_r]
                                            / k_scale[v_i0, v_i1, v_i2 // self.num_storage_per_group]
                                            + self.max_int_value
                                        ),
                                        0.0,
                                    ),
                                    self.max_int_value * 2.0,
                                ),
                            ),
                            T.Cast(self.kv_storage_dtype, v_r * self.kv_quantize_dtype.bits),
                        ),
                        tir.const(0, str(self.kv_storage_dtype)),
                    )
            for i0, i1, i2, r in T.grid(ntoken, T.int64(self.num_key_value_heads), T.int64(self.num_storage_weight), T.int64(self.num_elem_per_storage)):
                with T.block("v_compute_pack"):
                    v_i0, v_i1, v_i2, v_r = T.axis.remap("SSSR", [i0, i1, i2, r])
                    T.reads(
                        v_data[v_i0, v_i1, v_i2 * self.num_elem_per_storage + v_r],
                        v_scale[v_i0, v_i1, v_i2 // self.num_storage_per_group],
                    )
                    T.writes(v_compute[v_i0, v_i1, v_i2])
                    with T.init():
                        v_compute[v_i0, v_i1, v_i2] = 0
                    v_compute[v_i0, v_i1, v_i2] = v_compute[v_i0, v_i1, v_i2] + T.if_then_else(
                        v_i2 * self.num_elem_per_storage + v_r < self.head_dim,
                        T.shift_left(
                            T.Cast(
                                self.kv_storage_dtype,
                                T.min(
                                    T.max(
                                        T.round(
                                            v_data[v_i0, v_i1, v_i2 * self.num_elem_per_storage + v_r]
                                            / v_scale[v_i0, v_i1, v_i2 // self.num_storage_per_group]
                                            + self.max_int_value
                                        ),
                                        0.0,
                                    ),
                                    self.max_int_value * 2.0,
                                ),
                            ),
                            T.Cast(self.kv_storage_dtype, v_r * self.kv_quantize_dtype.bits),
                        ),
                        tir.const(0, str(self.kv_storage_dtype)),
                    )

            for global_pos, h, f in T.grid(ntoken, T.int64(self.num_key_value_heads), T.int64(self.num_storage)):
                if position_map[global_pos] != T.int32(-1):
                    with T.block("transpose_append"):
                        vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                        T.reads(
                            position_map[vgpos],
                            k_compute[vgpos, vh, 0:self.num_storage_weight],
                            v_compute[vgpos, vh, 0:self.num_storage_weight],
                            k_scale[vgpos, vh, 0:self.num_storage_scale],
                            v_scale[vgpos, vh, 0:self.num_storage_scale],
                        )
                        T.writes(pages[position_map[vgpos] // 16, 0:2, vh, position_map[vgpos] % 16, vf])
                        position: T.int32 = position_map[vgpos]  # type: ignore

                        if vf < self.num_storage_weight:
                            pages[T.floordiv(position, 16), 0, vh, T.floormod(position, 16), vf] = k_compute[vgpos, vh, vf]
                            pages[T.floordiv(position, 16), 1, vh, T.floormod(position, 16), vf] = v_compute[vgpos, vh, vf]
                        else:
                            pages[T.floordiv(position, 16), 0, vh, T.floormod(position, 16), vf] = T.reinterpret(
                                self.kv_storage_dtype, k_scale[vgpos, vh, vf - self.num_storage_weight]
                            )
                            pages[T.floordiv(position, 16), 1, vh, T.floormod(position, 16), vf] = T.reinterpret(
                                self.kv_storage_dtype, v_scale[vgpos, vh, vf - self.num_storage_weight]
                            )

        # fmt: on
        # pylint: enable=line-too-long,invalid-name

        return tir_kv_cache_transpose_append

    def kv_cache_dequantize(
        self,
        buffer: T.Buffer,
        indices: Tuple[tir.Var, ...],
    ):
        """TIR dequantizae kv"""

        d = indices[-1]
        bin_mask = (1 << self.kv_quantize_dtype.bits) - 1

        quantized_data = T.Cast(
            self.model_dtype,
            T.bitwise_and(
                T.shift_right(
                    buffer[indices[:-1] + (d // self.num_elem_per_storage,)],
                    T.Cast("int32", (d % self.num_elem_per_storage) * self.kv_quantize_dtype.bits),
                ),
                bin_mask,
            ),
        )
        data = (
            quantized_data - tir.const(self.max_int_value, str(self.model_dtype))
        ) * T.reinterpret(
            self.model_dtype,
            buffer[indices[:-1] + (self.num_storage_weight + (d // self.group_size),)],
        )
        return data


def get_kv_storage_dtype(kv_quant_scheme: str, model_dtype: str) -> DataType:
    """Get Cache storage dtype according to quantization scheme"""
    return {
        "kv_no_quant": DataType(model_dtype),
        "kv_group_quant_int_3": DataType("int3"),
        "kv_group_quant_int_4": DataType("int4"),
    }[kv_quant_scheme]


def get_paged_kv_cache_config(
    kv_quant_scheme: str,
    model_dtype: str,
    kwargs: Dict[str, Any],
) -> BaseKVConfig:
    """Get Paged KV Cache configuration"""
    return {
        "kv_no_quant": NoQuantizeKV(
            name="kv_no_quant",
            kind="no-quant",
            model_dtype=DataType(model_dtype),
            **kwargs,
        ),
        "kv_group_quant_int_3": GroupQuantizeKV(
            name="kv_group_quant_int_3",
            kind="group-quant",
            group_size=40,
            kv_quantize_dtype=get_kv_storage_dtype("kv_group_quant_int_3", model_dtype),
            model_dtype=DataType(model_dtype),
            **kwargs,
        ),
        "kv_group_quant_int_4": GroupQuantizeKV(
            name="kv_group_quant_int_4",
            kind="group-quant",
            group_size=32,
            kv_quantize_dtype=get_kv_storage_dtype("kv_group_quant_int_4", model_dtype),
            model_dtype=DataType(model_dtype),
            **kwargs,
        ),
    }[kv_quant_scheme]
