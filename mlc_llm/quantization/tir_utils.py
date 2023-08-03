"""TIR computation utilities for quantization."""

import tvm
from tvm import tir

# fmt: off
def _tir_f32x2_to_bf16x2_to_u32(v0: tir.PrimExpr, v1: tir.PrimExpr, round_to_even: bool=True):
    mask = tir.const((1 << 16) - 1, "uint32")
    res = []
    for data in [v0, v1]:
        u32_val = tir.reinterpret("uint32", data)
        if round_to_even:
            rounding_bias = ((u32_val >> tir.const(16, "uint32")) & tir.const(1, "uint32")) + tir.const(0x7FFF, "uint32")
            u32_val += rounding_bias
        res.append((u32_val >> tir.const(16, "uint32")) & mask)
    return res[0] | (res[1] << tir.const(16, "uint32"))


def _tir_u32_to_bf16x2_to_f32x2(x: tir.PrimExpr):
    mask = tir.const((1 << 16) - 1, "uint32")
    x0 = x & mask
    x1 = (x >> 16) & mask
    return (tir.reinterpret("float32", x << tir.const(16, "uint32")) for x in [x0, x1])


def _tir_u32_to_int_to_float(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert val.dtype == "uint32"
    mask = tvm.tir.const((1 << nbit) - 1, "uint32")
    return tir.Cast(dtype, (val >> (pos * nbit).astype("uint32")) & mask)


def _tir_packed_uint_to_uint_to_float(storage_nbit: int):
    storage_dtype = "uint" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype
        max_int_value = (1 << (nbit - 1)) - 1
        return ((val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & tir.const((1 << nbit) - 1, "uint32")).astype(dtype) - tir.const(max_int_value, dtype)

    return f_convert


def _tir_packed_int_to_int_to_float(storage_nbit: int):
    storage_dtype = "int" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype
        mask = tir.const((1 << nbit) - 1, "int32")
        unextended = (val >> (pos.astype("int32") * tir.const(nbit, "int32"))) & mask
        return tir.Cast(dtype, (unextended << tir.const(32 - nbit, "int32")) >> tir.const(32 - nbit, "int32"))

    return f_convert


def _tir_f32_to_uint_to_f4(val: tir.PrimExpr):
    assert val.dtype == "float32"
    val_u32 = tir.reinterpret("uint32", val)
    # e_f32 >  120 -> e_f4 = min(e_f32 - 120 + M_h, 7)
    # e_f32 == 120 -> e_f4 = 1
    # e_f32 < 120 -> e_f4 = 0
    m_h = (val_u32 >> tir.const(22, "uint32")) & tir.const(1, "uint32")
    e_f32 = (val_u32 >> tir.const(23, "uint32")) & tir.const(255, "uint32")
    s = (val_u32 >> tir.const(31, "uint32"))
    e_f4 = tir.Select(e_f32 > tir.const(120, "uint32"), tir.Min(e_f32 - tir.const(120, "uint32") + m_h, tir.const(7, "uint32")), tir.Select(e_f32 == tir.const(120, "uint32"), tir.const(1, "uint32"), tir.const(0, "uint32")))
    return (s << tir.const(3, "uint32")) | e_f4


def _tir_f16_to_uint_to_f4(val: tir.PrimExpr):
    assert val.dtype == "float16"
    val_u32 = tir.Cast("uint32", tir.reinterpret("uint16", val))
    m_h = (val_u32 >> tir.const(9, "uint32")) & tir.const(1, "uint32")
    e_f16 = (val_u32 >> tir.const(10, "uint32")) & tir.const(31, "uint32")
    s = (val_u32 >> tir.const(15, "uint32"))
    e_f4 = tir.Select(e_f16 > tir.const(8, "uint32"), tir.Min(e_f16 - tir.const(8, "uint32") + m_h, tir.const(7, "uint32")), tir.Select(e_f16 == tir.const(8, "uint32"), tir.const(1, "uint32"), tir.const(0, "uint32")))
    return (s << tir.const(3, "uint32")) | e_f4


def _tir_u32_to_f4_to_f32(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == "float32"
    assert val.dtype == "uint32"
    # e_f4 == 0 -> e_f32 = 0
    # e_f4 != 0 -> e_f32 = e_f4 + 120 = e_f4 | (1111000)_2
    mask = tvm.tir.const((1 << nbit) - 1, "uint32")
    f4 = (val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & mask
    s = f4 >> tir.const(3, "uint32")
    e_f4 = f4 & tir.const(7, "uint32")
    e_f32 = e_f4 | tir.const(120, "uint32")
    val_f32 = tir.reinterpret("float32", (e_f32 | (s << tir.const(8, "uint32"))) << tir.const(23, "uint32"))
    return tir.Select(e_f4 == tir.const(0, "uint32"), tir.const(0, "float32"), val_f32)


def _tir_u32_to_f4_to_f16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == "float16"
    assert val.dtype == "uint32"
    # e_f4 == 0 -> e_f16 = 0
    # e_f4 != 0 -> e_f16 = e_f4 + 8 = e_f4 | (1000)_2
    mask = tvm.tir.const((1 << nbit) - 1, "uint32")
    f4 = (val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & mask
    s = f4 >> tir.const(3, "uint32")
    e_f4 = f4 & tir.const(7, "uint32")
    e_f16 = e_f4 | tir.const(8, "uint32")
    val_f16 = tir.reinterpret("float16", (e_f16 | (s << tir.const(5, "uint32"))) << tir.const(10, "uint32"))
    return tir.Select(e_f4 == tir.const(0, "uint32"), tir.const(0, "float16"), val_f16)
# fmt: on
