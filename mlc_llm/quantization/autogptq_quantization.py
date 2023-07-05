from dataclasses import dataclass
from typing import List, Literal, Optional, Dict
import tvm
import numpy as np
from tvm._ffi.runtime_ctypes import Device
from tvm import relax, te, tir, topi
from tvm.script import tir as T

from . import tir_utils
from .quantization import QuantizationSpec
from .quantization import FQuantize, FDequantize
from tvm.relax.testing import nn
from transformers import AutoModelForCausalLM


def load_autogptq_params(
    model_path: str,
    param_list: List[relax.Var],
    pidx2pname: Dict[int, str],
    device: Device,
    excluded_params: List[str] = ["cos_cached", "sin_cached"],
) -> List[relax.Var]:
    try:
        import auto_gptq  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "Please install auto_gptq package to use AutoGPTQ quantization."
        )

    from auto_gptq import AutoGPTQForCausalLM

    model = AutoGPTQForCausalLM.from_quantized(model_path).cpu()
    param_dict = model.state_dict()
    for pidx, pname in pidx2pname.items():
        if any(excluded_param in pname for excluded_param in excluded_params):
            continue

        np_array = param_dict[pname].numpy()
        if np_array.dtype == np.int32:
            param_list[pidx] = tvm.nd.array(np_array.astype(np.uint32), device,)
        else:
            param_list[pidx] = tvm.nd.array(np_array, device)

    return param_list


@dataclass
class AutogptqQuantizationSpec(QuantizationSpec):
    """The quantization specification for group quantization algorithm."""

    mode: Literal["int3", "int4"]
    sym: bool
    storage_nbit: int
    group_size: int
    transpose: bool
    pre_quantized: bool = True

    _quantized_params = ["qweight", "qzeros", "scales", "g_idx"]

    def get_quantized_params(self, name: str, param: relax.Var) -> bool:
        quantized_params = {}
        for quantized_param in self._quantized_params:
            quantized_name = name.replace("weight", quantized_param)
            quantized_param = self.convert_param(quantized_name, param)
            quantized_params[quantized_name] = quantized_param
        return quantized_params

    def get_bits(self):
        return int(self.mode[-1])

    def convert_param(self, name: str, param: relax.Var) -> relax.Var:
        assert self.storage_nbit == 32, "Only support 32bit storage currently"
        assert param.struct_info.ndim == 2, "Only support 2D param currently"
        # assert self.transpose == False, "Only support transpose=False currently"

        # by default, torch stores weight in [outfeatures, infeatures]
        outfeatures, infeatures = param.struct_info.shape
        group_size = self.group_size if self.group_size != -1 else infeatures
        self.bits = self.get_bits()
        if "qweight" in name:
            _shape = (infeatures // self.storage_nbit * self.bits, outfeatures) if self.transpose else (outfeatures, infeatures // self.storage_nbit * self.bits)
            _dtype = "uint32"
        elif "qzeros" in name:
            _shape = (
                infeatures // group_size,
                outfeatures // self.storage_nbit * self.bits,
            ) if self.transpose else (outfeatures // self.storage_nbit * self.bits, infeatures // group_size)
            _dtype = "uint32"
        elif "scales" in name:
            _shape = (infeatures // group_size, outfeatures) if self.transpose else (outfeatures, infeatures // group_size)
            _dtype = "float16"
        elif "g_idx" in name:
            _shape = (infeatures,)
            _dtype = "uint32"
        else:
            raise ValueError(f"Unknown quantized param name {name}")
        print(f"Convert {name} to shape {_shape} and dtype {_dtype} from {param.struct_info.shape}")
        # print("raw param shape: ", param.struct_info.shape)
        new_param = relax.Var(name, relax.TensorStructInfo(_shape, _dtype))
        return new_param

    def get_quantize_func(
        self, param_info: relax.TensorStructInfo
    ) -> Optional[FDequantize]:
        return None

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FDequantize]:
        infeatures = param_info.shape.struct_info # type: ignore
        return decoding_func(
            sym=self.sym,
            group_size=self.group_size if self.group_size != -1 else infeatures,
            nbit=int(self.mode[-1]),
            mode=self.mode,
            storage_nbit=self.storage_nbit,
            dim_length=param_info.shape.values[-1],
            data_transposed=self.transpose,
            transpose_output=self.transpose,
            dtype=self.dtype,
        )


def decoding_func(
    sym: bool,
    group_size: int,
    nbit: int,
    mode: str,
    storage_nbit: int,
    dim_length: tir.PrimExpr,
    data_transposed: bool = True,
    transpose_output: bool = False,
    dtype: str = "float16",
) -> FDequantize:
    assert dtype in ["float16"], "Only support float16 currently"
    assert sym == False, "Only support sym=False currently"
    assert storage_nbit == 32, "Only support storage_nbit=32 currently"

    def te_decode_asym(qweight, qzeros, scales, g_idx):
        n_float_per_u32 = 32 // nbit
        def f_decode_asym(i, j):
            if data_transposed:
                zeros = tir_utils._tir_u32_to_int_to_float(
                    nbit,
                    qzeros[g_idx[i], j // n_float_per_u32],
                    j % n_float_per_u32,
                    dtype=dtype,
                )
                data_float = tir_utils._tir_u32_to_int_to_float(
                    nbit,
                    qweight[i // n_float_per_u32, j],
                    i % n_float_per_u32,
                    dtype=dtype,
                )
                scale_float, bias_float = scales[g_idx[i], j], zeros
            else:
                zeros = tir_utils._tir_u32_to_int_to_float(
                    nbit,
                    qzeros[i // n_float_per_u32, g_idx[j]],
                    i % n_float_per_u32,
                    dtype=dtype,
                )
                data_float = tir_utils._tir_u32_to_int_to_float(
                    nbit,
                    qweight[i, j // n_float_per_u32],
                    j % n_float_per_u32,
                    dtype=dtype,
                )
                scale_float, bias_float = scales[i, g_idx[j]], zeros
            w = data_float * scale_float + bias_float
            return w

        shape = (
            (dim_length, qweight.shape[1])
            if data_transposed
            else (qweight.shape[0], dim_length)
        )
        w = te.compute(shape=shape, fcompute=f_decode_asym, name="decode")
        if transpose_output:
            w = topi.transpose(w)
        return w

    return te_decode_asym


# fmt: on
