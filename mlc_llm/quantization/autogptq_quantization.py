from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple
from tvm import relax, te, tir, topi
from . import tir_utils
from .quantization import QuantizationSpec
from .quantization import FQuantize, FTEDequantize, convert_TE_func


@dataclass
class AutogptqQuantizationSpec(QuantizationSpec):
    """The quantization specification for group quantization algorithm."""

    mode: Literal["int2", "int3", "int4", "int8"]
    sym: bool
    group_size: int
    storage_nbit: int = 32

    quantized_suffix = ["qweight", "qzeros", "scales", "g_idx"]

    def get_loaded_tensor_info(
        self, pname: str, param_info: relax.TensorStructInfo
    ) -> Tuple[List[str], List[relax.TensorStructInfo]]:
        assert self.storage_nbit == 32, "Only support 32bit storage currently"

        quantized_pnames = self.quant_convert_pname_fwd(pname)
        if len(quantized_pnames) == 1:
            return quantized_pnames, [param_info]
        else:
            assert len(quantized_pnames) == 4
            assert param_info.ndim == 2
            nbit = int(self.mode[-1])
            tensor_info = []
            outfeatures, infeatures = param_info.shape.values
            group_size = self.group_size if self.group_size != -1 else infeatures

            def get_quantized_shape_dtype(quantized_pname: str):
                if quantized_pname.endswith("qweight"):
                    return (infeatures // self.storage_nbit * nbit, outfeatures), "uint32"
                elif quantized_pname.endswith("qzeros"):
                    return (
                        infeatures // group_size,
                        outfeatures // self.storage_nbit * nbit,
                    ), "uint32"
                elif quantized_pname.endswith("scales"):
                    return (infeatures // group_size, outfeatures), "float16"
                elif quantized_pname.endswith("g_idx"):
                    return (infeatures,), "uint32"
                else:
                    raise ValueError(f"Unrecognized quantized parameter name {quantized_pname}")

            for quantized_pname in quantized_pnames:
                shape, dtype = get_quantized_shape_dtype(quantized_pname)
                tensor_info.append(relax.TensorStructInfo(shape, dtype))

        return quantized_pnames, tensor_info

    def quant_convert_pname_fwd(self, torch_pname: str) -> List[str]:
        # For Llama:
        if "_proj.weight" in torch_pname:
            return [torch_pname.replace("weight", suffix) for suffix in self.quantized_suffix]
        return [torch_pname]

    def run_prequantize(self, model_path: str) -> str:
        # with auto-gptq >= 0.2.0
        try:
            import auto_gptq  # pylint: disable=import-outside-toplevel
            import transformers  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ImportError(
                "Please install auto_gptq package (version >= 0.2.0) and "
                "transformers package to use AutoGPTQ quantization."
            )
        import os
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quantized_model_path = (
            model_path
            + f"-gptq-i{self.mode[-1]}"
            + ("-sym" if self.sym else "")
            + f"-g{self.group_size}"
        )
        if os.path.isdir(quantized_model_path):
            return quantized_model_path

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        examples = [
            tokenizer(
                "MLC LLM is a universal solution that allows any language models "
                "to be deployed natively on a diverse set of hardware backends and "
                "native applications, plus a productive framework for everyone to "
                "further optimize model performance for their own use cases."
            )
        ]
        quantize_config = BaseQuantizeConfig(
            bits=int(self.mode[-1]),  # quantize bits
            desc_act=False,  # disable activation description
            group_size=self.group_size,  # disable group quantization
        )

        model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
        model.quantize(examples)

        # save quantized model
        model.save_quantized(quantized_model_path)
        tokenizer.save_pretrained(quantized_model_path)
        return quantized_model_path

    def get_quantize_func(self, param_info: relax.TensorStructInfo) -> Optional[FQuantize]:
        return None

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FQuantize]:
        return convert_TE_func(
            decoding_func(
                sym=self.sym,
                nbit=int(self.mode[-1]),
                storage_nbit=self.storage_nbit,
                dim_length=param_info.shape.values[-1],
                dtype=self.dtype,
            ),
            func_name="decode",
        )

    def convert_param_bkwd(self, torch_pname: str, torch_param):
        target_dtype = (
            self.dtype if "_proj." not in torch_pname or "scales" in torch_pname else "uint32"
        )

        # For Llama
        combined_layers = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
        if any([name in torch_pname for name in combined_layers]):
            return None
        return [(torch_pname, torch_param.astype(target_dtype))]

    def compute_relax_param(self, relax_pname: str, torch_params: List[Any]):
        import numpy as np

        # For Llama
        if "query_key_value_proj" in relax_pname:
            assert len(torch_params) == 3
        elif "gate_up_proj" in relax_pname:
            assert len(torch_params) == 2
        else:
            raise ValueError("Unexpected param loading")

        if "g_idx" in relax_pname:
            return torch_params[0].astype("uint32")
        else:
            target_dtype = self.dtype if "scales" in relax_pname else "uint32"
            return np.concatenate(torch_params, axis=-1).astype(target_dtype)


def decoding_func(
    sym: bool,
    nbit: int,
    storage_nbit: int,
    dim_length: tir.PrimExpr,
    dtype: str = "float16",
) -> FTEDequantize:
    assert dtype in ["float16"], "Only support float16 currently"
    assert sym == False, "Only support sym=False currently"
    assert storage_nbit == 32, "Only support storage_nbit=32 currently"

    def te_decode_asym(qweight, qzeros, scales, g_idx):
        n_float_per_u32 = 32 // nbit

        def f_decode_asym(i, j):
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
            scale_float, bias_float = scales[g_idx[i], j], zeros + 1
            w = (data_float - bias_float) * scale_float
            return w

        shape = (dim_length, qweight.shape[1])
        w = te.compute(shape=shape, fcompute=f_decode_asym, name="decode")
        w = topi.transpose(w)
        return w

    return te_decode_asym
