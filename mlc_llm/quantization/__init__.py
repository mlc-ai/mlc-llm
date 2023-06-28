from .quantization import FQuantize, FDequantize
from .quantization import QuantizationScheme
from .quantization import QuantizationSpec, NoQuantizationSpec, ParamQuantKind
from .group_quantization import GroupQuantizationSpec


# The predefined quantization schemes.
quantization_schemes = {
    "q0f16": QuantizationScheme("q0f16", NoQuantizationSpec("float16")),
    "q0f32": QuantizationScheme("q0f32", NoQuantizationSpec("float32")),
    "q3f16_0": QuantizationScheme(
        name="q3f16_0",
        linear_weight=GroupQuantizationSpec(
            dtype="float16",
            mode="int3",
            sym=True,
            storage_nbit=16,
            group_size=40,
            transpose=True,
        ),
        embedding_table=GroupQuantizationSpec(
            dtype="float16",
            mode="int3",
            sym=True,
            storage_nbit=16,
            group_size=40,
            transpose=False,
        ),
        final_fc_weight="same_as_linear_weight",
    ),
    "q4f16_0": QuantizationScheme(
        name="q4f16_0",
        linear_weight=GroupQuantizationSpec(
            dtype="float16",
            mode="int4",
            sym=True,
            storage_nbit=32,
            group_size=32,
            transpose=True,
        ),
        embedding_table=GroupQuantizationSpec(
            dtype="float16",
            mode="int4",
            sym=True,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        final_fc_weight="same_as_linear_weight",
    ),
    "q4f32_0": QuantizationScheme(
        name="q4f32_0",
        linear_weight=GroupQuantizationSpec(
            dtype="float32",
            mode="int4",
            sym=False,
            storage_nbit=32,
            group_size=32,
            transpose=True,
        ),
        embedding_table=GroupQuantizationSpec(
            dtype="float32",
            mode="int4",
            sym=False,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        final_fc_weight="same_as_linear_weight",
    ),
    # NOTE: `q8f16_0` and `q8f32_0` are not yet supported in the new quantization framework.
}
