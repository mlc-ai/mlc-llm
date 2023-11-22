from .quantization import FQuantize
from .quantization import QuantizationScheme
from .quantization import QuantizationSpec, NoQuantizationSpec, ParamQuantKind
from .quantization import QuantSpecUpdater
from .group_quantization import GroupQuantizationSpec
from .autogptq_quantization import AutogptqQuantizationSpec
from .ft_quantization import FTQuantizationSpec, FTQuantizeUpdater


# The predefined quantization schemes.
quantization_schemes = {
    "autogptq_llama_q4f16_0": QuantizationScheme(
        name="autogptq_llama_q4f16_0",
        linear_weight=AutogptqQuantizationSpec(
            dtype="float16",
            mode="int4",
            sym=False,
            group_size=128,
        ),
        embedding_table=NoQuantizationSpec("float16"),
        final_fc_weight=NoQuantizationSpec("float16"),
    ),
    "autogptq_llama_q4f16_1": QuantizationScheme(
        name="autogptq_llama_q4f16_1",
        linear_weight=AutogptqQuantizationSpec(
            dtype="float16",
            mode="int4",
            sym=False,
            group_size=-1,
        ),
        embedding_table=NoQuantizationSpec("float16"),
        final_fc_weight=NoQuantizationSpec("float16"),
    ),
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
    "q3f16_1": QuantizationScheme(
        name="q3f16_1",
        linear_weight=GroupQuantizationSpec(
            dtype="float16",
            mode="int3",
            sym=True,
            storage_nbit=16,
            group_size=40,
            transpose=False,
        ),
        embedding_table="same_as_linear_weight",
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
    "q4f16_1": QuantizationScheme(
        name="q4f16_1",
        linear_weight=GroupQuantizationSpec(
            dtype="float16",
            mode="int4",
            sym=True,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        embedding_table="same_as_linear_weight",
        final_fc_weight="same_as_linear_weight",
    ),
    "q4f16_2": QuantizationScheme(
        name="q4f16_2",
        linear_weight=GroupQuantizationSpec(
            dtype="float16",
            mode="int4",
            sym=True,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        embedding_table=NoQuantizationSpec("float16"),
        final_fc_weight=NoQuantizationSpec("float16"),
    ),
    "q4f16_ft": QuantizationScheme(
        name="q4f16_ft",
        linear_weight=FTQuantizationSpec(
            dtype="float16",
            nbit=4,
            group_size=-1,
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
        qspec_updater_class=FTQuantizeUpdater,
    ),
    "q4f16_ft_group": QuantizationScheme(
        name="q4f16_ft_group",
        linear_weight=FTQuantizationSpec(
            dtype="float16",
            nbit=4,
            group_size=64,
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
        qspec_updater_class=FTQuantizeUpdater,
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
    "q4f32_1": QuantizationScheme(
        name="q4f32_1",
        linear_weight=GroupQuantizationSpec(
            dtype="float32",
            mode="int4",
            sym=False,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        embedding_table="same_as_linear_weight",
        final_fc_weight="same_as_linear_weight",
    ),
    "q8f16_ft": QuantizationScheme(
        name="q8f16_ft",
        linear_weight=FTQuantizationSpec(
            dtype="float16",
            nbit=8,
        ),
        embedding_table=GroupQuantizationSpec(
            dtype="float16",
            mode="int8",
            sym=True,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        final_fc_weight="same_as_linear_weight",
        qspec_updater_class=FTQuantizeUpdater,
    ),
    "q8f16_ft_group": QuantizationScheme(
        name="q8f16_ft_group",
        linear_weight=FTQuantizationSpec(
            dtype="float16",
            nbit=8,
            group_size=64,
        ),
        embedding_table=GroupQuantizationSpec(
            dtype="float16",
            mode="int8",
            sym=True,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        final_fc_weight="same_as_linear_weight",
        qspec_updater_class=FTQuantizeUpdater,
    ),
    "q8f16_1": QuantizationScheme(
        name="q8f16_1",
        linear_weight=GroupQuantizationSpec(
            dtype="float16",
            mode="int8",
            sym=True,
            storage_nbit=32,
            group_size=32,
            transpose=False,
        ),
        embedding_table="same_as_linear_weight",
        final_fc_weight="same_as_linear_weight",
    ),
}
