from .quantization import FQuantize
from .quantization import QuantizationScheme
from .quantization import QuantizationSpec, NoQuantizationSpec, ParamQuantKind
from .group_quantization import GroupQuantizationSpec
from .autogptq_quantization import AutogptqQuantizationSpec, load_autogptq_params
from .rwkv_quantization import RWKVQuantizationSpec


# The predefined quantization schemes.
quantization_schemes = {
    "autogptq_llama_q4f16_0": QuantizationScheme(
        pre_quantized=True,
        name="autogptq_llama_q4f16_0",
        linear_weight=AutogptqQuantizationSpec(
            dtype="float16",
            mode="int4",
            sym=False,
            storage_nbit=32,
            group_size=-1,
            transpose=True,
        ),
        embedding_table=NoQuantizationSpec("float16"),
        final_fc_weight=NoQuantizationSpec("float16"),
        _base_model_prefix="model",
        _layers_block_name="model.layers",
        _inside_layer_modules=[
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"],
        ],
        _load_quantized_params_func=load_autogptq_params,
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
    "q8f16_0": QuantizationScheme(
        name="q8f16_0",
        linear_weight=RWKVQuantizationSpec(dtype="float16", mode="uint8", nbit=8),
        final_fc_weight="same_as_linear_weight",
    ),
}
