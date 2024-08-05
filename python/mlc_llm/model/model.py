"""A centralized registry of all existing model architures and their configurations."""

import dataclasses
from typing import Any, Callable, Dict, Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization.quantization import Quantization

from .baichuan import baichuan_loader, baichuan_model, baichuan_quantization
from .bert import bert_loader, bert_model, bert_quantization
from .chatglm3 import chatglm3_loader, chatglm3_model, chatglm3_quantization
from .cohere import cohere_loader, cohere_model, cohere_quantization
from .eagle import eagle_loader, eagle_model, eagle_quantization
from .gemma import gemma_loader, gemma_model, gemma_quantization
from .gemma2 import gemma2_loader, gemma2_model, gemma2_quantization
from .gpt2 import gpt2_loader, gpt2_model, gpt2_quantization
from .gpt_bigcode import gpt_bigcode_loader, gpt_bigcode_model, gpt_bigcode_quantization
from .gpt_neox import gpt_neox_loader, gpt_neox_model, gpt_neox_quantization
from .internlm import internlm_loader, internlm_model, internlm_quantization
from .internlm2 import internlm2_loader, internlm2_model, internlm2_quantization
from .llama import llama_loader, llama_model, llama_quantization
from .llava import llava_loader, llava_model, llava_quantization
from .medusa import medusa_loader, medusa_model, medusa_quantization
from .minicpm import minicpm_loader, minicpm_model, minicpm_quantization
from .mistral import mistral_loader, mistral_model, mistral_quantization
from .mixtral import mixtral_loader, mixtral_model, mixtral_quantization
from .orion import orion_loader, orion_model, orion_quantization
from .phi import phi_loader, phi_model, phi_quantization
from .phi3 import phi3_loader, phi3_model, phi3_quantization
from .phi3v import phi3v_loader, phi3v_model, phi3v_quantization
from .qwen import qwen_loader, qwen_model, qwen_quantization
from .qwen2 import qwen2_loader, qwen2_model, qwen2_quantization
from .qwen2_moe import qwen2_moe_loader, qwen2_moe_model, qwen2_moe_quantization
from .rwkv5 import rwkv5_loader, rwkv5_model, rwkv5_quantization
from .rwkv6 import rwkv6_loader, rwkv6_model, rwkv6_quantization
from .stable_lm import stablelm_loader, stablelm_model, stablelm_quantization
from .starcoder2 import starcoder2_loader, starcoder2_model, starcoder2_quantization

ModelConfig = Any
"""A ModelConfig is an object that represents a model architecture. It is required to have
a class method `from_file` with the following signature:

    def from_file(cls, path: Path) -> ModelConfig:
        ...
"""

FuncGetExternMap = Callable[[ModelConfig, Quantization], ExternMapping]
FuncQuantization = Callable[[ModelConfig, Quantization], Tuple[nn.Module, QuantizeMapping]]


@dataclasses.dataclass
class Model:
    """All about a model architecture: its configuration, its parameter loader and quantization.

    Parameters
    ----------
    name : str
        The name of the model.

    model : Callable[[ModelConfig], nn.Module]
        A method that creates the `nn.Module` that represents the model from `ModelConfig`.

    config : ModelConfig
        A class that has a `from_file` class method, whose signature is "Path -> ModelConfig".

    source : Dict[str, FuncGetExternMap]
        A dictionary that maps the name of a source format to parameter mapping.

    quantize: Dict[str, FuncQuantization]
        A dictionary that maps the name of a quantization method to quantized model and the
        quantization parameter mapping.
    """

    name: str
    config: ModelConfig
    model: Callable[[ModelConfig], nn.Module]
    source: Dict[str, FuncGetExternMap]
    quantize: Dict[str, FuncQuantization]


MODELS: Dict[str, Model] = {
    "llama": Model(
        name="llama",
        model=llama_model.LlamaForCausalLM,
        config=llama_model.LlamaConfig,
        source={
            "huggingface-torch": llama_loader.huggingface,
            "huggingface-safetensor": llama_loader.huggingface,
            "awq": llama_loader.awq,
        },
        quantize={
            "no-quant": llama_quantization.no_quant,
            "group-quant": llama_quantization.group_quant,
            "ft-quant": llama_quantization.ft_quant,
            "awq": llama_quantization.awq_quant,
            "per-tensor-quant": llama_quantization.per_tensor_quant,
        },
    ),
    "mistral": Model(
        name="mistral",
        model=mistral_model.MistralForCasualLM,
        config=mistral_model.MistralConfig,
        source={
            "huggingface-torch": mistral_loader.huggingface,
            "huggingface-safetensor": mistral_loader.huggingface,
            "awq": mistral_loader.awq,
        },
        quantize={
            "group-quant": mistral_quantization.group_quant,
            "no-quant": mistral_quantization.no_quant,
            "ft-quant": mistral_quantization.ft_quant,
        },
    ),
    "gemma": Model(
        name="gemma",
        model=gemma_model.GemmaForCausalLM,
        config=gemma_model.GemmaConfig,
        source={
            "huggingface-torch": gemma_loader.huggingface,
            "huggingface-safetensor": gemma_loader.huggingface,
        },
        quantize={
            "no-quant": gemma_quantization.no_quant,
            "group-quant": gemma_quantization.group_quant,
        },
    ),
    "gemma2": Model(
        name="gemma2",
        model=gemma2_model.Gemma2ForCausalLM,
        config=gemma2_model.Gemma2Config,
        source={
            "huggingface-torch": gemma2_loader.huggingface,
            "huggingface-safetensor": gemma2_loader.huggingface,
        },
        quantize={
            "no-quant": gemma2_quantization.no_quant,
            "group-quant": gemma2_quantization.group_quant,
        },
    ),
    "gpt2": Model(
        name="gpt2",
        model=gpt2_model.GPT2LMHeadModel,
        config=gpt2_model.GPT2Config,
        source={
            "huggingface-torch": gpt2_loader.huggingface,
            "huggingface-safetensor": gpt2_loader.huggingface,
        },
        quantize={
            "no-quant": gpt2_quantization.no_quant,
            "group-quant": gpt2_quantization.group_quant,
            "ft-quant": gpt2_quantization.ft_quant,
        },
    ),
    "mixtral": Model(
        name="mixtral",
        model=mixtral_model.MixtralForCasualLM,
        config=mixtral_model.MixtralConfig,
        source={
            "huggingface-torch": mixtral_loader.huggingface,
            "huggingface-safetensor": mixtral_loader.huggingface,
        },
        quantize={
            "no-quant": mixtral_quantization.no_quant,
            "group-quant": mixtral_quantization.group_quant,
            "ft-quant": mixtral_quantization.ft_quant,
            "per-tensor-quant": mixtral_quantization.per_tensor_quant,
        },
    ),
    "gpt_neox": Model(
        name="gpt_neox",
        model=gpt_neox_model.GPTNeoXForCausalLM,
        config=gpt_neox_model.GPTNeoXConfig,
        source={
            "huggingface-torch": gpt_neox_loader.huggingface,
            "huggingface-safetensor": gpt_neox_loader.huggingface,
        },
        quantize={
            "no-quant": gpt_neox_quantization.no_quant,
            "group-quant": gpt_neox_quantization.group_quant,
            "ft-quant": gpt_neox_quantization.ft_quant,
        },
    ),
    "gpt_bigcode": Model(
        name="gpt_bigcode",
        model=gpt_bigcode_model.GPTBigCodeForCausalLM,
        config=gpt_bigcode_model.GPTBigCodeConfig,
        source={
            "huggingface-torch": gpt_bigcode_loader.huggingface,
            "huggingface-safetensor": gpt_bigcode_loader.huggingface,
        },
        quantize={
            "no-quant": gpt_bigcode_quantization.no_quant,
            "group-quant": gpt_bigcode_quantization.group_quant,
            "ft-quant": gpt_bigcode_quantization.ft_quant,
        },
    ),
    "phi-msft": Model(
        name="phi-msft",
        model=phi_model.PhiForCausalLM,
        config=phi_model.PhiConfig,
        source={
            "huggingface-torch": phi_loader.huggingface,
            "huggingface-safetensor": phi_loader.huggingface,
        },
        quantize={
            "no-quant": phi_quantization.no_quant,
            "group-quant": phi_quantization.group_quant,
            "ft-quant": phi_quantization.ft_quant,
        },
    ),
    "phi": Model(
        name="phi",
        model=phi_model.PhiForCausalLM,
        config=phi_model.Phi1Config,
        source={
            "huggingface-torch": phi_loader.phi1_huggingface,
            "huggingface-safetensor": phi_loader.phi1_huggingface,
        },
        quantize={
            "no-quant": phi_quantization.no_quant,
            "group-quant": phi_quantization.group_quant,
            "ft-quant": phi_quantization.ft_quant,
        },
    ),
    "phi3": Model(
        name="phi3",
        model=phi3_model.Phi3ForCausalLM,
        config=phi3_model.Phi3Config,
        source={
            "huggingface-torch": phi3_loader.phi3_huggingface,
            "huggingface-safetensor": phi3_loader.phi3_huggingface,
        },
        quantize={
            "no-quant": phi3_quantization.no_quant,
            "group-quant": phi3_quantization.group_quant,
            "ft-quant": phi3_quantization.ft_quant,
        },
    ),
    "phi3_v": Model(
        name="phi3_v",
        model=phi3v_model.Phi3VForCausalLM,
        config=phi3v_model.Phi3VConfig,
        source={
            "huggingface-torch": phi3v_loader.huggingface,
            "huggingface-safetensor": phi3v_loader.huggingface,
        },
        quantize={
            "no-quant": phi3v_quantization.no_quant,
            "group-quant": phi3v_quantization.group_quant,
            "ft-quant": phi3v_quantization.ft_quant,
        },
    ),
    "qwen": Model(
        name="qwen",
        model=qwen_model.QWenLMHeadModel,
        config=qwen_model.QWenConfig,
        source={
            "huggingface-torch": qwen_loader.huggingface,
            "huggingface-safetensor": qwen_loader.huggingface,
        },
        quantize={
            "no-quant": qwen_quantization.no_quant,
            "group-quant": qwen_quantization.group_quant,
            "ft-quant": qwen_quantization.ft_quant,
        },
    ),
    "qwen2": Model(
        name="qwen2",
        model=qwen2_model.QWen2LMHeadModel,
        config=qwen2_model.QWen2Config,
        source={
            "huggingface-torch": qwen2_loader.huggingface,
            "huggingface-safetensor": qwen2_loader.huggingface,
        },
        quantize={
            "no-quant": qwen2_quantization.no_quant,
            "group-quant": qwen2_quantization.group_quant,
            "ft-quant": qwen2_quantization.ft_quant,
        },
    ),
    "qwen2_moe": Model(
        name="qwen2_moe",
        model=qwen2_moe_model.Qwen2MoeForCausalLM,
        config=qwen2_moe_model.Qwen2MoeConfig,
        source={
            "huggingface-torch": qwen2_moe_loader.huggingface,
            "huggingface-safetensor": qwen2_moe_loader.huggingface,
        },
        quantize={
            "no-quant": qwen2_moe_quantization.no_quant,
            "group-quant": qwen2_moe_quantization.group_quant,
            "ft-quant": qwen2_moe_quantization.ft_quant,
        },
    ),
    "stablelm": Model(
        name="stablelm",
        model=stablelm_model.StableLmForCausalLM,
        config=stablelm_model.StableLmConfig,
        source={
            "huggingface-torch": stablelm_loader.huggingface,
            "huggingface-safetensor": stablelm_loader.huggingface,
        },
        quantize={
            "no-quant": stablelm_quantization.no_quant,
            "group-quant": stablelm_quantization.group_quant,
            "ft-quant": stablelm_quantization.ft_quant,
        },
    ),
    "baichuan": Model(
        name="baichuan",
        model=baichuan_model.BaichuanForCausalLM,
        config=baichuan_model.BaichuanConfig,
        source={
            "huggingface-torch": baichuan_loader.huggingface,
            "huggingface-safetensor": baichuan_loader.huggingface,
        },
        quantize={
            "no-quant": baichuan_quantization.no_quant,
            "group-quant": baichuan_quantization.group_quant,
            "ft-quant": baichuan_quantization.ft_quant,
        },
    ),
    "internlm": Model(
        name="internlm",
        model=internlm_model.InternLMForCausalLM,
        config=internlm_model.InternLMConfig,
        source={
            "huggingface-torch": internlm_loader.huggingface,
            "huggingface-safetensor": internlm_loader.huggingface,
        },
        quantize={
            "no-quant": internlm_quantization.no_quant,
            "group-quant": internlm_quantization.group_quant,
            "ft-quant": internlm_quantization.ft_quant,
        },
    ),
    "internlm2": Model(
        name="internlm2",
        model=internlm2_model.InternLM2ForCausalLM,
        config=internlm2_model.InternLM2Config,
        source={
            "huggingface-torch": internlm2_loader.huggingface,
            "huggingface-safetensor": internlm2_loader.huggingface,
        },
        quantize={
            "no-quant": internlm2_quantization.no_quant,
            "group-quant": internlm2_quantization.group_quant,
            "ft-quant": internlm2_quantization.ft_quant,
        },
    ),
    "rwkv5": Model(
        name="rwkv5",
        model=rwkv5_model.RWKV5_ForCasualLM,
        config=rwkv5_model.RWKV5Config,
        source={
            "huggingface-torch": rwkv5_loader.huggingface,
            "huggingface-safetensor": rwkv5_loader.huggingface,
        },
        quantize={
            "no-quant": rwkv5_quantization.no_quant,
            "group-quant": rwkv5_quantization.group_quant,
            "ft-quant": rwkv5_quantization.ft_quant,
        },
    ),
    "orion": Model(
        name="orion",
        model=orion_model.OrionForCasualLM,
        config=orion_model.OrionConfig,
        source={
            "huggingface-torch": orion_loader.huggingface,
            "huggingface-safetensor": orion_loader.huggingface,
        },
        quantize={
            "no-quant": orion_quantization.no_quant,
            "group-quant": orion_quantization.group_quant,
        },
    ),
    "llava": Model(
        name="llava",
        model=llava_model.LlavaForCasualLM,
        config=llava_model.LlavaConfig,
        source={
            "huggingface-torch": llava_loader.huggingface,
            "huggingface-safetensor": llava_loader.huggingface,
            "awq": llava_loader.awq,
        },
        quantize={
            "group-quant": llava_quantization.group_quant,
            "no-quant": llava_quantization.no_quant,
            "awq": llava_quantization.awq_quant,
        },
    ),
    "rwkv6": Model(
        name="rwkv6",
        model=rwkv6_model.RWKV6_ForCasualLM,
        config=rwkv6_model.RWKV6Config,
        source={
            "huggingface-torch": rwkv6_loader.huggingface,
            "huggingface-safetensor": rwkv6_loader.huggingface,
        },
        quantize={
            "no-quant": rwkv6_quantization.no_quant,
            "group-quant": rwkv6_quantization.group_quant,
        },
    ),
    "chatglm": Model(
        name="chatglm",
        model=chatglm3_model.ChatGLMForCausalLM,
        config=chatglm3_model.GLMConfig,
        source={
            "huggingface-torch": chatglm3_loader.huggingface,
            "huggingface-safetensor": chatglm3_loader.huggingface,
        },
        quantize={
            "no-quant": chatglm3_quantization.no_quant,
            "group-quant": chatglm3_quantization.group_quant,
        },
    ),
    "eagle": Model(
        name="eagle",
        model=eagle_model.EagleForCasualLM,
        config=eagle_model.EagleConfig,
        source={
            "huggingface-torch": eagle_loader.huggingface,
            "huggingface-safetensor": eagle_loader.huggingface,
            "awq": eagle_loader.awq,
        },
        quantize={
            "no-quant": eagle_quantization.no_quant,
            "group-quant": eagle_quantization.group_quant,
            "ft-quant": eagle_quantization.ft_quant,
            "awq": eagle_quantization.awq_quant,
        },
    ),
    "bert": Model(
        name="bert",
        model=bert_model.BertModel,
        config=bert_model.BertConfig,
        source={
            "huggingface-torch": bert_loader.huggingface,
            "huggingface-safetensor": bert_loader.huggingface,
        },
        quantize={
            "no-quant": bert_quantization.no_quant,
            "group-quant": bert_quantization.group_quant,
            "ft-quant": bert_quantization.ft_quant,
        },
    ),
    "medusa": Model(
        name="medusa",
        model=medusa_model.MedusaModel,
        config=medusa_model.MedusaConfig,
        source={
            "huggingface-torch": medusa_loader.huggingface,
            "huggingface-safetensor": medusa_loader.huggingface,
        },
        quantize={
            "no-quant": medusa_quantization.no_quant,
        },
    ),
    "starcoder2": Model(
        name="starcoder2",
        model=starcoder2_model.Starcoder2ForCausalLM,
        config=starcoder2_model.Starcoder2Config,
        source={
            "huggingface-torch": starcoder2_loader.huggingface,
            "huggingface-safetensor": starcoder2_loader.huggingface,
        },
        quantize={
            "no-quant": starcoder2_quantization.no_quant,
            "group-quant": starcoder2_quantization.group_quant,
            "ft-quant": starcoder2_quantization.ft_quant,
        },
    ),
    "cohere": Model(
        name="cohere",
        model=cohere_model.CohereForCausalLM,
        config=cohere_model.CohereConfig,
        source={
            "huggingface-torch": cohere_loader.huggingface,
            "huggingface-safetensor": cohere_loader.huggingface,
        },
        quantize={
            "no-quant": cohere_quantization.no_quant,
            "group-quant": cohere_quantization.group_quant,
            "ft-quant": cohere_quantization.ft_quant,
        },
    ),
    "minicpm": Model(
        name="minicpm",
        model=minicpm_model.MiniCPMForCausalLM,
        config=minicpm_model.MiniCPMConfig,
        source={
            "huggingface-torch": minicpm_loader.huggingface,
            "huggingface-safetensor": minicpm_loader.huggingface,
        },
        quantize={
            "no-quant": minicpm_quantization.no_quant,
            "group-quant": minicpm_quantization.group_quant,
            "ft-quant": minicpm_quantization.ft_quant,
        },
    ),
}
