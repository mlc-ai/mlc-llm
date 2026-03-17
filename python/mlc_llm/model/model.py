"""A centralized registry of all existing model architures and their configurations."""

import dataclasses
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization import make_quantization_functions
from mlc_llm.quantization.quantization import Quantization

from .baichuan import baichuan_loader, baichuan_model
from .bert import bert_loader, bert_model
from .chatglm3 import chatglm3_loader, chatglm3_model
from .cohere import cohere_loader, cohere_model
from .deepseek import deepseek_loader, deepseek_model
from .deepseek_v2 import deepseek_v2_loader, deepseek_v2_model
from .eagle import eagle_loader, eagle_model
from .gemma import gemma_loader, gemma_model
from .gemma2 import gemma2_loader, gemma2_model
from .gemma3 import gemma3_loader, gemma3_model
from .gpt2 import gpt2_loader, gpt2_model
from .gpt_bigcode import gpt_bigcode_loader, gpt_bigcode_model
from .gpt_j import gpt_j_loader, gpt_j_model
from .gpt_neox import gpt_neox_loader, gpt_neox_model
from .internlm import internlm_loader, internlm_model
from .internlm2 import internlm2_loader, internlm2_model
from .llama import llama_loader, llama_model
from .llama4 import llama4_loader, llama4_model
from .llava import llava_loader, llava_model
from .medusa import medusa_loader, medusa_model
from .minicpm import minicpm_loader, minicpm_model
from .ministral3 import ministral3_loader, ministral3_model
from .mistral import mistral_loader, mistral_model
from .mixtral import mixtral_loader, mixtral_model
from .nemotron import nemotron_loader, nemotron_model
from .olmo import olmo_loader, olmo_model
from .orion import orion_loader, orion_model
from .phi import phi_loader, phi_model
from .phi3 import phi3_loader, phi3_model
from .phi3v import phi3v_loader, phi3v_model
from .qwen import qwen_loader, qwen_model
from .qwen2 import qwen2_loader, qwen2_model
from .qwen2_moe import qwen2_moe_loader, qwen2_moe_model
from .qwen3 import qwen3_loader, qwen3_model
from .qwen3_moe import qwen3_moe_loader, qwen3_moe_model
from .rwkv5 import rwkv5_loader, rwkv5_model
from .rwkv6 import rwkv6_loader, rwkv6_model
from .stable_lm import stablelm_loader, stablelm_model
from .starcoder2 import starcoder2_loader, starcoder2_model

ModelConfig = Any
"""A ModelConfig is an object that represents a model architecture. It is required to have
a class method `from_file` with the following signature:

    def from_file(cls, path: Path) -> ModelConfig:
        ...
"""

FuncGetExternMap = Callable[[ModelConfig, Quantization], ExternMapping]
FuncQuantization = Callable[[ModelConfig, Quantization], Tuple[nn.Module, QuantizeMapping]]


@dataclasses.dataclass
class EmbeddingMetadata:
    """Embedding model metadata.

    Parameters
    ----------
    model_type: Literal["encoder", "decoder"]
        The type of the embedding model.

    pooling_strategy: Literal["cls", "mean", "last"]
        The pooling strategy to use for the embedding model.

    normalize: bool = True
        Default to normalize the embedding.
    """

    model_type: Literal["encoder", "decoder"]
    pooling_strategy: Literal["cls", "mean", "last"]
    normalize: bool = True


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

    model_task: Literal["chat", "embedding"] = "chat"
        A task of the model to distinguish between chat and embedding models. Default to "chat".

    embedding_metadata: Optional[EmbeddingMetadata] = None
        Metadata for the embedding model. Default to None.
    """

    name: str
    config: ModelConfig
    model: Callable[[ModelConfig], nn.Module]
    source: Dict[str, FuncGetExternMap]
    quantize: Dict[str, FuncQuantization]

    model_task: Literal["chat", "embedding"] = "chat"
    embedding_metadata: Optional[EmbeddingMetadata] = None

    def __post_init__(self):
        if self.model_task == "embedding" and self.embedding_metadata is None:
            raise ValueError(f"[Model] {self.name}: Embedding model must have embedding metadata.")
        if self.model_task == "chat" and self.embedding_metadata is not None:
            raise ValueError(
                f"[Model] {self.name}: Chat model not expected to have embedding metadata."
            )


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
        quantize=make_quantization_functions(
            llama_model.LlamaForCausalLM,
            supports_awq=True,
            supports_per_tensor=True,
        ),
    ),
    "llama4": Model(
        name="llama4",
        model=llama4_model.Llama4ForCausalLM,
        config=llama4_model.Llama4Config,
        source={
            "huggingface-torch": llama4_loader.huggingface,
            "huggingface-safetensor": llama4_loader.huggingface,
        },
        quantize=make_quantization_functions(
            llama4_model.Llama4ForCausalLM,
            supports_per_tensor=True,
        ),
    ),
    "mistral": Model(
        name="mistral",
        model=mistral_model.MistralForCausalLM,
        config=mistral_model.MistralConfig,
        source={
            "huggingface-torch": mistral_loader.huggingface,
            "huggingface-safetensor": mistral_loader.huggingface,
            "awq": mistral_loader.awq,
        },
        quantize=make_quantization_functions(
            mistral_model.MistralForCausalLM,
        ),
    ),
    "ministral3": Model(
        name="ministral3",
        model=ministral3_model.Mistral3ForConditionalGeneration,
        config=ministral3_model.Ministral3Config,
        source={
            "huggingface-torch": ministral3_loader.huggingface,
            "huggingface-safetensor": ministral3_loader.huggingface,
        },
        quantize=make_quantization_functions(
            ministral3_model.Mistral3ForConditionalGeneration,
            supports_block_scale=True,
        ),
    ),
    "gemma": Model(
        name="gemma",
        model=gemma_model.GemmaForCausalLM,
        config=gemma_model.GemmaConfig,
        source={
            "huggingface-torch": gemma_loader.huggingface,
            "huggingface-safetensor": gemma_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gemma_model.GemmaForCausalLM,
            supports_ft_quant=False,
        ),
    ),
    "gemma2": Model(
        name="gemma2",
        model=gemma2_model.Gemma2ForCausalLM,
        config=gemma2_model.Gemma2Config,
        source={
            "huggingface-torch": gemma2_loader.huggingface,
            "huggingface-safetensor": gemma2_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gemma2_model.Gemma2ForCausalLM,
            supports_ft_quant=False,
        ),
    ),
    "gemma3": Model(
        name="gemma3",
        model=gemma3_model.Gemma3ForCausalLM,
        config=gemma3_model.Gemma3Config,
        source={
            "huggingface-torch": gemma3_loader.huggingface,
            "huggingface-safetensor": gemma3_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gemma3_model.Gemma3ForCausalLM,
            supports_ft_quant=False,
        ),
    ),
    "gemma3_text": Model(
        name="gemma3_text",
        model=gemma3_model.Gemma3ForCausalLM,
        config=gemma3_model.Gemma3Config,
        source={
            "huggingface-torch": gemma3_loader.huggingface,
            "huggingface-safetensor": gemma3_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gemma3_model.Gemma3ForCausalLM,
            supports_ft_quant=False,
        ),
    ),
    "gpt2": Model(
        name="gpt2",
        model=gpt2_model.GPT2LMHeadModel,
        config=gpt2_model.GPT2Config,
        source={
            "huggingface-torch": gpt2_loader.huggingface,
            "huggingface-safetensor": gpt2_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gpt2_model.GPT2LMHeadModel,
        ),
    ),
    "mixtral": Model(
        name="mixtral",
        model=mixtral_model.MixtralForCausalLM,
        config=mixtral_model.MixtralConfig,
        source={
            "huggingface-torch": mixtral_loader.huggingface,
            "huggingface-safetensor": mixtral_loader.huggingface,
        },
        quantize=make_quantization_functions(
            mixtral_model.MixtralForCausalLM,
            supports_awq=True,
            awq_unsupported_message="AWQ is not implemented for Mixtral models.",
            supports_per_tensor=True,
        ),
    ),
    "gpt_neox": Model(
        name="gpt_neox",
        model=gpt_neox_model.GPTNeoXForCausalLM,
        config=gpt_neox_model.GPTNeoXConfig,
        source={
            "huggingface-torch": gpt_neox_loader.huggingface,
            "huggingface-safetensor": gpt_neox_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gpt_neox_model.GPTNeoXForCausalLM,
        ),
    ),
    "gpt_bigcode": Model(
        name="gpt_bigcode",
        model=gpt_bigcode_model.GPTBigCodeForCausalLM,
        config=gpt_bigcode_model.GPTBigCodeConfig,
        source={
            "huggingface-torch": gpt_bigcode_loader.huggingface,
            "huggingface-safetensor": gpt_bigcode_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gpt_bigcode_model.GPTBigCodeForCausalLM,
        ),
    ),
    "phi-msft": Model(
        name="phi-msft",
        model=phi_model.PhiForCausalLM,
        config=phi_model.PhiConfig,
        source={
            "huggingface-torch": phi_loader.huggingface,
            "huggingface-safetensor": phi_loader.huggingface,
        },
        quantize=make_quantization_functions(
            phi_model.PhiForCausalLM,
        ),
    ),
    "phi": Model(
        name="phi",
        model=phi_model.PhiForCausalLM,
        config=phi_model.Phi1Config,
        source={
            "huggingface-torch": phi_loader.phi1_huggingface,
            "huggingface-safetensor": phi_loader.phi1_huggingface,
        },
        quantize=make_quantization_functions(
            phi_model.PhiForCausalLM,
        ),
    ),
    "phi3": Model(
        name="phi3",
        model=phi3_model.Phi3ForCausalLM,
        config=phi3_model.Phi3Config,
        source={
            "huggingface-torch": phi3_loader.phi3_huggingface,
            "huggingface-safetensor": phi3_loader.phi3_huggingface,
        },
        quantize=make_quantization_functions(
            phi3_model.Phi3ForCausalLM,
        ),
    ),
    "phi3_v": Model(
        name="phi3_v",
        model=phi3v_model.Phi3VForCausalLM,
        config=phi3v_model.Phi3VConfig,
        source={
            "huggingface-torch": phi3v_loader.huggingface,
            "huggingface-safetensor": phi3v_loader.huggingface,
        },
        quantize=make_quantization_functions(
            phi3v_model.Phi3VForCausalLM,
        ),
    ),
    "qwen": Model(
        name="qwen",
        model=qwen_model.QWenLMHeadModel,
        config=qwen_model.QWenConfig,
        source={
            "huggingface-torch": qwen_loader.huggingface,
            "huggingface-safetensor": qwen_loader.huggingface,
        },
        quantize=make_quantization_functions(
            qwen_model.QWenLMHeadModel,
        ),
    ),
    "qwen2": Model(
        name="qwen2",
        model=qwen2_model.QWen2LMHeadModel,
        config=qwen2_model.QWen2Config,
        source={
            "huggingface-torch": qwen2_loader.huggingface,
            "huggingface-safetensor": qwen2_loader.huggingface,
        },
        quantize=make_quantization_functions(
            qwen2_model.QWen2LMHeadModel,
        ),
    ),
    "qwen2_moe": Model(
        name="qwen2_moe",
        model=qwen2_moe_model.Qwen2MoeForCausalLM,
        config=qwen2_moe_model.Qwen2MoeConfig,
        source={
            "huggingface-torch": qwen2_moe_loader.huggingface,
            "huggingface-safetensor": qwen2_moe_loader.huggingface,
        },
        quantize=make_quantization_functions(
            qwen2_moe_model.Qwen2MoeForCausalLM,
        ),
    ),
    "qwen3": Model(
        name="qwen3",
        model=qwen3_model.Qwen3LMHeadModel,
        config=qwen3_model.Qwen3Config,
        source={
            "huggingface-torch": qwen3_loader.huggingface,
            "huggingface-safetensor": qwen3_loader.huggingface,
        },
        quantize=make_quantization_functions(
            qwen3_model.Qwen3LMHeadModel,
            supports_block_scale=True,
        ),
    ),
    "qwen3-embedding": Model(
        name="qwen3-embedding",
        model=qwen3_model.Qwen3EmbeddingModel,
        config=qwen3_model.Qwen3Config,
        source={
            "huggingface-torch": qwen3_loader.huggingface_embedding,
            "huggingface-safetensor": qwen3_loader.huggingface_embedding,
        },
        quantize=make_quantization_functions(
            qwen3_model.Qwen3EmbeddingModel,
            supports_block_scale=True,
        ),
        model_task="embedding",
        embedding_metadata=EmbeddingMetadata(
            model_type="decoder",
            pooling_strategy="last",
            normalize=True,
        ),
    ),
    "qwen3_moe": Model(
        name="qwen3_moe",
        model=qwen3_moe_model.Qwen3MoeForCausalLM,
        config=qwen3_moe_model.Qwen3MoeConfig,
        source={
            "huggingface-torch": qwen3_moe_loader.huggingface,
            "huggingface-safetensor": qwen3_moe_loader.huggingface,
        },
        quantize=make_quantization_functions(
            qwen3_moe_model.Qwen3MoeForCausalLM,
            supports_block_scale=True,
        ),
    ),
    "deepseek_v2": Model(
        name="deepseek_v2",
        model=deepseek_v2_model.DeepseekV2ForCausalLM,
        config=deepseek_v2_model.DeepseekV2Config,
        source={
            "huggingface-torch": deepseek_v2_loader.huggingface,
            "huggingface-safetensor": deepseek_v2_loader.huggingface,
        },
        quantize=make_quantization_functions(
            deepseek_v2_model.DeepseekV2ForCausalLM,
        ),
    ),
    "deepseek_v3": Model(
        name="deepseek_v3",
        model=deepseek_v2_model.DeepseekV2ForCausalLM,
        config=deepseek_v2_model.DeepseekV2Config,
        source={
            "huggingface-torch": deepseek_v2_loader.huggingface,
            "huggingface-safetensor": deepseek_v2_loader.huggingface,
        },
        quantize=make_quantization_functions(
            deepseek_v2_model.DeepseekV2ForCausalLM,
            supports_block_scale=True,
        ),
    ),
    "stablelm": Model(
        name="stablelm",
        model=stablelm_model.StableLmForCausalLM,
        config=stablelm_model.StableLmConfig,
        source={
            "huggingface-torch": stablelm_loader.huggingface,
            "huggingface-safetensor": stablelm_loader.huggingface,
        },
        quantize=make_quantization_functions(
            stablelm_model.StableLmForCausalLM,
        ),
    ),
    "baichuan": Model(
        name="baichuan",
        model=baichuan_model.BaichuanForCausalLM,
        config=baichuan_model.BaichuanConfig,
        source={
            "huggingface-torch": baichuan_loader.huggingface,
            "huggingface-safetensor": baichuan_loader.huggingface,
        },
        quantize=make_quantization_functions(
            baichuan_model.BaichuanForCausalLM,
        ),
    ),
    "internlm": Model(
        name="internlm",
        model=internlm_model.InternLMForCausalLM,
        config=internlm_model.InternLMConfig,
        source={
            "huggingface-torch": internlm_loader.huggingface,
            "huggingface-safetensor": internlm_loader.huggingface,
        },
        quantize=make_quantization_functions(
            internlm_model.InternLMForCausalLM,
        ),
    ),
    "internlm2": Model(
        name="internlm2",
        model=internlm2_model.InternLM2ForCausalLM,
        config=internlm2_model.InternLM2Config,
        source={
            "huggingface-torch": internlm2_loader.huggingface,
            "huggingface-safetensor": internlm2_loader.huggingface,
        },
        quantize=make_quantization_functions(
            internlm2_model.InternLM2ForCausalLM,
        ),
    ),
    "rwkv5": Model(
        name="rwkv5",
        model=rwkv5_model.RWKV5_ForCausalLM,
        config=rwkv5_model.RWKV5Config,
        source={
            "huggingface-torch": rwkv5_loader.huggingface,
            "huggingface-safetensor": rwkv5_loader.huggingface,
        },
        quantize=make_quantization_functions(
            rwkv5_model.RWKV5_ForCausalLM,
        ),
    ),
    "orion": Model(
        name="orion",
        model=orion_model.OrionForCausalLM,
        config=orion_model.OrionConfig,
        source={
            "huggingface-torch": orion_loader.huggingface,
            "huggingface-safetensor": orion_loader.huggingface,
        },
        quantize=make_quantization_functions(
            orion_model.OrionForCausalLM,
            supports_ft_quant=False,
        ),
    ),
    "llava": Model(
        name="llava",
        model=llava_model.LlavaForCausalLM,
        config=llava_model.LlavaConfig,
        source={
            "huggingface-torch": llava_loader.huggingface,
            "huggingface-safetensor": llava_loader.huggingface,
            "awq": llava_loader.awq,
        },
        quantize=make_quantization_functions(
            llava_model.LlavaForCausalLM,
            supports_awq=True,
            supports_ft_quant=False,
        ),
    ),
    "rwkv6": Model(
        name="rwkv6",
        model=rwkv6_model.RWKV6_ForCausalLM,
        config=rwkv6_model.RWKV6Config,
        source={
            "huggingface-torch": rwkv6_loader.huggingface,
            "huggingface-safetensor": rwkv6_loader.huggingface,
        },
        quantize=make_quantization_functions(
            rwkv6_model.RWKV6_ForCausalLM,
            supports_ft_quant=False,
        ),
    ),
    "chatglm": Model(
        name="chatglm",
        model=chatglm3_model.ChatGLMForCausalLM,
        config=chatglm3_model.GLMConfig,
        source={
            "huggingface-torch": chatglm3_loader.huggingface,
            "huggingface-safetensor": chatglm3_loader.huggingface,
        },
        quantize=make_quantization_functions(
            chatglm3_model.ChatGLMForCausalLM,
            supports_ft_quant=False,
        ),
    ),
    "eagle": Model(
        name="eagle",
        model=eagle_model.EagleForCausalLM,
        config=eagle_model.EagleConfig,
        source={
            "huggingface-torch": eagle_loader.huggingface,
            "huggingface-safetensor": eagle_loader.huggingface,
            "awq": eagle_loader.awq,
        },
        quantize=make_quantization_functions(
            eagle_model.EagleForCausalLM,
            supports_awq=True,
        ),
    ),
    "bert": Model(
        name="bert",
        model=bert_model.BertModel,
        config=bert_model.BertConfig,
        source={
            "huggingface-torch": bert_loader.huggingface,
            "huggingface-safetensor": bert_loader.huggingface,
        },
        quantize=make_quantization_functions(
            bert_model.BertModel,
        ),
        model_task="embedding",
        embedding_metadata=EmbeddingMetadata(
            model_type="encoder",
            pooling_strategy="cls",
            normalize=True,
        ),
    ),
    "medusa": Model(
        name="medusa",
        model=medusa_model.MedusaModel,
        config=medusa_model.MedusaConfig,
        source={
            "huggingface-torch": medusa_loader.huggingface,
            "huggingface-safetensor": medusa_loader.huggingface,
        },
        quantize=make_quantization_functions(
            medusa_model.MedusaModel,
            supports_group_quant=False,
            supports_ft_quant=False,
        ),
    ),
    "starcoder2": Model(
        name="starcoder2",
        model=starcoder2_model.Starcoder2ForCausalLM,
        config=starcoder2_model.Starcoder2Config,
        source={
            "huggingface-torch": starcoder2_loader.huggingface,
            "huggingface-safetensor": starcoder2_loader.huggingface,
        },
        quantize=make_quantization_functions(
            starcoder2_model.Starcoder2ForCausalLM,
        ),
    ),
    "cohere": Model(
        name="cohere",
        model=cohere_model.CohereForCausalLM,
        config=cohere_model.CohereConfig,
        source={
            "huggingface-torch": cohere_loader.huggingface,
            "huggingface-safetensor": cohere_loader.huggingface,
        },
        quantize=make_quantization_functions(
            cohere_model.CohereForCausalLM,
        ),
    ),
    "minicpm": Model(
        name="minicpm",
        model=minicpm_model.MiniCPMForCausalLM,
        config=minicpm_model.MiniCPMConfig,
        source={
            "huggingface-torch": minicpm_loader.huggingface,
            "huggingface-safetensor": minicpm_loader.huggingface,
        },
        quantize=make_quantization_functions(
            minicpm_model.MiniCPMForCausalLM,
        ),
    ),
    "deepseek": Model(
        name="deepseek",
        model=deepseek_model.DeepseekForCausalLM,
        config=deepseek_model.DeepseekConfig,
        source={
            "huggingface-torch": deepseek_loader.huggingface,
            "huggingface-safetensor": deepseek_loader.huggingface,
        },
        quantize=make_quantization_functions(
            deepseek_model.DeepseekForCausalLM,
        ),
    ),
    "gptj": Model(
        name="gptj",
        model=gpt_j_model.GPTJForCausalLM,
        config=gpt_j_model.GPTJConfig,
        source={
            "huggingface-torch": gpt_j_loader.huggingface,
            "huggingface-safetensor": gpt_j_loader.huggingface,
        },
        quantize=make_quantization_functions(
            gpt_j_model.GPTJForCausalLM,
        ),
    ),
    "olmo": Model(
        name="olmo",
        model=olmo_model.OLMoForCausalLM,
        config=olmo_model.OLMoConfig,
        source={
            "huggingface-torch": olmo_loader.huggingface,
            "huggingface-safetensor": olmo_loader.huggingface,
            "awq": olmo_loader.awq,
        },
        quantize=make_quantization_functions(
            olmo_model.OLMoForCausalLM,
            supports_awq=True,
            supports_per_tensor=True,
        ),
    ),
    "nemotron": Model(
        name="nemotron",
        model=nemotron_model.NemotronForCausalLM,
        config=nemotron_model.NemotronConfig,
        source={
            "huggingface-torch": nemotron_loader.huggingface,
            "huggingface-safetensor": nemotron_loader.huggingface,
        },
        quantize=make_quantization_functions(
            nemotron_model.NemotronForCausalLM,
            supports_awq=True,
            supports_per_tensor=True,
        ),
    ),
    "bert-bge": Model(
        name="bert-bge",
        model=bert_model.BertModel,
        config=bert_model.BertConfig,
        source={
            "huggingface-torch": bert_loader.huggingface_bge,
            "huggingface-safetensor": bert_loader.huggingface_bge,
        },
        quantize=make_quantization_functions(
            bert_model.BertModel,
        ),
        model_task="embedding",
        embedding_metadata=EmbeddingMetadata(
            model_type="encoder",
            pooling_strategy="cls",
            normalize=True,
        ),
    ),
}
