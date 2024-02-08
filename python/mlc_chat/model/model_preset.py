"""A builtin set of models available in MLC LLM."""

from typing import Any, Dict

"""A centralized registry of all existing model architures and their configurations."""
import dataclasses
from typing import Any, Callable, Dict, Tuple

from tvm.relax.frontend import nn

from ..loader import ExternMapping, QuantizeMapping
from ..quantization.quantization import QUANTIZATION

from .gpt2 import gpt2_loader, gpt2_model, gpt2_quantization
from .gpt_bigcode import gpt_bigcode_loader, gpt_bigcode_model, gpt_bigcode_quantization
from .gpt_neox import gpt_neox_loader, gpt_neox_model, gpt_neox_quantization
from .llama import llama_loader, llama_model, llama_quantization
from .mistral import mistral_loader, mistral_model, mistral_quantization
from .llava import llava_loader, llava_model, llava_quantization

ModelConfig = Any
"""A ModelConfig is an object that represents a model architecture. It is required to have
a class method `from_file` with the following signature:

    def from_file(cls, path: Path) -> ModelConfig:
        ...
"""

FuncGetExternMap = Callable[[ModelConfig, QUANTIZATION], ExternMapping]
FuncQuantization = Callable[[ModelConfig, QUANTIZATION], Tuple[nn.Module, QuantizeMapping]]


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
        model=llama_model.LlamaForCasualLM,
        config=llama_model.LlamaConfig,
        source={
            "huggingface-torch": llama_loader.huggingface,
            "huggingface-safetensor": llama_loader.huggingface,
            "awq": llama_loader.awq,
        },
        quantize={
            "no-quant": llama_quantization.no_quant,
            "group-quant": llama_quantization.group_quant,
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
        },
    ),
}
MODEL_PRESETS: Dict[str, Any] = {
    "llava": {
        "architectures": ["LlavaForCasualLM"],
        "model_type": "llava",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 13824,
        "max_position_embeddings": 2048,
        "context_window_size": 4096,
        "model_type": "llama",
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 40,
        "pad_token_id": 0,
        "pretraining_tp": 2,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
    },
    "llama2_7b": {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
    },
    "llama2_13b": {
        "_name_or_path": "meta-llama/Llama-2-13b-hf",
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 13824,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 40,
        "pad_token_id": 0,
        "pretraining_tp": 2,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
    },
    "llama2_70b": {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 28672,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
    },
    "codellama_7b": {
        "_name_or_path": "codellama/CodeLlama-7b-hf",
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 16384,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.33.0.dev0",
        "use_cache": True,
        "vocab_size": 32016,
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
    },
    "codellama_13b": {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 13824,
        "max_position_embeddings": 16384,
        "model_type": "llama",
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 40,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.32.0.dev0",
        "use_cache": True,
        "vocab_size": 32016,
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
    },
    "codellama_34b": {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 22016,
        "max_position_embeddings": 16384,
        "model_type": "llama",
        "num_attention_heads": 64,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.32.0.dev0",
        "use_cache": True,
        "vocab_size": 32016,
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
    },
    "mistral_7b": {
        "architectures": ["MistralForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mistral",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.34.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
        "sliding_window_size": 4096,
        "prefill_chunk_size": 128,
        "attention_sink_size": 4,
    },
    "gpt2": {
        "architectures": ["GPT2LMHeadModel"],
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "hidden_act": "gelu_new",
        "n_embd": 768,
        "initializer_range": 0.02,
        "n_positions": 1024,
        "model_type": "gpt2",
        "n_head": 12,
        "n_layer": 12,
        "layer_norm_epsilon": 1e-05,
        "transformers_version": "4.26.0.dev0",
        "use_cache": True,
        "vocab_size": 50257,
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
    },
    "gpt_bigcode": {
        "activation_function": "gelu_pytorch_tanh",
        "architectures": ["GPTBigCodeForCausalLM"],
        "attention_softmax_in_fp32": True,
        "multi_query": True,
        "attn_pdrop": 0.1,
        "bos_token_id": 49152,
        "embd_pdrop": 0.1,
        "eos_token_id": 49152,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt_bigcode",
        "n_embd": 2048,
        "n_head": 16,
        "n_inner": 8192,
        "n_layer": 24,
        "n_positions": 2048,
        "resid_pdrop": 0.1,
        "runner_max_sequence_length": None,
        "scale_attention_softmax_in_fp32": True,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "transformers_version": "4.28.0.dev0",
        "use_cache": True,
        "vocab_size": 49280,
    },
    "Mixtral-8x7B-v0.1": {
        "architectures": ["MixtralForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mixtral",
        "num_attention_heads": 32,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "output_router_logits": False,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.02,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.36.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
    },
    "redpajama_3b_v1": {
        "_name_or_path": "/root/fm/models/rp_3b_800b_real_fp16",
        "architectures": ["GPTNeoXForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "gelu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 10240,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 2048,
        "model_type": "gpt_neox",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "rotary_emb_base": 10000,
        "rotary_pct": 1.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.28.1",
        "use_cache": True,
        "use_parallel_residual": False,
        "vocab_size": 50432,
    },
    "phi-1_5": {
        "_name_or_path": "microsoft/phi-1_5",
        "activation_function": "gelu_new",
        "architectures": ["PhiForCausalLM"],
        "attn_pdrop": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_phi.PhiConfig",
            "AutoModelForCausalLM": "modeling_phi.PhiForCausalLM",
        },
        "embd_pdrop": 0.0,
        "flash_attn": False,
        "flash_rotary": False,
        "fused_dense": False,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "phi-msft",
        "n_embd": 2048,
        "n_head": 32,
        "n_head_kv": None,
        "n_inner": None,
        "n_layer": 24,
        "n_positions": 2048,
        "resid_pdrop": 0.0,
        "rotary_dim": 32,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.34.1",
        "vocab_size": 51200,
    },
    "phi-2": {
        "_name_or_path": "microsoft/phi-2",
        "activation_function": "gelu_new",
        "architectures": ["PhiForCausalLM"],
        "attn_pdrop": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_phi.PhiConfig",
            "AutoModelForCausalLM": "modeling_phi.PhiForCausalLM",
        },
        "embd_pdrop": 0.0,
        "flash_attn": False,
        "flash_rotary": False,
        "fused_dense": False,
        "img_processor": None,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "phi-msft",
        "n_embd": 2560,
        "n_head": 32,
        "n_head_kv": None,
        "n_inner": None,
        "n_layer": 32,
        "n_positions": 2048,
        "resid_pdrop": 0.1,
        "rotary_dim": 32,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.35.2",
        "vocab_size": 51200,
    },
    "qwen": {
        "architectures": ["QWenLMHeadModel"],
        "auto_map": {
            "AutoConfig": "configuration_qwen.QWenConfig",
            "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel",
        },
        "attn_dropout_prob": 0.0,
        "bf16": False,
        "emb_dropout_prob": 0.0,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "initializer_range": 0.02,
        "kv_channels": 128,
        "layer_norm_epsilon": 1e-06,
        "max_position_embeddings": 8192,
        "model_type": "qwen",
        "no_bias": True,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "rotary_emb_base": 10000,
        "rotary_pct": 1.0,
        "scale_attn_weights": True,
        "seq_length": 8192,
        "tie_word_embeddings": False,
        "tokenizer_class": "QWenTokenizer",
        "transformers_version": "4.32.0",
        "use_cache": True,
        "use_dynamic_ntk": True,
        "use_flash_attn": "auto",
        "use_logn_attn": True,
        "vocab_size": 151936,
    },
    "stablelm_epoch": {
        "architectures": ["StableLMEpochForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_stablelm_epoch.StableLMEpochConfig",
            "AutoModelForCausalLM": "modeling_stablelm_epoch.StableLMEpochForCausalLM",
        },
        "bos_token_id": 100257,
        "eos_token_id": 100257,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 5632,
        "max_position_embeddings": 4096,
        "model_type": "stablelm_epoch",
        "norm_eps": 1e-05,
        "num_attention_heads": 32,
        "num_heads": 32,
        "num_hidden_layers": 24,
        "num_key_value_heads": 32,
        "rope_pct": 0.25,
        "rope_theta": 10000,
        "rotary_scaling_factor": 1.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.36.2",
        "use_cache": True,
        "use_qkv_bias": True,
        "vocab_size": 100352,
    },
}
