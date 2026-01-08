"""
Model-specific configurations for QAT training
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ModelConfig:
    """Model-specific configuration for QAT"""
    
    # Model Architecture
    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    
    # Tokenizer Settings
    vocab_size: int
    max_position_embeddings: int
    
    # Quantization Settings
    target_modules: List[str]
    quantization_bits: int = 4
    group_size: int = 32
    
    # Template Settings
    conversation_template: str
    eos_token: str
    bos_token: str
    pad_token: Optional[str] = None


# Llama 3.2 1B Configuration
LLAMA_3_2_1B_CONFIG = ModelConfig(
    model_type="llama",
    hidden_size=2048,
    num_attention_heads=32,
    num_hidden_layers=16,
    intermediate_size=8192,
    vocab_size=128256,
    max_position_embeddings=131072,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    quantization_bits=4,
    group_size=32,
    conversation_template="llama3",
    eos_token="<|eot_id|>",
    bos_token="<|begin_of_text|>",
    pad_token="<|finetune_right_pad_id|>"
)

# Llama 3.2 3B Configuration
LLAMA_3_2_3B_CONFIG = ModelConfig(
    model_type="llama",
    hidden_size=3072,
    num_attention_heads=24,
    num_hidden_layers=28,
    intermediate_size=8192,
    vocab_size=128256,
    max_position_embeddings=131072,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    quantization_bits=4,
    group_size=32,
    conversation_template="llama3",
    eos_token="<|eot_id|>",
    bos_token="<|begin_of_text|>",
    pad_token="<|finetune_right_pad_id|>"
)

# Configuration registry
MODEL_CONFIGS = {
    "llama-3.2-1b": LLAMA_3_2_1B_CONFIG,
    "llama-3.2-3b": LLAMA_3_2_3B_CONFIG,
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Try to infer from model name
    if "1b" in model_name.lower():
        print(f"Using Llama 3.2 1B config for {model_name}")
        return LLAMA_3_2_1B_CONFIG
    elif "3b" in model_name.lower():
        print(f"Using Llama 3.2 3B config for {model_name}")
        return LLAMA_3_2_3B_CONFIG
    else:
        print(f"Unknown model {model_name}, using default Llama 3.2 1B config")
        return LLAMA_3_2_1B_CONFIG


def get_conversation_templates() -> Dict[str, Dict[str, str]]:
    """Get conversation templates for different models"""
    return {
        "llama3": {
            "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        },
        "default": {
            "system": "<|system|>\n{content}\n",
            "user": "<|user|>\n{content}\n", 
            "assistant": "<|assistant|>\n{content}\n",
        },
        "alpaca": {
            "system": "### System:\n{content}\n\n",
            "user": "### Human: {content}\n",
            "assistant": "### Assistant: {content}\n",
        },
        "vicuna": {
            "system": "SYSTEM: {content}\n",
            "user": "USER: {content}\n",
            "assistant": "ASSISTANT: {content}\n",
        }
    }


def get_mlc_quantization_mapping() -> Dict[str, Any]:
    """Get quantization mapping for MLC-LLM compatibility"""
    return {
        "q4f16_1": {
            "name": "q4f16_1",
            "kind": "group-quant",
            "group_size": 32,
            "quantize_dtype": "int4",
            "storage_dtype": "uint32", 
            "model_dtype": "float16",
            "linear_weight_layout": "NK",
            "quantize_embedding": True,
            "quantize_final_fc": True,
        }
    }