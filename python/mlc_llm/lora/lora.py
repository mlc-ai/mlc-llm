from typing import Dict
from pathlib import Path
import logging
from tvm import tir
from tvm.relax.frontend.nn import Object, Module, ModuleList, Linear
from mlc_llm.quantization.group_quantization import GroupQuantizeLinear
from mlc_llm.quantization.awq_quantization import AWQQuantizeLinear
from mlc_llm.quantization.ft_quantization import FTQuantizeLinear
from mlc_llm.quantization.per_tensor_quantization import PerTensorQuantizeLinear
from mlc_llm.support.config import ConfigBase
from mlc_llm.lora.lora_config import LoRAConfig
from mlc_llm.lora.lora_layer import LinearLoraA, LinearLoraB
from mlc_llm.lora.backend.tir_backend import TIRLoraBackend


logger = logging.getLogger(__name__)


def get_module_name(base_model: Module, name):
    if hasattr(base_model, 'get_module_name'):
        return base_model.get_module_name(name)
    # Fallback solution of mapping from config module name to module name in model class.
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }
    return params_mapping.get(name, name)


def get_q_kv_output_dim(base_model: Module, model_config: ConfigBase):
    if hasattr(base_model, 'get_q_kv_output_dim'):
        return base_model.get_q_kv_output_dim()
    # Fallback solution of get_q_kv_output_dim for different modules
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    return model_config.num_attention_heads * model_config.head_dim, model_config.num_key_value_heads * model_config.head_dim


def get_stacked_multiply(module_name):
    stacked_rank = {
        "qkv_proj": 3,
        "gate_up_proj": 2,
    }
    return stacked_rank.get(module_name, 1)


def set_lora(base_model: Module, model_config: ConfigBase, lora_paths: Dict[str, Path], model_dtype: str):
    max_loras_per_batch = tir.Var("max_loras_per_batch", dtype="int64")
    # Only tir_lora_backend exists currently
    lora_backend = TIRLoraBackend()
    # Use the first lora to construct lora_layer
    lora_config = LoRAConfig(next(iter(lora_paths.values())))

    def set_lora_layer(module):
        if isinstance(module, ModuleList):
            for subitem in module:
                set_lora_layer(subitem)
            return
        for name, item in module.__dict__.items():
            if isinstance(item, (Linear, GroupQuantizeLinear, AWQQuantizeLinear, FTQuantizeLinear, PerTensorQuantizeLinear)):
                for hf_module in lora_config.target_modules:
                    mlc_name = get_module_name(module, hf_module)
                    if mlc_name in name:
                        lora_stacked_num = get_stacked_multiply(mlc_name)
                        q_output_dim, kv_output_dim = get_q_kv_output_dim(module, model_config)
                        item.lora_A = LinearLoraA(
                            in_features=item.in_features,
                            out_features=lora_stacked_num * lora_config.r,
                            max_loras_per_batch=max_loras_per_batch,
                            lora_backend=lora_backend,
                            layer_index=lora_config.global_layer_index,
                            max_batch_size=model_config.max_batch_size,
                            out_dtype=model_dtype,
                        )
                        item.lora_B = LinearLoraB(
                            in_features=lora_config.r,
                            out_features=item.out_features,
                            stacked_num=lora_stacked_num,
                            lora_module_name=mlc_name,
                            lora_backend=lora_backend,
                            max_loras_per_batch=max_loras_per_batch,
                            q_output_dim=q_output_dim,
                            kv_output_dim=kv_output_dim,
                            out_dtype=model_dtype,
                        )
                        lora_config.global_layer_index += 1
                        break
            elif isinstance(item, Module) or isinstance(item, ModuleList):
                set_lora_layer(item)
    
    set_lora_layer(base_model)