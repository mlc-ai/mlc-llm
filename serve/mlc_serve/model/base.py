from dataclasses import dataclass
from typing import Optional
import os 
import json
import inspect

# TODO(@sunggg): consider transition to something like Pydantic
@dataclass
class ModelArtifactConfig:
    model_artifact_path: Optional[str] = None
    num_shards: Optional[int] = None
    quantization: Optional[str] = None
    model_type: Optional[str] = None
    library_name: Optional[str] = None
    max_context_length: Optional[int] = None
    vocab_size: Optional[int] = None
    sliding_window: Optional[int] = None
    build_options: Optional[str] = None
    num_key_value_heads: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    hidden_size: Optional[int] = None

    @classmethod
    def _from_json(config_cls, json_obj: dict):
        return config_cls(
            **{
                k: v
                for k, v in json_obj.items()
                if k in inspect.signature(config_cls).parameters
            }
        )

def get_model_artifact_config(model_artifact_path):
    json_object = {"model_artifact_path": model_artifact_path}
    for config_file_name in [ 
        "build_config.json",
        "model/mlc-model-config.json"
    ]:
        config_file_path = os.path.join(model_artifact_path, config_file_name)
        assert os.path.exists(config_file_path), f"{config_file_path} should exist. Did you build with `--enable-batching`?"
        with open(config_file_path, mode="rt", encoding="utf-8") as f:
            json_object.update(json.load(f))
    
    return ModelArtifactConfig._from_json(json_object)
    