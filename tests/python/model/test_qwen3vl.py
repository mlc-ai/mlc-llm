
# pylint: disable=invalid-name,missing-docstring
import json
import os
import pytest
from tvm import relax

from mlc_llm.model import MODELS
from mlc_llm.model.qwen3_vl.qwen3_vl_config import Qwen3VLConfig

# Directory containing the test config
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "qwen3_vl_2b_instruct")
CONFIG_PATH = os.path.join(TEST_DATA_DIR, "config.json")

@pytest.mark.skipif(not os.path.exists(CONFIG_PATH), reason="Test config not found")
def test_qwen3vl_creation():
    # Load config from file
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    # Instantiate Qwen3VLConfig
    config = Qwen3VLConfig.from_dict(config_dict)
    
    # Get model info and class
    model_info = MODELS["qwen3_vl"]
    model_class = model_info.model
    
    # Create model
    model = model_class(config)
    
    # Export to TVM to verify structure and creation
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),
    )
    
    # Basic assertions
    import tvm
    assert isinstance(mod, tvm.IRModule)

    assert len(named_params) > 0
    
    # Verify some parameter shapes/types if needed, or just that it didn't crash
    print("Qwen3-VL Model created successfully.")
    for name, param in named_params:
         print(f"{name}: {param.shape} {param.dtype}")

if __name__ == "__main__":
    test_qwen3vl_creation()
