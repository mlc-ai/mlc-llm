import numpy as np
import pytest
import torch
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm import runtime

from mlc_llm.model.qwen3_vl.qwen3_vl_vision import (
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionConfig,
)

def get_test_config():
    return Qwen3VLVisionConfig(
        hidden_size=32,
        spatial_merge_size=2,
        out_hidden_size=64,
        # Other params not needed for merger
        depth=1, hidden_act="gelu", intermediate_size=64, num_heads=4, in_channels=3, patch_size=14, temporal_patch_size=2, num_position_embeddings=100, deepstack_visual_indexes=[]
    )

class PyTorchPatchMerger(torch.nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = torch.nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = torch.nn.GELU()
        self.linear_fc2 = torch.nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x):
        target_size = self.hidden_size
        
        if self.use_postshuffle_norm:
             x = x.view(-1, target_size)
             x = self.norm(x)
        else:
             x = self.norm(x)
        
        x = x.view(-1, target_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

class TVMPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.model = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm)
        
    def forward(self, x: nn.Tensor):
        return self.model(x)

def run_test(use_postshuffle_norm):
    config = get_test_config()
    merge_sq = config.spatial_merge_size ** 2
    
    # Setup inputs
    # If use_postshuffle_norm=False, input is (N, hidden_size)
    # And we expect it to be reshaped to (N/merge_sq, hidden_size*merge_sq)
    # So N must be divisible by merge_sq.
    
    batch = 16 # divisible by 4 (2^2)
    inp_dim = config.hidden_size
    
    np.random.seed(42)
    input_np = np.random.randn(batch, inp_dim).astype("float32")
    
    # PyTorch
    torch_model = PyTorchPatchMerger(config, use_postshuffle_norm)
    torch_in = torch.from_numpy(input_np)
    
    with torch.no_grad():
        torch_out = torch_model(torch_in).numpy()
        
    # TVM
    tvm_model = TVMPatchMerger(config, use_postshuffle_norm)
    mod, params = tvm_model.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor(input_np.shape, "float32")}}
    )
    
    # Weight mapping
    # TVM: model.linear_fc1.weight, model.norm.weight etc.
    # PyTorch: linear_fc1.weight, norm.weight
    
    param_dict = dict(params)
    torch_params = dict(torch_model.named_parameters())
    
    for k in param_dict.keys():
        # k example: model.linear_fc1.weight
        pt_key = k.replace("model.", "")
        if pt_key in torch_params:
            param_dict[k] = runtime.tensor(torch_params[pt_key].detach().numpy())
        else:
            print(f"Warning: {k} not found in torch model")

    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    tvm_out = vm["forward"](runtime.tensor(input_np), *param_dict.values()).numpy()
    
    np.testing.assert_allclose(torch_out, tvm_out, rtol=1e-5, atol=1e-5)
    print(f"PatchMerger test passed (use_postshuffle_norm={use_postshuffle_norm})")

def test_qwen3_vl_patch_merger():
    run_test(use_postshuffle_norm=False)
    run_test(use_postshuffle_norm=True)

if __name__ == "__main__":
    test_qwen3_vl_patch_merger()
