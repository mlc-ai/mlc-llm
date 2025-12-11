
import math
import numpy as np
import pytest
import torch
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm import runtime
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.op import wrap_nested
from tvm.relax.op import strided_slice as relax_strided_slice

def op_strided_slice(x, axes, begin, end):
    return wrap_nested(relax_strided_slice(x._expr, axes, begin, end), name="strided_slice")

from mlc_llm.model.qwen3_vl.qwen3_vl_vision import (
    Qwen3VLVisionModel,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionBlock,
    Qwen3VLVisionConfig,
)

# Helper function to get valid config for testing
def get_test_config():
    return Qwen3VLVisionConfig(
        depth=1,
        hidden_size=32,
        hidden_act="gelu",
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=100,
        deepstack_visual_indexes=[],
    )

# TVM Wrapper
class TVMRotPosEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3VLVisionModel(config)
        
    def forward(self, grid_thw: nn.Tensor):
        # Compute total tokens for the call
        # grid_thw is (N, 3)
        # We can use op.sum(prod) logic or slice
        t = op_strided_slice(grid_thw, axes=[1], begin=[0], end=[1])
        h = op_strided_slice(grid_thw, axes=[1], begin=[1], end=[2])
        w = op_strided_slice(grid_thw, axes=[1], begin=[2], end=[3])
        total_tokens = nn.op.sum(t * h * w)
        return self.model.rot_pos_emb(grid_thw, total_tokens)

# PyTorch Helper
class PyTorchVisionRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

# Define EXACT PyTorch Code from comments
class Qwen3VLExactPyTorchRotary(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        head_dim = config.hidden_size // config.num_heads
        # Use PyTorch implementation of Rotary
        self.rotary_pos_emb = PyTorchVisionRotaryEmbedding(head_dim // 2)

    def forward(self, grid_thw):
        # EXACT CODE FROM COMMENT
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

def test_rot_pos_emb():
    # Configuration
    config = get_test_config()
    dim = config.hidden_size // config.num_heads // 2 # Head dim // 2
    
    # Setup
    torch.manual_seed(42)
    torch_model = Qwen3VLExactPyTorchRotary(config)
    
    # Create random grid_thw
    # dim=3 (t, h, w)
    # Ensure h, w are divisible by spatial_merge_size (2)
    grid_thw_np = np.array([
        [1, 14, 14],
        [2, 28, 28]
    ], dtype="int64")
    
    torch_input = torch.from_numpy(grid_thw_np)
    
    # Run PyTorch
    with torch.no_grad():
        torch_output = torch_model(torch_input).numpy()
        
    # Run TVM
    tvm_model = TVMRotPosEmb(config)
    
    # Export TVM
    mod, params = tvm_model.export_tvm(
        spec={
            "forward": {
                "grid_thw": nn.spec.Tensor(grid_thw_np.shape, "int64")
            }
        }
    )
    
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    # Run
    # params is a list of tuples, convert to dict
    param_dict = dict(params)
    tvm_output = vm["forward"](runtime.tensor(grid_thw_np), *param_dict.values()).numpy()
    
    # Verify
    np.testing.assert_allclose(torch_output, tvm_output, rtol=1e-5, atol=1e-5)
    print("RotPosEmb test passed!")

def test_rot_pos_emb_shapes():
    # Test with different shapes to ensure dynamic handling works
    config = get_test_config()
    
    # TVM Model
    tvm_model = TVMRotPosEmb(config)
    mod, params = tvm_model.export_tvm(
        spec={"forward": {"grid_thw": nn.spec.Tensor(["N", 3], "int64")}}
    )
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    param_dict = dict(params)

    test_cases = [
        # (N, t, h, w) - Total tokens will be sum(t*h*w)
        [(1, 14, 14), (1, 28, 28)], # Single image split into two (unlikely but possible grid input format?)
        [(1, 14, 14)], # Standard single image
        [(2, 14, 14), (2, 28, 28), (1, 56, 56)], # Mixed batch
    ]
    
    # PyTorch Reference Helper
    head_dim = config.hidden_size // config.num_heads
    # Needs to match Qwen3VLExactPyTorchRotary logic
    class RefModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.spatial_merge_size = config.spatial_merge_size
            self.model = Qwen3VLExactPyTorchRotary(config)
        def forward(self, grid_thw):
            return self.model(grid_thw)

    ref_model = RefModel(config)

    for case_idx, shapes in enumerate(test_cases):
        # Construct grid_thw
        grid_data = []
        for (t, h, w) in shapes:
            grid_data.append([t, h, w])
        grid_np = np.array(grid_data, dtype="int64")
        
        # Run Reference
        with torch.no_grad():
            ref_out = ref_model(torch.from_numpy(grid_np)).numpy()
            
        # Run TVM
        tvm_out = vm["forward"](runtime.tensor(grid_np), *param_dict.values()).numpy()
        
        np.testing.assert_allclose(ref_out, tvm_out, rtol=1e-5, atol=1e-5)
        print(f"Shape test case {case_idx} passed with shape {grid_np.shape} -> output {tvm_out.shape}")

if __name__ == "__main__":
    test_rot_pos_emb()
    test_rot_pos_emb_shapes()
