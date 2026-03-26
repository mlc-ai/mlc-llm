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
from tvm import runtime

def op_strided_slice(x, axes, begin, end):
    return wrap_nested(relax_strided_slice(x._expr, axes, begin, end), name="strided_slice")

from mlc_llm.model.qwen3_vl.qwen3_vl_vision import (
    Qwen3VLVisionModel,
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
        num_position_embeddings=100, # 10x10 grid
        deepstack_visual_indexes=[],
    )

class PyTorchReference(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.pos_embed = torch.nn.Embedding(config.num_position_embeddings, config.hidden_size)

    def forward(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

# TVM Wrapper
class TVMFastInterpolate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3VLVisionModel(config)
        
    def forward(self, grid_thw: nn.Tensor):
        # Calculate total tokens for output shape
        t = op_strided_slice(grid_thw, axes=[1], begin=[0], end=[1])
        h = op_strided_slice(grid_thw, axes=[1], begin=[1], end=[2])
        w = op_strided_slice(grid_thw, axes=[1], begin=[2], end=[3])
        total_tokens = nn.op.sum(t * h * w)
        return self.model.fast_pos_embed_interpolate(grid_thw, total_tokens)

def test_fast_pos_embed_interpolate():
    config = get_test_config()
    
    # Setup PyTorch
    torch.manual_seed(42)
    torch_ref = PyTorchReference(config)
    
    # Setup inputs
    # spatial_merge_size = 2, so h, w must be divisible by 2
    # num_position_embeddings = 100 -> num_grid_per_side = 10
    
    # Case 1: Single image
    grid_thw_np = np.array([
        [1, 14, 14],
        [2, 28, 28] 
    ], dtype="int64")
    
    torch_input = torch.from_numpy(grid_thw_np)
    
    # Run PyTorch
    with torch.no_grad():
        torch_out = torch_ref(torch_input).numpy()
        
    # Setup TVM
    tvm_model = TVMFastInterpolate(config)
    
    # Load weights from PyTorch model to TVM model
    # The weight name in Qwen3VLVisionModel is pos_embed.weight
    # But in TVMFastInterpolate it's model.pos_embed.weight
    
    # Export TVM
    mod, params = tvm_model.export_tvm(
        spec={
            "forward": {
                "grid_thw": nn.spec.Tensor(grid_thw_np.shape, "int64")
            }
        }
    )
    
    # Update params with PyTorch weights
    param_dict = dict(params)
    # Mapping
    # model.pos_embed.weight -> torch_ref.pos_embed.weight
    # Find the key for pos_embed weight
    pos_embed_key = None
    for k in param_dict.keys():
        if "pos_embed" in k:
            pos_embed_key = k
            break
            
    if pos_embed_key:
        param_dict[pos_embed_key] = runtime.tensor(torch_ref.pos_embed.weight.detach().numpy())
    else:
        print("Warning: pos_embed weight not found in params")

    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    tvm_out = vm["forward"](runtime.tensor(grid_thw_np), *param_dict.values()).numpy()
    
    np.testing.assert_allclose(torch_out, tvm_out, rtol=1e-5, atol=1e-5)
    print("FastPosEmbedInterpolate test passed!")

if __name__ == "__main__":
    test_fast_pos_embed_interpolate()
