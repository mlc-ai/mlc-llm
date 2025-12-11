import numpy as np
import pytest
import torch
import torch.nn.functional as F
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm import runtime

from mlc_llm.model.qwen3_vl.qwen3_vl_vision import (
    Qwen3VLVisionModel,
    Qwen3VLVisionConfig,
)
from mlc_llm.model.qwen3_vl.qwen2_vl import PatchEmbed, VisionRotaryEmbedding, VisionAttention
from mlc_llm.model.qwen3_vl.qwen_2_5_vl import Qwen2_5_VLVisionBlock

from test_qwen3_vl_interpolate import PyTorchReference as PyTorchFastPosEmbed
from test_qwen3_vl_rot_pos_emb import Qwen3VLExactPyTorchRotary

# Replicate PyTorch Reference Classes locally to ensure exact behavior matching
class PyTorchVisionAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = torch.nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = torch.nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states, rotary_pos_emb=None, **kwargs):
        b, s, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        if rotary_pos_emb is not None:
             # rotary_pos_emb: (total_tokens, dim//2)
             cos_sin = rotary_pos_emb
             cos, sin = torch.cos(cos_sin), torch.sin(cos_sin)
             cos = cos.unsqueeze(0).unsqueeze(2) # (1, s, 1, d/2)
             sin = sin.unsqueeze(0).unsqueeze(2)
             
             # Repeat to match head_dim
             cos = torch.cat([cos, cos], dim=-1)
             sin = torch.cat([sin, sin], dim=-1)
             
             # Apply
             def apply_rotary_pos_emb(x, cos, sin):
                 x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                 return (x * cos) + (torch.cat([-x2, x1], dim=-1) * sin)
                 
             q = apply_rotary_pos_emb(q, cos, sin)
             k = apply_rotary_pos_emb(k, cos, sin)
        
        q = q.transpose(1, 2) # (b, h, s, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, self.dim)
        out = self.proj(out)
        return out

class PyTorchMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_fc2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = torch.nn.GELU()
    
    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

class PyTorchVisionBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = PyTorchVisionAttention(config)
        self.mlp = PyTorchMLP(config)
        
    def forward(self, hidden_states, rotary_pos_emb=None, **kwargs):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_pos_emb=rotary_pos_emb, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

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
        
        # Consistent with verified logic in PatchMerger test
        if self.use_postshuffle_norm:
             x = x.reshape(-1, self.hidden_size)
             x = self.norm(x)
        else:
             x = self.norm(x)
             x = x.reshape(-1, self.hidden_size)
        
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

class PyTorchReferenceModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        
        # Patch Embed
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.patch_embed_proj = torch.nn.Conv3d(
            config.in_channels, config.hidden_size, kernel_size=kernel_size, stride=kernel_size, bias=True
        )
        
        # Real Implementations
        self.fast_pos_embed_model = PyTorchFastPosEmbed(config)
        self.rot_pos_emb_model = Qwen3VLExactPyTorchRotary(config)
        
        # Blocks
        self.blocks = torch.nn.ModuleList([PyTorchVisionBlock(config) for _ in range(config.depth)])
        
        # Mergers
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.merger = PyTorchPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_merger_list = torch.nn.ModuleList([
            PyTorchPatchMerger(config, use_postshuffle_norm=True) for _ in range(len(config.deepstack_visual_indexes))
        ])

    def forward(self, hidden_states, grid_thw):
        # 1. Patch Embed
        x = self.patch_embed_proj(hidden_states) # (N, dim, t, h, w)
        x = x.flatten(2).transpose(1, 2).flatten(0, 1) # (N*t*h*w, dim)
        
        # 2. Fast Pos Embed
        pos = self.fast_pos_embed_model(grid_thw)
        x = x + pos
        
        # 3. RoPE
        rot = self.rot_pos_emb_model(grid_thw)
        
        # 4. Blocks (Reshape to 3D for attention)
        # We need batch dim for attention. Here total batch is 1 (seq of mixed images).
        # x is (total_tokens, dim).
        x = x.unsqueeze(0) # (1, total_tokens, dim)
        
        deepstacks = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rotary_pos_emb=rot)
            if i in self.deepstack_visual_indexes:
                 idx = self.deepstack_visual_indexes.index(i)
                 deepstacks.append(self.deepstack_merger_list[idx](x))
                 
        x = self.merger(x)
        return x, deepstacks

class TVMModelWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3VLVisionModel(config)
        
    def forward(self, hidden_states, grid_thw):
        return self.model(hidden_states, grid_thw)

def test_model_flow():
    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=16,
        in_channels=3,
        num_position_embeddings=100,
        deepstack_visual_indexes=[0],
        hidden_act="gelu",
        intermediate_size=64
    )
    
    t, h_patches, w_patches = 1, 2, 2
    grid_thw_np = np.array([[t, h_patches, w_patches]], dtype="int64")
    input_pixels_np = np.random.randn(1, 3, 1, 4, 4).astype("float32")
    
    # 1. Run PyTorch
    torch.manual_seed(42)
    torch_model = PyTorchReferenceModel(config)
    torch_pixels = torch.from_numpy(input_pixels_np)
    torch_grid = torch.from_numpy(grid_thw_np)
    
    with torch.no_grad():
        torch_out, torch_deep = torch_model(torch_pixels, torch_grid)
        
    # 2. Run TVM
    tvm_model = TVMModelWrapper(config)
    
    # Export
    mod, params = tvm_model.export_tvm(
        spec={
            "forward": {
                "hidden_states": nn.spec.Tensor(input_pixels_np.shape, "float32"),
                "grid_thw": nn.spec.Tensor(grid_thw_np.shape, "int64")
            }
        }
    )
    
    # Map weights
    param_dict = dict(params)
    torch_params = dict(torch_model.named_parameters())
    
    # Track updated keys
    updated_keys = set()
    
    for k in param_dict.keys():
        pt_key = k.replace("model.", "") # TVM wrapper prefix
        
        # Mappings
        if "patch_embed" in k:
            pt_key = k.replace("model.patch_embed.proj", "patch_embed_proj")
        elif "rotary_pos_emb" in k:
            continue
        elif "pos_embed" in k:
             # TVM: pos_embed.weight
             # PyTorch: fast_pos_embed_model.pos_embed.weight
             pt_key = "fast_pos_embed_model.pos_embed.weight"
        
        # Blocks match directly often
        
        if pt_key in torch_params:
            param_dict[k] = runtime.tensor(torch_params[pt_key].detach().numpy())
            updated_keys.add(k)
        else:
             print(f"WARNING: Key {k} (mapped to {pt_key}) not found in PyTorch model")
            
    # Manually ensure critical weights and mark updated
    if "model.pos_embed.weight" in param_dict and "model.pos_embed.weight" not in updated_keys:
        param_dict["model.pos_embed.weight"] = runtime.tensor(torch_model.fast_pos_embed_model.pos_embed.weight.detach().numpy())
        updated_keys.add("model.pos_embed.weight")
    
    if "model.patch_embed.proj.weight" in param_dict and "model.patch_embed.proj.weight" not in updated_keys:
         param_dict["model.patch_embed.proj.weight"] = runtime.tensor(torch_model.patch_embed_proj.weight.detach().numpy())
         updated_keys.add("model.patch_embed.proj.weight")
    
    if "model.patch_embed.proj.bias" in param_dict and "model.patch_embed.proj.bias" not in updated_keys:
         param_dict["model.patch_embed.proj.bias"] = runtime.tensor(torch_model.patch_embed_proj.bias.detach().numpy())
         updated_keys.add("model.patch_embed.proj.bias")
         
    # Check if all params updated
    for k in param_dict.keys():
        if k not in updated_keys and "rotary_pos_emb" not in k:
             raise ValueError(f"Parameter {k} was not updated from PyTorch model!")
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    outputs = vm["forward"](runtime.tensor(input_pixels_np), runtime.tensor(grid_thw_np), *param_dict.values())
    
    # Verify return type
    if hasattr(outputs, "dtype"):
        outputs = [outputs]
    else:
        outputs = [outputs[i] for i in range(len(outputs))]
        
    # outputs[0] -> main output
    # outputs[1] -> deepstack output
    
    tvm_main = outputs[0].numpy()
    tvm_deep = outputs[1].numpy()
    
    # Compare Main
    np.testing.assert_allclose(torch_out.numpy(), tvm_main, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(torch_deep[0].numpy(), tvm_deep, rtol=1e-5, atol=1e-5)
    
    print("Model Flow Test Passed with Numerical Verification!")

if __name__ == "__main__":
    test_model_flow()
