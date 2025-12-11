
import torch
import tvm
import tvm.testing
from tvm import runtime
from tvm.relax.frontend import nn
from tvm import relax
import numpy as np

from mlc_llm.model.qwen3_vl.qwen2_vl import PatchEmbed, VisionRotaryEmbedding

def test_patch_embed():
    # Configuration
    patch_size = 14
    temporal_patch_size = 2
    in_channels = 3
    embed_dim = 32 # Small dim for testing
    
    # TVM Model
    class TVMPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = PatchEmbed(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim
            )
            
        def forward(self, x):
            return self.model(x)

    # PyTorch Reference
    class PyTorchPatchEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            target_kernel_size = [temporal_patch_size, patch_size, patch_size]
            self.proj = torch.nn.Conv3d(
                in_channels, 
                embed_dim, 
                kernel_size=target_kernel_size, 
                stride=target_kernel_size, 
                bias=False
            )
            self.in_channels = in_channels
            self.temporal_patch_size = temporal_patch_size
            self.patch_size = patch_size
            self.embed_dim = embed_dim

        def forward(self, hidden_states):
            target_dtype = self.proj.weight.dtype
            hidden_states = hidden_states.view(
                -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            )
            hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
            return hidden_states

    # Inputs
    num_patches = 5
    input_flat_size = in_channels * temporal_patch_size * patch_size * patch_size
    input_data_np = np.random.randn(num_patches, input_flat_size).astype("float32")
    
    # Setup PyTorch
    torch_model = PyTorchPatchEmbed()
    torch_input = torch.from_numpy(input_data_np)
    
    # Setup TVM
    tvm_model = TVMPatchEmbed()
    
    # Run PyTorch
    with torch.no_grad():
        torch_output = torch_model(torch_input).numpy()
        
    # Run TVM
    mod, params = tvm_model.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([num_patches, input_flat_size], "float32")
            }
        }
    )
    
    param_dict = dict(params)
    w_key = list(param_dict.keys())[0]
    weight_np = torch_model.proj.weight.detach().numpy()
    param_dict[w_key] = runtime.tensor(weight_np)
    
    # Build
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    # Run
    tvm_output = vm["forward"](runtime.tensor(input_data_np), *param_dict.values())
    
    # Verify
    np.testing.assert_allclose(torch_output, tvm_output.numpy(), rtol=1e-5, atol=1e-5)
    print("PatchEmbed test passed!")


def test_vision_rotary_embedding():
    dim = 32
    theta = 10000.0
    seqlen = 10
    
    # TVM
    class TVMRotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = VisionRotaryEmbedding(dim, theta)
            
        def forward(self, seqlen: int):
            return self.model(seqlen)
            
    # PyTorch
    class PyTorchRotary(torch.nn.Module):
        def __init__(self, dim, theta=10000.0):
            super().__init__()
            self.dim = dim
            self.theta = theta
            self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            
        def forward(self, seqlen):
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            return freqs

    torch_model = PyTorchRotary(dim, theta)
    torch_out = torch_model(seqlen).numpy()
    
    tvm_model = TVMRotary()
    
    # We pass seqlen as int.
    mod, params = tvm_model.export_tvm(
        spec={
            "forward": {
                "seqlen": int
            }
        }
    )
    
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    # Run. 
    # For int spec, we generally pass ShapeTuple for the VM to unpack if it expects scalar args mapped from shape.
    tvm_out = vm["forward"](tvm.runtime.ShapeTuple([seqlen]))
    
    np.testing.assert_allclose(torch_out, tvm_out.numpy(), rtol=1e-5, atol=1e-5)
    print("VisionRotaryEmbedding test passed!")

if __name__ == "__main__":
    test_patch_embed()
    test_vision_rotary_embedding()
