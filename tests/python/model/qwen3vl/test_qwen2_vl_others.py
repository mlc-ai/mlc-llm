
import torch
import tvm
import tvm.testing
from tvm import runtime
from tvm.relax.frontend import nn
from tvm import relax
import numpy as np

from mlc_llm.model.qwen3_vl.qwen2_vl import VisionAttention, Qwen2RMSNorm

def test_qwen2_rms_norm():
    hidden_size = 32
    eps = 1e-6
    
    # TVM
    class TVMNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Qwen2RMSNorm(hidden_size, eps=eps)
            
        def forward(self, x):
            return self.model(x)

    # PyTorch
    class PyTorchNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

    input_data = np.random.randn(2, 10, hidden_size).astype("float32")
    
    tvm_model = TVMNorm()
    # Export
    mod, params = tvm_model.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor(input_data.shape, "float32")}}
    )
    
    # Initialize PyTorch
    torch_model = PyTorchNorm(hidden_size, eps)
    
    # Copy weights (ones)
    weight_np = torch_model.weight.detach().numpy()
    param_dict = dict(params)
    param_dict["model.weight"] = runtime.tensor(weight_np)
    
    # Build
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    # Run TVM
    tvm_out = vm["forward"](runtime.tensor(input_data), *param_dict.values())
    
    # Run PyTorch
    torch_out = torch_model(torch.from_numpy(input_data)).detach().numpy()
    
    np.testing.assert_allclose(torch_out, tvm_out.numpy(), rtol=1e-5, atol=1e-5)
    print("Qwen2RMSNorm test passed!")

def test_vision_attention():
    # Mock config
    class Config:
        hidden_size = 64
        num_heads = 4
    
    config = Config()
    
    # --- Helper Definitions (Exact copy from rope_utils.py for the test) ---
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_vision(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q, k = q.float(), k.float()
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        q_embed = q_embed.to(orig_q_dtype)
        k_embed = k_embed.to(orig_k_dtype)
        return q_embed, k_embed
    # -----------------------------------------------------

    # TVM Wrapper
    class TVMAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = VisionAttention(config)
            
        # We match signature of VisionAttention.forward
        def forward(self, x, cu_seqlens, rotary_pos_emb):
            return self.model(x, cu_seqlens, rotary_pos_emb=rotary_pos_emb)

    # PyTorch Reference
    class PyTorchAttn(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dim = config.hidden_size
            self.num_heads = config.num_heads
            self.head_dim = self.dim // self.num_heads
            self.qkv = torch.nn.Linear(self.dim, self.dim * 3, bias=True)
            self.proj = torch.nn.Linear(self.dim, self.dim)
            self.scaling = self.head_dim**-0.5

        def forward(self, hidden_states, rotary_pos_emb=None):
            # rotary_pos_emb here corresponds to 'freqs' passed to the TVM model
            # But apply_rotary_pos_emb_vision needs cos, sin.
            # So in this reference model, we will compute cos/sin from the passed freqs 
            # to mimic exactly what happens.
            
            b, s, _ = hidden_states.shape
            qkv = self.qkv(hidden_states)
            qkv = qkv.reshape(b, s, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            
            # apply_rotary_pos_emb_vision expects q, k in shape (batch, seqlen, num_heads, head_dim)
            # which matches current q, k shape.
            
            if rotary_pos_emb is not None:
                # rotary_pos_emb is 'freqs' (seqlen, head_dim) or (seqlen, head_dim/2) ?
                # VisionRotaryEmbedding returns freqs as implicit angles.
                # In PyTorch VisionRotaryEmbedding (from qwen2_vl.py comments):
                # freqs = torch.outer(seq, self.inv_freq) 
                # Shape: (seqlen, dim/2). 
                # But typically for RoPE we duplicate them to (seqlen, dim) for cos/sin?
                # Or apply_rotary_pos_emb_vision handles it?
                
                # Look at apply_rotary_pos_emb_vision:
                # q_embed = (q * cos) + ...
                # q: (b, s, h, d)
                # cos: unsqueezed to (s, 1, d).
                # So cos must have last dim 'd' (head_dim).
                
                # But freqs from outer product is head_dim/2.
                # So we must repeat/cat to get full head_dim.
                
                freqs = rotary_pos_emb
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
                
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            
            q = q.transpose(1, 2) # (b, h, s, d)
            k = k.transpose(1, 2) # (b, h, s, d)
            v = v.transpose(1, 2) # (b, h, s, d)
            
            attn = (q @ k.transpose(-2, -1)) * self.scaling
            attn = attn.softmax(dim=-1)
            out = attn @ v
            
            out = out.transpose(1, 2).reshape(b, s, self.dim)
            out = self.proj(out)
            return out

    # Input Setup
    b, s, d = 1, 10, config.hidden_size
    head_dim = d // config.num_heads
    input_data = np.random.randn(b, s, d).astype("float32")
    cu_seqlens_np = np.array([0, s], dtype="int32")
    
    # Generate constant freqs for testing RoPE
    # shape: (s, head_dim // 2)
    freqs_np = np.random.randn(s, head_dim // 2).astype("float32")
    
    tvm_model = TVMAttn()
    torch_model = PyTorchAttn(config)
    
    # Export TVM
    mod, params = tvm_model.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([b, s, d], "float32"),
                "cu_seqlens": nn.spec.Tensor([2], "int32"),
                "rotary_pos_emb": nn.spec.Tensor([s, head_dim // 2], "float32")
            }
        }
    )
    
    # Sync weights
    param_dict = dict(params)
    qkv_w = torch_model.qkv.weight.detach().numpy()
    qkv_b = torch_model.qkv.bias.detach().numpy()
    proj_w = torch_model.proj.weight.detach().numpy()
    proj_b = torch_model.proj.bias.detach().numpy()
    
    param_dict["model.qkv.weight"] = runtime.tensor(qkv_w)
    param_dict["model.qkv.bias"] = runtime.tensor(qkv_b)
    param_dict["model.proj.weight"] = runtime.tensor(proj_w)
    param_dict["model.proj.bias"] = runtime.tensor(proj_b)
    
    # Build
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    # Run PyTorch
    torch_in = torch.from_numpy(input_data)
    rotary_in = torch.from_numpy(freqs_np)
    with torch.no_grad():
        torch_out = torch_model(torch_in, rotary_pos_emb=rotary_in).numpy()
    
    # Run TVM
    tvm_out = vm["forward"](
        runtime.tensor(input_data), 
        runtime.tensor(cu_seqlens_np),
        runtime.tensor(freqs_np),
        *param_dict.values()
    )
    
    np.testing.assert_allclose(torch_out, tvm_out.numpy(), rtol=1e-5, atol=1e-5)
    print("VisionAttention test passed!")


if __name__ == "__main__":
    test_qwen2_rms_norm()
    test_vision_attention()
