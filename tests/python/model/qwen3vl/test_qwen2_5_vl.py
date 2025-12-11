
import torch
import tvm
import tvm.testing
from tvm import runtime
from tvm.relax.frontend import nn
from tvm import relax
import numpy as np

from mlc_llm.model.qwen3_vl.qwen_2_5_vl import Qwen2_5_VLMLP, Qwen2_5_VLVisionBlock

# Helper for RoPE (needed for Attention inside Block)
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

class Config:
    hidden_size = 64
    intermediate_size = 128
    num_heads = 4
    hidden_act = "silu"

config = Config()

def test_mlp():
    # TVM
    class TVMMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Qwen2_5_VLMLP(config, bias=True)
            
        def forward(self, x):
            return self.model(x)
            
    # PyTorch
    class PyTorchMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
            self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
            self.down_proj = torch.nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
            self.act_fn = torch.nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            
    # Inputs
    b, s, d = 1, 10, config.hidden_size
    input_data = np.random.randn(b, s, d).astype("float32")
    
    tvm_model = TVMMLP()
    torch_model = PyTorchMLP()
    
    mod, params = tvm_model.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor([b, s, d], "float32")}}
    )
    
    # Sync weights
    param_dict = dict(params)
    param_dict["model.gate_proj.weight"] = runtime.tensor(torch_model.gate_proj.weight.detach().numpy())
    param_dict["model.gate_proj.bias"] = runtime.tensor(torch_model.gate_proj.bias.detach().numpy())
    param_dict["model.up_proj.weight"] = runtime.tensor(torch_model.up_proj.weight.detach().numpy())
    param_dict["model.up_proj.bias"] = runtime.tensor(torch_model.up_proj.bias.detach().numpy())
    param_dict["model.down_proj.weight"] = runtime.tensor(torch_model.down_proj.weight.detach().numpy())
    param_dict["model.down_proj.bias"] = runtime.tensor(torch_model.down_proj.bias.detach().numpy())
    
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    tvm_out = vm["forward"](runtime.tensor(input_data), *param_dict.values())
    torch_out = torch_model(torch.from_numpy(input_data)).detach().numpy()
    
    np.testing.assert_allclose(torch_out, tvm_out.numpy(), rtol=1e-5, atol=1e-5)
    print("Qwen2_5_VLMLP test passed!")

def test_vision_block():
    # TVM
    class TVMBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Qwen2_5_VLVisionBlock(config)
            
        def forward(self, x, cu_seqlens, rotary_pos_emb):
            return self.model(x, cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            
    # PyTorch Reference
    class PyTorchBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.RMSNorm(config.hidden_size, eps=1e-6)
            self.norm2 = torch.nn.RMSNorm(config.hidden_size, eps=1e-6)
            
            # Attn
            self.dim = config.hidden_size
            self.num_heads = config.num_heads
            self.head_dim = self.dim // self.num_heads
            self.qkv = torch.nn.Linear(self.dim, self.dim * 3, bias=True)
            self.proj = torch.nn.Linear(self.dim, self.dim)
            self.scaling = self.head_dim**-0.5
            
            # MLP
            self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
            self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
            self.down_proj = torch.nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
            self.act_fn = torch.nn.SiLU()

        def forward(self, hidden_states, rotary_pos_emb=None):
            # Attention block
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
            
            b, s, _ = hidden_states.shape
            qkv = self.qkv(hidden_states)
            qkv = qkv.reshape(b, s, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            
            if rotary_pos_emb is not None:
                freqs = rotary_pos_emb
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) * self.scaling
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(b, s, self.dim)
            out = self.proj(out)
            
            hidden_states = residual + out
            
            # MLP Block
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
            hidden_states = residual + hidden_states
            
            return hidden_states

    # Inputs
    b, s, d = 1, 10, config.hidden_size
    head_dim = d // config.num_heads
    input_data = np.random.randn(b, s, d).astype("float32")
    cu_seqlens_np = np.array([0, s], dtype="int32")
    freqs_np = np.random.randn(s, head_dim // 2).astype("float32")
    
    tvm_model = TVMBlock()
    torch_model = PyTorchBlock()
    
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
    
    def sync_linear(tvm_name, torch_module):
        param_dict[f"{tvm_name}.weight"] = runtime.tensor(torch_module.weight.detach().numpy())
        if torch_module.bias is not None:
            param_dict[f"{tvm_name}.bias"] = runtime.tensor(torch_module.bias.detach().numpy())
            

    # Wait, Qwen2RMSNorm uses nn.Parameter name 'weight'.
    param_dict["model.norm1.weight"] = runtime.tensor(torch_model.norm1.weight.detach().numpy())
    param_dict["model.norm2.weight"] = runtime.tensor(torch_model.norm2.weight.detach().numpy())
    
    sync_linear("model.attn.qkv", torch_model.qkv)
    sync_linear("model.attn.proj", torch_model.proj)
    
    sync_linear("model.mlp.gate_proj", torch_model.gate_proj)
    sync_linear("model.mlp.up_proj", torch_model.up_proj)
    sync_linear("model.mlp.down_proj", torch_model.down_proj)

    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    
    tvm_out = vm["forward"](
        runtime.tensor(input_data), 
        runtime.tensor(cu_seqlens_np),
        runtime.tensor(freqs_np),
        *param_dict.values()
    )
    
    torch_in = torch.from_numpy(input_data)
    rotary_in = torch.from_numpy(freqs_np)
    with torch.no_grad():
        torch_out = torch_model(torch_in, rotary_pos_emb=rotary_in).numpy()
        
    np.testing.assert_allclose(torch_out, tvm_out.numpy(), rtol=1e-5, atol=1e-5)
    print("Qwen2_5_VLVisionBlock test passed!")

if __name__ == "__main__":
    test_mlp()
    test_vision_block()
