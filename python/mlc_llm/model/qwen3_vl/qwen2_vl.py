import math
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.op import wrap_nested
from tvm import relax as rx

def _wrap_op(f, *args):
    args = [x._expr if isinstance(x, Tensor) else x for x in args]
    return wrap_nested(f(*args), name=f.__name__)

def op_cos(x): return _wrap_op(rx.op.cos, x)
def op_sin(x): return _wrap_op(rx.op.sin, x)
def op_power(a, b): return _wrap_op(rx.op.power, a, b)


from typing import Optional, Tuple

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]

        self.proj = nn.Conv3D(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = op.reshape(
            hidden_states, 
            (-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        )
            
        hidden_states = self.proj(hidden_states)
        hidden_states = op.reshape(hidden_states, (-1, self.embed_dim))
        return hidden_states



class VisionRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> Tensor:
        theta_const = rx.const(self.theta, "float32")
        inv_freq = op.divide(Tensor(_expr=rx.const(1.0, "float32")), op_power(theta_const, (op.arange(0, self.dim, 2, dtype="float32") / self.dim)))
        
        seq = op.arange(0, seqlen, dtype="float32")
        
        seq = op.reshape(seq, (seqlen, 1))
        inv_freq = op.reshape(inv_freq, (1, self.dim // 2))
        
        freqs = seq * inv_freq
        return freqs


class VisionAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        rotary_pos_emb: Optional[Tensor] = None,
        position_embeddings: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tensor:


        b, s, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = op.reshape(qkv, (b, s, 3, self.num_heads, self.head_dim))
        
        q, k, v = op.split(qkv, 3, axis=2)
        q = op.squeeze(q, axis=2)
        k = op.squeeze(k, axis=2)
        v = op.squeeze(v, axis=2)
        
        # Apply RoPE if provided
        if rotary_pos_emb is not None:
            freqs = rotary_pos_emb
            cos = op_cos(freqs)
            sin = op_sin(freqs)
            
            # Reshape for broadcasting: (1, s, 1, d/2)
            cos = op.reshape(cos, (1, s, 1, self.head_dim // 2))
            sin = op.reshape(sin, (1, s, 1, self.head_dim // 2))
            
            # Use repeat to match head_dim
            cos = op.concat([cos, cos], dim=-1)
            sin = op.concat([sin, sin], dim=-1)
            
            def rotate_half(x):
                x1, x2 = op.split(x, 2, axis=-1) # split last dim
                return op.concat([op.negative(x2), x1], dim=-1)
            
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            
        # Attention
        q = op.permute_dims(q, (0, 2, 1, 3)) # (b, h, s, d)
        k = op.permute_dims(k, (0, 2, 1, 3)) # (b, h, s, d)
        v = op.permute_dims(v, (0, 2, 1, 3)) # (b, h, s, d)
        
        # k.T -> (b, h, d, s)
        k_t = op.permute_dims(k, (0, 1, 3, 2))
        
        attn_weights = op.matmul(q, k_t) # (b, h, s, s)
        attn_weights = attn_weights * self.scaling
        
        attn_weights = op.softmax(attn_weights, axis=-1)
        
        attn_output = op.matmul(attn_weights, v) # (b, h, s, d)
        
        # Transpose back: (b, s, h, d)
        attn_output = op.permute_dims(attn_output, (0, 2, 1, 3))
        
        # Reshape to (b, s, dim)
        attn_output = op.reshape(attn_output, (b, s, self.dim))
        
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        
        self.weight = nn.Parameter((hidden_size,), dtype="float32")
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        return op.rms_norm(hidden_states, self.weight, axes=-1, epsilon=self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"



class Qwen2VLModel(nn.Module):
    base_model_prefix = "model"
    accepts_loss_kwargs = False

    # expects qwen2vlconfig object
    def __init__(self, config):
        self.rope_deltas = None  # cache rope_deltas here

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(
        self,
        input_ids: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        video_grid_thw: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError