from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

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

        # TODO - i am assuming tvm has the same conv3d as pytorch
        self.proj = nn.Conv3D(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:

        '''
        TODO - translate pytorch to tvm
        '''

        raise NotImplementedError
        # target_dtype = self.proj.weight.dtype
        # hidden_states = hidden_states.view(
        #     -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        # )
        # hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        # return hidden_states



class VisionRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> Tensor:
        # TODO - assuming op.arange syntax == torch.arange, changed dtype to the string literal float32, idk how tvm does dtypes
        self.inv_freq = 1.0 / (self.theta ** (op.arange(0, self.dim, 2, dtype="float32") / self.dim))
        pass

        # TODO - translate pytorch to tvm
        raise NotImplementedError
        # seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs = torch.outer(seq, self.inv_freq)
        # return freqs


class VisionAttention(nn.Module):

    # fyi this expects a Qwen2VLVisionConfig
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
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

        # TODO - translate pytorch to tvm

        raise NotImplementedError

        # seq_length = hidden_states.shape[0]
        # query_states, key_states, value_states = (
        #     self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # )
        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # query_states = query_states.transpose(0, 1).unsqueeze(0)
        # key_states = key_states.transpose(0, 1).unsqueeze(0)
        # value_states = value_states.transpose(0, 1).unsqueeze(0)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # if self.config._attn_implementation == "flash_attention_2":
        #     # Flash Attention 2: Use cu_seqlens for variable length attention
        #     max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        #     attn_output, _ = attention_interface(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask=None,
        #         scaling=self.scaling,
        #         dropout=0.0 if not self.training else self.attention_dropout,
        #         cu_seq_lens_q=cu_seqlens,
        #         cu_seq_lens_k=cu_seqlens,
        #         max_length_q=max_seqlen,
        #         max_length_k=max_seqlen,
        #         is_causal=False,
        #         **kwargs,
        #     )
        # else:
        #     # Other implementations: Process each chunk separately
        #     lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        #     splits = [
        #         torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        #     ]

        #     attn_outputs = [
        #         attention_interface(
        #             self,
        #             q,
        #             k,
        #             v,
        #             attention_mask=None,
        #             scaling=self.scaling,
        #             dropout=0.0 if not self.training else self.attention_dropout,
        #             is_causal=False,
        #             **kwargs,
        #         )[0]
        #         for q, k, v in zip(*splits)
        #     ]
        #     attn_output = torch.cat(attn_outputs, dim=1)

        # attn_output = attn_output.reshape(seq_length, -1).contiguous()
        # attn_output = self.proj(attn_output)
        # return attn_output


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        
        # fyi assuming nn.Parameter is a thing

        # do i need to have nn.Parameter, or can it just be op.ones?
        self.weight = nn.Parameter((hidden_size,), dtype="float32")
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        # TODO - translate pytorch to tvm
        
        raise NotImplementedError
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to("float32")
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)

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