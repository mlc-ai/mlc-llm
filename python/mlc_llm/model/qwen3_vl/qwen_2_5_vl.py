from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .qwen2_vl import Qwen2RMSNorm, VisionAttention, Qwen2VLModel
from mlc_llm.model.qwen3.qwen3_model import ACT2FN

from typing import Optional

class Qwen2_5_VLVisionAttention(VisionAttention):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.dim = config.hidden_size


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        rotary_pos_emb: Optional[Tensor] = None,
        position_embeddings: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLModel(Qwen2VLModel):
    # config
    base_model_prefix = "model"
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)