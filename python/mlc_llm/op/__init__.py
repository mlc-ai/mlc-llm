"""Extern module for compiler."""

from . import moe_matmul, moe_misc
from .attention import attention
from .batch_spec_verify import batch_spec_verify
from .extern import configure, enable, get_store
from .ft_gemm import faster_transformer_dequantize_gemm
from .pipeline_parallel import pipeline_stage_boundary

"""Operator helper sub-package for MLC-LLM.

Besides standard utilities (Rope, Top-p pivot, …) we expose a provisional
`lora_dense` helper implemented in pure Relax so every backend works today.
Once an upstream Relax primitive lands we will re-export that instead without
changing call-sites in the rest of the code-base.
"""

# Base helpers that already existed.
from .mrope import (  # noqa: F401
    MultimodalRotaryEmbedding,
    VisionPositionMetadata,
    apply_multimodal_rotary_pos_emb,
    get_mrope_position_ids,
)
from .position_embedding import llama_rope  # noqa: F401
from .top_p_pivot import top_p_pivot, top_p_renorm  # noqa: F401

# New provisional fused LoRA op
from .lora import lora_dense  # noqa: F401
