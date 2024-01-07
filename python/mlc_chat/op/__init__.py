"""Extern module for compiler."""
from .attention import attention
from .extern import configure, enable, get_store
from .ft_gemm import faster_transformer_dequantize_gemm
from .position_embedding import llama_rope
