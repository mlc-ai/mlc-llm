"""
A compiler for MLC Chat. By default, it is not imported to MLC Chat to avoid unnecessary dependency,
but users could optionally import it if they want to use the compiler.
"""
from .compile import (  # pylint: disable=redefined-builtin
    CompileArgs,
    OptimizationFlags,
    compile,
)
from .model import MODELS, Model
from .parameter import ExternMapping, HuggingFaceLoader, QuantizeMapping
from .quantization import QUANT
