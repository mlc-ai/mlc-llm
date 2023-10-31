"""
A compiler for MLC Chat. By default, it is not imported to MLC Chat to avoid unnecessary dependency,
but users could optionally import it if they want to use the compiler.
"""
from . import compiler_pass
from .compile import CompileArgs, compile  # pylint: disable=redefined-builtin
from .flags_optimization import OptimizationFlags
from .model import MODEL_PRESETS, MODELS, Model
from .parameter import ExternMapping, HuggingFaceLoader, QuantizeMapping
from .quantization import QUANTIZATION
