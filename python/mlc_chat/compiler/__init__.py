"""
A compiler for MLC Chat. By default, it is not imported to MLC Chat to avoid unnecessary dependency,
but users could optionally import it if they want to use the compiler.
"""
from . import compiler_pass
from .compile import CompileArgs, compile  # pylint: disable=redefined-builtin
from .convert_weight import ConversionArgs, convert_weight
from .flags_model_config_override import ModelConfigOverride
from .flags_optimization import OptimizationFlags
from .loader import LOADER, ExternMapping, HuggingFaceLoader, QuantizeMapping
from .model import MODEL_PRESETS, MODELS, Model
from .quantization import QUANTIZATION
