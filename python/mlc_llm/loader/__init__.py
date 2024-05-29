"""
A subpackage of the compiler that represents mapping between external parameters, quantized
parameters and parameters in MLC-defined models.
"""

from .huggingface_loader import HuggingFaceLoader
from .loader import LOADER, Loader
from .mapping import ExternMapping, QuantizeMapping
