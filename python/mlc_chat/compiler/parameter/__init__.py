"""
A subpackage of the compiler that represents mapping between external parameters, quantized
parameters and parameters in MLC-defined models.
"""
from .hf_loader import HFLoader
from .mapping import ExternMapping, QuantizeMapping
