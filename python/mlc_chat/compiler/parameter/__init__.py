"""
A subpackage of the compiler that represents mapping between external parameters, quantized
parameters and parameters in MLC-defined models.
"""
from .hf_torch_loader import HFTorchLoader
from .mapping import ExternMapping, QuantizeMapping
