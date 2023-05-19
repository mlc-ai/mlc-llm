from logging import getLogger
from typing import Union

import torch
import torch.nn as nn
from transformers import AutoConfig
import transformers

from ._const import SUPPORTED_MODELS, CPU, CUDA_0


logger = getLogger(__name__)


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def move_to_device(obj: Union[torch.Tensor, nn.Module], device: torch.device):
    if get_device(obj) != device:
        obj = obj.to(device)
    return obj


# def find_layers(module, layers=None, name=''):
#     if not layers:
#         layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]

#     if type(module) in layers:
#         return {name: module}
#     res = {}
#     for name1, child in module.named_children():
#         res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
#     return res

def find_layers(module, layer_types=None, name='', found=None):
    if found is None:
        found = {}
        
    layer_types = layer_types or [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for child_name, child_module in module.named_children():
        full_name = name + '.' + child_name if name else child_name
        if isinstance(child_module, tuple(layer_types)):
            found[full_name] = child_module
        find_layers(child_module, layer_types, full_name, found)
    
    return found



def get_module_by_name(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


def make_quant(module, names, bits, groupsize, name='', use_triton=False, use_tvm=True):
    if use_triton:
        from ..nn_modules.qlinear_triton import QuantLinear
    elif use_tvm:
        from ..nn_modules.qlinear_tvm import QuantLinear
    else:
        from ..nn_modules.qlinear import QuantLinear

    if isinstance(module, QuantLinear):
        return
    for _name, child in module.named_children():
        name1 = name + '.' + _name if name != '' else _name
        # print(_name, name1)
        if name1 in names:
            ori_layer_device = get_device(getattr(module, _name))
            if type(child) == nn.Linear:
                in_features = child.in_features
                out_features = child.out_features
            elif type(child) == nn.Conv2d:
                in_features = child.in_channels
                out_features = child.out_channels
            elif type(child) == transformers.pytorch_utils.Conv1D:            
                in_features = child.weight.shape[0]
                out_features = child.weight.shape[1]
            new_layer = QuantLinear(bits, groupsize, in_features, out_features, child.bias is not None)
            new_layer.device = ori_layer_device
            setattr(module, _name, new_layer.to(ori_layer_device))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1, use_triton=use_triton, use_tvm=use_tvm)

# def make_quant(module, names, bits, groupsize, use_triton=False, use_tvm=True):
#     if use_triton:
#         from ..nn_modules.qlinear_triton import QuantLinear
#     elif use_tvm:
#         from ..nn_modules.qlinear_tvm import QuantLinear
#     else:
#         from ..nn_modules.qlinear import QuantLinear

#     if isinstance(module, QuantLinear):
#         return
#     for name, child in module.named_children():
#         print(name)
#         if name in names:
#             delattr(module, child)
#             if type(child) == nn.Linear:
#                 in_features = child.in_features
#                 out_features = child.out_features
#             elif type(child) == nn.Conv2d:
#                 in_features = child.in_channels
#                 out_features = child.out_channels
#             elif type(child) == transformers.pytorch_utils.Conv1D:            
#                 in_features = child.weight.shape[0]
#                 out_features = child.weight.shape[1]
#             in_features = child.in_features
#             out_features = child.out_features
#             new_layer = QuantLinear(bits, groupsize, in_features, out_features, child.bias is not None)
#             setattr(module, name, new_layer)
#         else:
#             make_quant(child, names, bits, groupsize)
#     return module


def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    use_triton=False,
    use_tvm=False,
    autotune_warmup: bool = False,
    force_layer_back_to_cpu: bool = False
):
    if use_triton:
        from ..nn_modules.qlinear_triton import QuantLinear, autotune_warmup_linear
    elif use_tvm:
        from ..nn_modules.qlinear_tvm import QuantLinear
    else:
        from ..nn_modules.qlinear import QuantLinear

    if force_layer_back_to_cpu:
        model.to(CPU)

    logger.info('Packing model...')
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, bits, group_size, use_triton=use_triton, use_tvm=use_tvm)
    qlayers = find_layers(model, [QuantLinear])
    for name in qlayers:
        logger.info(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        # so far can only pack layer on CPU
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)
        layers[name], scale, zero, g_idx = layers[name].to(CPU), scale.to(CPU), zero.to(CPU), g_idx.to(CPU)
        qlayers[name].pack(layers[name], scale, zero, g_idx)
        qlayers[name].to(layer_device)
    logger.info('Model packed.')

    if use_triton and autotune_warmup:
        logger.warning(
            "using autotune_warmup will move model to GPU, make sure you have enough VRAM to load the whole model."
        )
        autotune_warmup_linear(model.to(CUDA_0), seqlen=model.seqlen)


def check_and_get_model_type(model_dir):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


__all__ = [
    "get_device",
    "move_to_device",
    "find_layers",
    "get_module_by_name",
    "make_quant",
    "pack_model",
    "check_and_get_model_type"
]
