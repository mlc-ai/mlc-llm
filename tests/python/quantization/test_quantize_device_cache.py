"""Regression tests for device-bound quantization function caches."""

import copy
from types import SimpleNamespace

import pytest
import tvm_ffi

from mlc_llm.quantization import QUANTIZATION, group_quantization, per_tensor_quantization


@pytest.mark.parametrize(
    "config_name,module",
    [
        ("q4f16_1", group_quantization),
        ("e4m3_e4m3_f16", per_tensor_quantization),
    ],
)
def test_quantize_cache_is_bound_to_input_device(monkeypatch, config_name, module):
    config = copy.deepcopy(QUANTIZATION[config_name])
    config._quantize_func_cache.clear()
    compiled_devices = []

    def compile_fake(mod, device):
        compiled_devices.append(device.index)
        bound_device = device.index
        return lambda weight: bound_device

    monkeypatch.setattr(module, "compile_quantize_func", compile_fake)
    for index in (0, 0, 1):
        weight = SimpleNamespace(
            shape=(16, 128), dtype="float16", device=tvm_ffi.device("cuda", index)
        )
        assert config.quantize_weight(weight) == index
    assert compiled_devices == [0, 1]


def test_ft_quantize_cache_is_bound_to_input_device(monkeypatch):
    from tvm_tirx.target import Target

    from mlc_llm.quantization import ft_quantization as ft

    config = copy.deepcopy(QUANTIZATION["q4f16_ft"])
    config._quantize_func_cache.clear()
    compiled_devices = []
    original_get_global_func = ft.tvm_ffi.get_global_func

    def get_global_func(name, *args, **kwargs):
        if name == "relax.ext.cutlass":
            return lambda *args: None
        return original_get_global_func(name, *args, **kwargs)

    class FakeVM:
        def __init__(self, executable, device):
            self.device = device
            compiled_devices.append(device.index)

        def __getitem__(self, name):
            return lambda weight: self.device.index

    monkeypatch.setattr(ft.tvm_ffi, "get_global_func", get_global_func)
    monkeypatch.setattr(ft.dl, "ApplyDefaultSchedule", lambda *args: lambda mod: mod)
    monkeypatch.setattr(ft.relax, "build", lambda *args, **kwargs: None)
    monkeypatch.setattr(ft.relax, "VirtualMachine", FakeVM)
    with Target({"kind": "cuda", "arch": "sm_103a"}):
        for index in (0, 0, 1):
            weight = SimpleNamespace(
                shape=(16, 128), dtype="float16", device=tvm_ffi.device("cuda", index)
            )
            assert config.quantize_weight(weight) == index
    assert compiled_devices == [0, 1]


def test_group_quantize_cache_on_two_cuda_devices():
    import numpy as np
    import torch
    import tvm_tirx

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    config = copy.deepcopy(QUANTIZATION["q4f16_1"])
    config._quantize_func_cache.clear()
    values = np.random.default_rng(441).normal(size=(16, 128)).astype("float16")
    results = []
    for index in (0, 1):
        device = tvm_tirx.cuda(index)
        result = config.quantize_weight(tvm_tirx.runtime.tensor(values, device))
        assert all(item.device.index == index for item in result)
        results.append(result)
    for first, second in zip(*results):
        np.testing.assert_array_equal(first.numpy(), second.numpy())
