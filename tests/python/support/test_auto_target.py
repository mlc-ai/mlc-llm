import json

import pytest
from tvm_ffi import get_global_func, register_global_func
from tvm_tirx.support import nvcc
from tvm_tirx.target import Target

from mlc_llm.support import auto_target
from mlc_llm.support.auto_target import _apply_webgpu_subgroups

# test category "unittest"
pytestmark = [pytest.mark.unittest]


def test_apply_webgpu_subgroups_enables_webgpu_target():
    target = Target("webgpu")

    updated = _apply_webgpu_subgroups(target, True)

    assert updated is not target
    assert dict(target.export())["supports_subgroups"] is False
    assert dict(updated.export())["supports_subgroups"] is True


def test_apply_webgpu_subgroups_non_webgpu_target_is_unchanged():
    target = Target("llvm")

    updated = _apply_webgpu_subgroups(target, True)

    assert updated is target
    assert dict(updated.export()) == dict(target.export())


@pytest.mark.parametrize("target_kind", ["webgpu", "llvm"])
@pytest.mark.parametrize("enable_subgroups", [False, None])
def test_apply_webgpu_subgroups_disabled_is_unchanged(target_kind, enable_subgroups):
    target = Target(target_kind)

    updated = _apply_webgpu_subgroups(target, enable_subgroups)

    assert updated is target
    assert dict(updated.export()) == dict(target.export())


@pytest.mark.parametrize("multi_arch", [None, "75, 90a"])
@pytest.mark.parametrize("use_nvshmem", [False, True])
def test_cuda_compile_callback(monkeypatch, multi_arch, use_nvshmem):
    calls = []

    def compile_nvcc(code, target_format, arch, options, **kwargs):
        calls.append((code, target_format, arch, options, kwargs["use_nvshmem"]))
        return bytearray(b"compiled"), ""

    monkeypatch.setattr(auto_target, "MLC_MULTI_ARCH", multi_arch)
    monkeypatch.setattr(nvcc, "_compile_cuda_nvcc", compile_nvcc)
    previous = get_global_func("tvm_callback_cuda_compile")
    code = "#include <nvshmem.h>" if use_nvshmem else "// source"
    try:
        auto_target._register_cuda_hook(Target({"kind": "cuda", "arch": "sm_80"}))
        callback = get_global_func("tvm_callback_cuda_compile")
        binary, target_format = callback(
            code,
            json.dumps({"cuda": {"arch": "sm_90a", "nvcc": ["-lineinfo"], "ptxas": []}}),
        )
    finally:
        register_global_func("tvm_callback_cuda_compile", previous, override=True)

    assert bytes(binary) == b"compiled"
    assert target_format == ("cubin" if use_nvshmem else "fatbin")
    expected_arch = (
        "sm_90a"
        if multi_arch is None
        else [
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_90a,code=sm_90a",
        ]
    )
    assert calls == [(code, target_format, expected_arch, ["-lineinfo"], use_nvshmem)]
