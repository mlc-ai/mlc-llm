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
    monkeypatch.setattr(auto_target, "_pack_nvshmem_cubins", lambda images: bytearray(b"compiled"))
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
    if use_nvshmem and multi_arch is not None:
        assert target_format == "fatbin"
        assert calls == [
            (code, "cubin", "sm_75", ["-lineinfo"], True),
            (code, "cubin", "sm_90a", ["-lineinfo"], True),
        ]
    else:
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


def test_nvshmem_multiarch_compiles_each_image_before_packing(monkeypatch):
    calls = []

    def compile_nvcc(code, target_format, arch, options, **kwargs):
        calls.append((target_format, arch, kwargs["use_nvshmem"]))
        assert isinstance(arch, str)
        return bytearray(arch.encode()), ""

    images = []

    def pack_cubins(cubins):
        images.extend(cubins)
        return bytearray(b"packed")

    monkeypatch.setattr(auto_target, "MLC_MULTI_ARCH", "90a,103a")
    monkeypatch.setattr(auto_target, "_pack_nvshmem_cubins", pack_cubins, raising=False)
    monkeypatch.setattr(nvcc, "_compile_cuda_nvcc", compile_nvcc)
    previous = get_global_func("tvm_callback_cuda_compile")
    try:
        auto_target._register_cuda_hook(Target({"kind": "cuda", "arch": "sm_103a"}))
        callback = get_global_func("tvm_callback_cuda_compile")
        binary, target_format = callback(
            "#include <nvshmem.h>", json.dumps({"cuda": {"arch": "sm_103a"}})
        )
    finally:
        register_global_func("tvm_callback_cuda_compile", previous, override=True)

    assert calls == [("cubin", "sm_90a", True), ("cubin", "sm_103a", True)]
    assert images == [("90a", bytearray(b"sm_90a")), ("103a", bytearray(b"sm_103a"))]
    assert bytes(binary) == b"packed"
    assert target_format == "fatbin"


def test_pack_nvshmem_cubins(monkeypatch):
    from pathlib import Path

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] is True
        assert Path(command[3].split("file=", 1)[1]).read_bytes() == b"sm90"
        assert Path(command[4].split("file=", 1)[1]).read_bytes() == b"sm103"
        Path(command[1].split("=", 1)[1]).write_bytes(b"fatbin")

    monkeypatch.setattr(nvcc, "find_cuda_path", lambda: "/test/cuda")
    monkeypatch.setattr(auto_target.subprocess, "run", run)
    assert auto_target._pack_nvshmem_cubins([("90a", b"sm90"), ("103a", b"sm103")]) == b"fatbin"
    assert commands[0][0] == "/test/cuda/bin/fatbinary"
    assert commands[0][3].startswith("--image3=kind=elf,sm=90a,file=")
    assert commands[0][4].startswith("--image3=kind=elf,sm=103a,file=")


def test_nvshmem_single_arch_uses_cubin(monkeypatch):
    calls = []

    def compile_nvcc(code, target_format, arch, options, **kwargs):
        calls.append((target_format, arch, kwargs["use_nvshmem"]))
        return bytearray(b"single"), ""

    monkeypatch.setattr(auto_target, "MLC_MULTI_ARCH", "103a")
    monkeypatch.setattr(nvcc, "_compile_cuda_nvcc", compile_nvcc)
    previous = get_global_func("tvm_callback_cuda_compile")
    try:
        auto_target._register_cuda_hook(Target({"kind": "cuda", "arch": "sm_103a"}))
        callback = get_global_func("tvm_callback_cuda_compile")
        binary, target_format = callback(
            "#include <nvshmem.h>", json.dumps({"cuda": {"arch": "sm_103a"}})
        )
    finally:
        register_global_func("tvm_callback_cuda_compile", previous, override=True)
    assert calls == [("cubin", "sm_103a", True)]
    assert bytes(binary) == b"single"
    assert target_format == "cubin"
