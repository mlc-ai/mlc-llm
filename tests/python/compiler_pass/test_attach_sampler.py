import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.runtime import tensor

from mlc_llm.compiler_pass.attach_sampler import (
    AttachGPUSamplingFunc,
    _attach_greedy_sampling_func,
)


def _build_greedy_module(
    target: tvm.target.Target,
    active_vocab_size: int,
    webgpu_sampler_subgroups: bool = False,
    webgpu_sampler_workgroup_size: int | None = None,
) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    gv = _attach_greedy_sampling_func(
        bb,
        target,
        active_vocab_size,
        webgpu_sampler_subgroups,
        webgpu_sampler_workgroup_size,
    )
    mod = bb.finalize()
    mod[gv] = (
        mod[gv]
        .with_attr("tir_var_upper_bound", {"batch_size": 4})
        .with_attr("tir_non_negative_var", ["vocab_size"])
    )
    return mod


def test_attach_webgpu_greedy_sampler():
    mod = AttachGPUSamplingFunc(
        tvm.target.Target("webgpu"),
        {"batch_size": 4},
        {"active_vocab_size": 7},
    )(tvm.IRModule())

    assert "sample_with_temperature_zero" in mod
    func = mod["sample_with_temperature_zero"]
    assert func.ret_ty == relax.TensorType((func.params[0].ty.shape[0],), "int32")
    assert func.attrs["tir_var_upper_bound"]["batch_size"] == 4
    assert "greedy_argmax" in mod


def test_webgpu_greedy_sampler_builds():
    target = tvm.target.Target("webgpu", host="llvm")
    relax.build(_build_greedy_module(target, active_vocab_size=7), target=target)


def test_webgpu_greedy_sampler_subgroups_are_local():
    target = tvm.target.Target("webgpu", host="llvm")
    mod = _build_greedy_module(
        target,
        active_vocab_size=7,
        webgpu_sampler_subgroups=True,
    )

    sampler_target = mod["greedy_argmax"].attrs["target"]
    assert dict(target.export())["supports_subgroups"] is False
    assert dict(target.export())["thread_warp_size"] == 1
    assert dict(sampler_target.export())["supports_subgroups"] is True
    assert dict(sampler_target.export())["thread_warp_size"] == 32
    relax.build(mod, target=target)


@pytest.mark.parametrize("workgroup_size", [1024])
def test_webgpu_greedy_sampler_workgroup_size_is_local(workgroup_size: int):
    target = tvm.target.Target("webgpu", host="llvm")
    mod = _build_greedy_module(
        target,
        active_vocab_size=128256,
        webgpu_sampler_subgroups=True,
        webgpu_sampler_workgroup_size=workgroup_size,
    )

    sampler = mod["greedy_argmax"]
    sampler_target = sampler.attrs["target"]
    assert dict(target.export())["max_num_threads"] == 256
    assert dict(sampler_target.export())["max_num_threads"] == workgroup_size
    assert f'T.thread_binding({workgroup_size}, thread="threadIdx.x")' in sampler.script()
    relax.build(mod, target=target)


@pytest.mark.parametrize("workgroup_size", [48])
def test_webgpu_greedy_sampler_rejects_invalid_workgroup_size(workgroup_size: int):
    with pytest.raises(ValueError, match="workgroup size must be one of"):
        _build_greedy_module(
            tvm.target.Target("webgpu", host="llvm"),
            active_vocab_size=128256,
            webgpu_sampler_workgroup_size=workgroup_size,
        )


def test_greedy_sampler_runtime():
    device = tvm.metal()
    if not device.exist:
        pytest.skip("Metal runtime is unavailable")

    target = tvm.target.Target("metal", host="llvm")
    executable = relax.build(_build_greedy_module(target, active_vocab_size=7), target=target)
    vm = relax.VirtualMachine(executable, device)
    logits = np.array(
        [
            [[1, 3, 2, 3, -1, 0, 2, 100, 200, 300]],
            [[-2, -1, 0, 2, 5, 4, 3, 100, 200, 300]],
            [[-5, -4, -3, -2, -1, 9, 8, 100, 200, 300]],
            [[np.finfo("float32").min] * 7 + [100, 200, 300]],
        ],
        dtype="float32",
    )

    result = vm["sample_with_temperature_zero"](tensor(logits, device)).numpy()
    np.testing.assert_array_equal(result, np.array([1, 4, 5, 0], dtype="int32"))
