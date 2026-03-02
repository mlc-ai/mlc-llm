from itertools import product
from typing import Tuple

import ml_dtypes
import numpy as np
import pytest
import torch
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.s_tir import dlight as dl

from mlc_llm.compiler_pass.dispatch_triton_kernel import DispatchTritonKernel
from mlc_llm.op import batch_matmul, cutlass, moe_matmul, triton
from mlc_llm.quantization.block_scale_quantization import rowwise_group_quant_fp8

# test category "op_correctness"
pytestmark = [pytest.mark.op_correctness]

block_size = (128, 128)
fp8_dtype = "float8_e4m3fn"

torch_fp8_dtype = torch.float8_e4m3fn
torch_device = torch.device("cuda")

torch.set_grad_enabled(False)


def test_fp8_block_matmul_cutlass(M: int, N: int, K: int, dtype: str):
    class TestModule(nn.Module):
        def __init__(self):
            pass

        def cutlass_gemm(self, x: nn.Tensor, w: nn.Tensor, w_scale: nn.Tensor):
            n, k = w.shape
            m = x.shape[0]
            # assert n % block_size[0] == 0
            assert k % block_size[1] == 0
            assert (n + block_size[0] - 1) // block_size[0] == w_scale.shape[0]
            assert k // block_size[1] == w_scale.shape[1]
            assert x.shape[1] == k
            x_fp8, x_scale = rowwise_group_quant_fp8(
                x, block_size[1], w.dtype, transpose_scale=True
            )
            assert x_fp8.dtype == w.dtype
            assert x_scale.dtype == "float32"
            o = cutlass.fp8_groupwise_scaled_gemm(x_fp8, x_scale, w, w_scale, block_size, x.dtype)
            return x_fp8, x_scale, o

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "cutlass_gemm": {
                "x": spec.Tensor(("m", K), dtype),
                "w": spec.Tensor((N, K), fp8_dtype),
                "w_scale": spec.Tensor(
                    (
                        (N + block_size[0] - 1) // block_size[0],
                        (K + block_size[1] - 1) // block_size[1],
                    ),
                    "float32",
                ),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    exec = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(exec, device)

    x_torch = torch.rand(M, K, dtype=getattr(torch, dtype), device=torch_device) * 2 - 1
    w_full_torch = torch.rand(N, K, dtype=getattr(torch, dtype), device=torch_device) * 2 - 1
    w_torch, w_scale_torch = blockwise_quant_fp8(w_full_torch, block_size, torch_fp8_dtype)
    x_torch, x_fp8_torch, x_scale_torch = rowwise_quant_fp8(x_torch, block_size, torch_fp8_dtype)
    o_torch = blockwise_matmul(x_fp8_torch, x_scale_torch, w_torch, w_scale_torch, x_torch.dtype)
    x_tvm = tvm.runtime.tensor(x_torch.view(torch.float16).cpu().numpy().view(dtype), device=device)
    w_tvm = tvm.runtime.tensor(
        w_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
    )
    w_scale_tvm = tvm.runtime.tensor(w_scale_torch.cpu().numpy(), device=device)
    x_fp8_tvm, x_scale_tvm, o_tvm = vm["cutlass_gemm"](x_tvm, w_tvm, w_scale_tvm)
    x_fp8_tvm = x_fp8_tvm.numpy()
    x_scale_tvm = x_scale_tvm.numpy()
    o_tvm = o_tvm.numpy()

    np.testing.assert_allclose(
        x_fp8_tvm,
        x_fp8_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype),
        atol=1e-1,
        rtol=1e-1,
    )
    np.testing.assert_allclose(x_scale_tvm.T, x_scale_torch.cpu().numpy(), atol=1e-5, rtol=1e-5)
    atol = 0.5
    rtol = 1e-4
    o_tvm_flat = o_tvm.flatten()
    o_torch_flat = o_torch.view(torch.float16).cpu().numpy().view(dtype).flatten()
    failed_indices = np.where(
        np.abs(o_tvm_flat - o_torch_flat) > (atol + rtol * np.abs(o_torch_flat))
    )[0]
    if len(failed_indices) > 0:
        print(f"failed_indices: {failed_indices}, size: {len(failed_indices)}")
        print(f"o_tvm_flat[failed_indices]: {o_tvm_flat[failed_indices]}")
        print(f"o_torch_flat[failed_indices]: {o_torch_flat[failed_indices]}")
    np.testing.assert_allclose(
        o_tvm,
        o_torch.view(torch.float16).cpu().numpy().view(dtype),
        atol=atol,
        rtol=rtol,
    )


def test_fp8_block_matmul_triton(M: int, N: int, K: int, dtype: str):
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)

    class TestModule(nn.Module):
        def __init__(self):
            pass

        def triton_gemm(self, x: nn.Tensor, w: nn.Tensor, w_scale: nn.Tensor):
            n, k = w.shape
            m = x.shape[0]
            assert (n + block_size[0] - 1) // block_size[0] == w_scale.shape[0]
            assert (k + block_size[1] - 1) // block_size[1] == w_scale.shape[1]
            assert x.shape[1] == k
            x_fp8, x_scale = rowwise_group_quant_fp8(
                x, block_size[1], w.dtype, transpose_scale=False
            )
            assert x_fp8.dtype == w.dtype
            assert x_scale.dtype == "float32"
            o = triton.fp8_groupwise_scaled_gemm(
                x_fp8,
                x_scale,
                w,
                w_scale,
                block_size,
                x.dtype,
            )
            return x_fp8, x_scale, o

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "triton_gemm": {
                "x": spec.Tensor(("m", K), dtype),
                "w": spec.Tensor((N, K), fp8_dtype),
                "w_scale": spec.Tensor(
                    (
                        (N + block_size[0] - 1) // block_size[0],
                        (K + block_size[1] - 1) // block_size[1],
                    ),
                    "float32",
                ),
            },
        },
        allow_extern=True,
    )
    mod = DispatchTritonKernel(target)(mod)  # type: ignore
    exec = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(exec, device)

    x_torch = torch.randn(M, K, dtype=getattr(torch, dtype), device=torch_device)
    w_full_torch = torch.randn(N, K, dtype=getattr(torch, dtype), device=torch_device)
    w_torch, w_scale_torch = blockwise_quant_fp8(w_full_torch, block_size, torch_fp8_dtype)
    x_torch, x_fp8_torch, x_scale_torch = rowwise_quant_fp8(x_torch, block_size, torch_fp8_dtype)
    o_torch = blockwise_matmul(x_fp8_torch, x_scale_torch, w_torch, w_scale_torch, x_torch.dtype)
    x_tvm = tvm.runtime.tensor(x_torch.view(torch.float16).cpu().numpy().view(dtype), device=device)
    w_tvm = tvm.runtime.tensor(
        w_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
    )
    w_scale_tvm = tvm.runtime.tensor(w_scale_torch.cpu().numpy(), device=device)
    x_fp8_tvm, x_scale_tvm, o_tvm = vm["triton_gemm"](x_tvm, w_tvm, w_scale_tvm)
    x_fp8_tvm = x_fp8_tvm.numpy()
    x_scale_tvm = x_scale_tvm.numpy()
    o_tvm = o_tvm.numpy()
    np.testing.assert_allclose(
        x_fp8_tvm,
        x_fp8_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype),
        atol=1e-1,
        rtol=1e-1,
    )
    np.testing.assert_allclose(x_scale_tvm, x_scale_torch.cpu().numpy(), atol=1e-5, rtol=1e-5)
    atol = 0.5
    rtol = 1e-4
    o_tvm_flat = o_tvm.flatten()
    o_torch_flat = o_torch.view(torch.float16).cpu().numpy().view(dtype).flatten()
    failed_indices = np.where(
        np.abs(o_tvm_flat - o_torch_flat) > (atol + rtol * np.abs(o_torch_flat))
    )[0]
    if len(failed_indices) > 0:
        print(f"failed_indices: {failed_indices}, size: {len(failed_indices)}")
        print(f"o_tvm_flat[failed_indices]: {o_tvm_flat[failed_indices]}")
        print(f"o_torch_flat[failed_indices]: {o_torch_flat[failed_indices]}")
    np.testing.assert_allclose(
        o_tvm,
        o_torch.view(torch.float16).cpu().numpy().view(dtype),
        atol=atol,
        rtol=rtol,
    )


def test_fp8_block_group_matmul_cutlass(M: int, N: int, K: int, dtype: str):
    num_experts = 256
    top_k = 8

    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)

    class TestModule(nn.Module):
        def __init__(self):
            pass

        def cutlass_group_gemm(
            self,
            x: nn.Tensor,
            w: nn.Tensor,
            w_scale: nn.Tensor,
            indptr: nn.Tensor,
        ):
            e, n, k = w.shape
            m = x.shape[0]
            assert e == num_experts
            assert (n + block_size[0] - 1) // block_size[0] == w_scale.shape[1]
            assert (k + block_size[1] - 1) // block_size[1] == w_scale.shape[2]
            assert x.shape[1] == k
            x_fp8, x_scale = rowwise_group_quant_fp8(
                x, block_size[1], w.dtype, transpose_scale=False
            )
            assert x_fp8.dtype == w.dtype
            assert x_scale.dtype == "float32"
            o = cutlass.fp8_groupwise_scaled_group_gemm(
                x_fp8,
                x_scale,
                w,
                w_scale,
                indptr,
                block_size,
                x.dtype,
            )
            return x_fp8, x_scale, o

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "cutlass_group_gemm": {
                "x": spec.Tensor(("m", K), dtype),
                "w": spec.Tensor((num_experts, N, K), fp8_dtype),
                "w_scale": spec.Tensor(
                    (
                        num_experts,
                        (N + block_size[0] - 1) // block_size[0],
                        (K + block_size[1] - 1) // block_size[1],
                    ),
                    "float32",
                ),
                "indptr": spec.Tensor((num_experts,), "int64"),
            },
        },
        allow_extern=True,
    )
    exec = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(exec, device)

    # Randomly sample `top_k` experts for each token with pytorch
    expert_choices = torch.randint(
        0, num_experts, (M * top_k,), device=torch_device, dtype=torch.int32
    )

    factor = 1
    # Balance so that the number of tokens for each expert is a multiple of `factor`
    token_balance = 0
    num_tokens_list = [int((expert_choices == i).sum().to("cpu")) for i in range(num_experts)]
    for i in range(num_experts):
        if token_balance > 0:
            diff = min(token_balance, num_tokens_list[i])
            num_tokens_list[i] -= diff
            token_balance -= diff
        if num_tokens_list[i] % factor != 0:
            token_balance += factor - num_tokens_list[i] % factor
            num_tokens_list[i] += factor - num_tokens_list[i] % factor
    assert sum(num_tokens_list) == M * top_k

    indptr = torch.zeros(num_experts + 1, device=torch_device, dtype=torch.int64)
    for i in range(num_experts):
        indptr[i + 1] = indptr[i] + (expert_choices == i).sum()
    token_ids_list = []
    for i in range(num_experts):
        # Get the indices of the tokens that belong to the i-th expert
        token_ids = torch.where(expert_choices == i)[0]
        token_ids_list.append(token_ids)

    x_torch = torch.randn(M * top_k, K, dtype=getattr(torch, dtype), device=torch_device)
    w_full_torch = torch.randn(num_experts, N, K, dtype=getattr(torch, dtype), device=torch_device)
    w_torch, w_scale_torch = blockwise_quant_fp8(w_full_torch, block_size, torch_fp8_dtype)
    x_torch, x_fp8_torch, x_scale_torch = rowwise_quant_fp8(x_torch, block_size, torch_fp8_dtype)
    o_torch = blockwise_group_matmul(
        x_fp8_torch,
        x_scale_torch,
        w_torch,
        w_scale_torch,
        indptr,
        x_torch.dtype,
    )
    x_tvm = tvm.runtime.tensor(x_torch.view(torch.float16).cpu().numpy().view(dtype), device=device)
    w_tvm = tvm.runtime.tensor(
        w_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
    )
    w_scale_tvm = tvm.runtime.tensor(w_scale_torch.cpu().numpy(), device=device)
    indptr_tvm = tvm.runtime.tensor(indptr[1:].cpu().numpy(), device=device)
    x_fp8_tvm, x_scale_tvm, o_tvm = vm["cutlass_group_gemm"](
        x_tvm,
        w_tvm,
        w_scale_tvm,
        indptr_tvm,
    )
    x_fp8_tvm = x_fp8_tvm.numpy()
    x_scale_tvm = x_scale_tvm.numpy()
    o_tvm = o_tvm.numpy()
    np.testing.assert_allclose(
        x_fp8_tvm,
        x_fp8_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype),
        atol=1e-1,
        rtol=1e-1,
    )
    np.testing.assert_allclose(x_scale_tvm, x_scale_torch.cpu().numpy(), atol=1e-5, rtol=1e-5)
    atol = 0.5
    rtol = 1e-4
    o_tvm_flat = o_tvm.flatten()
    o_torch_flat = o_torch.view(torch.float16).cpu().numpy().view(dtype).flatten()
    failed_indices = np.where(
        np.abs(o_tvm_flat - o_torch_flat) > (atol + rtol * np.abs(o_torch_flat))
    )[0]
    if len(failed_indices) > 0:
        print(f"failed_indices: {failed_indices}, size: {len(failed_indices)}")
        print(f"o_tvm_flat[failed_indices]: {o_tvm_flat[failed_indices]}")
        print(f"o_torch_flat[failed_indices]: {o_torch_flat[failed_indices]}")
    np.testing.assert_allclose(
        o_tvm,
        o_torch.view(torch.float16).cpu().numpy().view(dtype),
        atol=atol,
        rtol=rtol,
    )


def test_fp8_block_group_matmul_triton(M: int, N: int, K: int, dtype: str):
    num_experts = 256
    top_k = 8

    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)

    class TestModule(nn.Module):
        def __init__(self):
            pass

        def triton_group_gemm(
            self,
            x: nn.Tensor,
            w: nn.Tensor,
            w_scale: nn.Tensor,
            indptr: nn.Tensor,
        ):
            e, n, k = w.shape
            m = x.shape[0]
            assert e == num_experts
            assert (n + block_size[0] - 1) // block_size[0] == w_scale.shape[1]
            assert (k + block_size[1] - 1) // block_size[1] == w_scale.shape[2]
            assert x.shape[1] == k
            x_fp8, x_scale = rowwise_group_quant_fp8(
                x, block_size[1], w.dtype, transpose_scale=False
            )
            assert x_fp8.dtype == w.dtype
            assert x_scale.dtype == "float32"
            o = triton.fp8_groupwise_scaled_group_gemm(
                x_fp8,
                x_scale,
                w,
                w_scale,
                indptr,
                block_size,
                x.dtype,
            )
            return x_fp8, x_scale, o

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "triton_group_gemm": {
                "x": spec.Tensor(("m", K), dtype),
                "w": spec.Tensor((num_experts, N, K), fp8_dtype),
                "w_scale": spec.Tensor(
                    (
                        num_experts,
                        (N + block_size[0] - 1) // block_size[0],
                        (K + block_size[1] - 1) // block_size[1],
                    ),
                    "float32",
                ),
                "indptr": spec.Tensor((num_experts + 1,), "int32"),
            },
        },
        allow_extern=True,
    )
    mod = DispatchTritonKernel(target)(mod)  # type: ignore
    exec = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(exec, device)

    # Randomly sample `top_k` experts for each token with pytorch
    expert_choices = torch.randint(
        0, num_experts, (M * top_k,), device=torch_device, dtype=torch.int32
    )

    indptr = torch.zeros(num_experts + 1, device=torch_device, dtype=torch.int32)
    for i in range(num_experts):
        indptr[i + 1] = indptr[i] + (expert_choices == i).sum()
    token_ids_list = []
    for i in range(num_experts):
        # Get the indices of the tokens that belong to the i-th expert
        token_ids = torch.where(expert_choices == i)[0]
        token_ids_list.append(token_ids)

    x_torch = torch.randn(M * top_k, K, dtype=getattr(torch, dtype), device=torch_device)
    w_full_torch = torch.randn(num_experts, N, K, dtype=getattr(torch, dtype), device=torch_device)
    w_torch, w_scale_torch = blockwise_quant_fp8(w_full_torch, block_size, torch_fp8_dtype)
    x_torch, x_fp8_torch, x_scale_torch = rowwise_quant_fp8(x_torch, block_size, torch_fp8_dtype)
    o_torch = blockwise_group_matmul(
        x_fp8_torch,
        x_scale_torch,
        w_torch,
        w_scale_torch,
        indptr,
        x_torch.dtype,
    )
    x_tvm = tvm.runtime.tensor(x_torch.view(torch.float16).cpu().numpy().view(dtype), device=device)
    w_tvm = tvm.runtime.tensor(
        w_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
    )
    w_scale_tvm = tvm.runtime.tensor(w_scale_torch.cpu().numpy(), device=device)
    indptr_tvm = tvm.runtime.tensor(indptr.cpu().numpy(), device=device)
    x_fp8_tvm, x_scale_tvm, o_tvm = vm["triton_group_gemm"](
        x_tvm,
        w_tvm,
        w_scale_tvm,
        indptr_tvm,
    )
    x_fp8_tvm = x_fp8_tvm.numpy()
    x_scale_tvm = x_scale_tvm.numpy()
    o_tvm = o_tvm.numpy()
    np.testing.assert_allclose(
        x_fp8_tvm,
        x_fp8_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype),
        atol=1e-1,
        rtol=1e-1,
    )
    np.testing.assert_allclose(x_scale_tvm, x_scale_torch.cpu().numpy(), atol=1e-5, rtol=1e-5)
    atol = 0.5
    rtol = 1e-4
    o_tvm_flat = o_tvm.flatten()
    o_torch_flat = o_torch.view(torch.float16).cpu().numpy().view(dtype).flatten()
    failed_indices = np.where(
        np.abs(o_tvm_flat - o_torch_flat) > (atol + rtol * np.abs(o_torch_flat))
    )[0]
    if len(failed_indices) > 0:
        print(f"failed_indices: {failed_indices}, size: {len(failed_indices)}")
        print(f"o_tvm_flat[failed_indices]: {o_tvm_flat[failed_indices]}")
        print(f"o_torch_flat[failed_indices]: {o_torch_flat[failed_indices]}")
    np.testing.assert_allclose(
        o_tvm,
        o_torch.view(torch.float16).cpu().numpy().view(dtype),
        atol=atol,
        rtol=rtol,
    )


def test_fp8_block_bmm_cutlass(M: int, N: int, K: int, H: int, dtype: str):
    class TestModule(nn.Module):
        def __init__(self):
            pass

        def cutlass_bmm(self, x: nn.Tensor, w: nn.Tensor, w_scale: nn.Tensor):
            _, n, k = w.shape
            assert w.shape[0] == x.shape[0] == H
            assert n % block_size[0] == 0
            assert k % block_size[1] == 0
            assert n // block_size[0] == w_scale.shape[1]
            assert k // block_size[1] == w_scale.shape[2]
            assert x.shape[2] == k
            o = batch_matmul.quantized_bmm(x, w, w_scale, block_size)
            return o

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "cutlass_bmm": {
                "x": spec.Tensor((H, "m", K), dtype),
                "w": spec.Tensor((H, N, K), fp8_dtype),
                "w_scale": spec.Tensor(
                    (
                        H,
                        (N + block_size[0] - 1) // block_size[0],
                        (K + block_size[1] - 1) // block_size[1],
                    ),
                    "float32",
                ),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    exec = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(exec, device)

    x_torch = torch.randn(H, M, K, dtype=getattr(torch, dtype), device=torch_device)
    w_full_torch = torch.randn(H, N, K, dtype=getattr(torch, dtype), device=torch_device)
    w_torch, w_scale_torch = blockwise_quant_fp8(w_full_torch, block_size, torch_fp8_dtype)
    x_torch, x_fp8_torch, x_scale_torch = rowwise_quant_fp8(x_torch, block_size, torch_fp8_dtype)
    o_torch = blockwise_bmm(x_fp8_torch, x_scale_torch, w_torch, w_scale_torch, x_torch.dtype)
    x_tvm = tvm.runtime.tensor(x_torch.view(torch.float16).cpu().numpy().view(dtype), device=device)
    w_tvm = tvm.runtime.tensor(
        w_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
    )
    w_scale_tvm = tvm.runtime.tensor(w_scale_torch.cpu().numpy(), device=device)
    o_tvm = vm["cutlass_bmm"](x_tvm, w_tvm, w_scale_tvm)
    o_tvm = o_tvm.numpy()
    atol = 0.5
    rtol = 1e-4
    o_tvm_flat = o_tvm.flatten()
    o_torch_flat = o_torch.view(torch.float16).cpu().numpy().view(dtype).flatten()
    failed_indices = np.where(
        np.abs(o_tvm_flat - o_torch_flat) > (atol + rtol * np.abs(o_torch_flat))
    )[0]
    if len(failed_indices) > 0:
        print(f"failed_indices: {failed_indices}, size: {len(failed_indices)}")
        print(f"o_tvm_flat[failed_indices]: {o_tvm_flat[failed_indices]}")
        print(f"o_torch_flat[failed_indices]: {o_torch_flat[failed_indices]}")
    np.testing.assert_allclose(
        o_tvm,
        o_torch.view(torch.float16).cpu().numpy().view(dtype),
        atol=atol,
        rtol=rtol,
    )


def test_fp8_block_gemv_tir(N: int, K: int, up: bool, dtype: str):
    num_experts = 256
    top_k = 8
    M = 1 if up else top_k

    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)

    class TestModule(nn.Module):
        def __init__(self):
            pass

        def tir_moe_gemv(
            self,
            x: nn.Tensor,
            w: nn.Tensor,
            w_scale: nn.Tensor,
            expert_indices: nn.Tensor,
        ):
            e, n, k = w.shape
            m = x.shape[0]
            assert e == num_experts
            assert (n + block_size[0] - 1) // block_size[0] == w_scale.shape[1]
            assert (k + block_size[1] - 1) // block_size[1] == w_scale.shape[2]
            assert x.shape[1] == k
            o = moe_matmul.dequantize_block_scale_float8_gemv(
                x, w, w_scale, expert_indices, block_size, x.dtype
            )
            return o

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "tir_moe_gemv": {
                "x": spec.Tensor((M, K), dtype),
                "w": spec.Tensor((num_experts, N, K), fp8_dtype),
                "w_scale": spec.Tensor(
                    (
                        num_experts,
                        (N + block_size[0] - 1) // block_size[0],
                        (K + block_size[1] - 1) // block_size[1],
                    ),
                    "float32",
                ),
                "expert_indices": spec.Tensor((1, top_k), "int32"),
            },
        },
        allow_extern=True,
    )
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)
    exec = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(exec, device)

    # Randomly sample `top_k` experts for each token with pytorch
    expert_choices = torch.randint(0, num_experts, (top_k,), device=torch_device, dtype=torch.int32)
    indptr = torch.zeros(num_experts + 1, device=torch_device, dtype=torch.int32)
    for i in range(num_experts):
        indptr[i + 1] = indptr[i] + (expert_choices == i).sum()
    token_ids_list = []
    for i in range(num_experts):
        # Get the indices of the tokens that belong to the i-th expert
        token_ids = torch.where(expert_choices == i)[0]
        token_ids_list.append(token_ids)

    x_torch = torch.randn(M, K, dtype=getattr(torch, dtype), device=torch_device)
    w_full_torch = torch.randn(num_experts, N, K, dtype=getattr(torch, dtype), device=torch_device)
    w_torch, w_scale_torch = blockwise_quant_fp8(w_full_torch, block_size, torch_fp8_dtype)
    x_input_torch = torch.repeat_interleave(x_torch, top_k, dim=0) if up else x_torch
    o_torch = blockwise_group_matmul_unquantized(
        x_input_torch, w_torch, w_scale_torch, expert_choices
    )
    x_tvm = tvm.runtime.tensor(x_torch.view(torch.float16).cpu().numpy().view(dtype), device=device)
    w_tvm = tvm.runtime.tensor(
        w_torch.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
    )
    w_scale_tvm = tvm.runtime.tensor(w_scale_torch.cpu().numpy(), device=device)
    expert_choices = tvm.runtime.tensor(
        expert_choices.reshape(1, top_k).cpu().numpy(), device=device
    )
    o_tvm = vm["tir_moe_gemv"](x_tvm, w_tvm, w_scale_tvm, expert_choices)
    o_tvm = o_tvm.numpy()
    atol = 0.5
    rtol = 1e-4
    o_tvm_flat = o_tvm.flatten()
    o_torch_flat = o_torch.view(torch.float16).cpu().numpy().view(dtype).flatten()
    failed_indices = np.where(
        np.abs(o_tvm_flat - o_torch_flat) > (atol + rtol * np.abs(o_torch_flat))
    )[0]
    if len(failed_indices) > 0:
        print(f"failed_indices: {failed_indices}, size: {len(failed_indices)}")
        print(f"o_tvm_flat[failed_indices]: {o_tvm_flat[failed_indices]}")
        print(f"o_torch_flat[failed_indices]: {o_torch_flat[failed_indices]}")
    np.testing.assert_allclose(
        o_tvm,
        o_torch.view(torch.float16).cpu().numpy().view(dtype),
        atol=atol,
        rtol=rtol,
    )


def blockwise_matmul(
    x_fp8_torch: torch.Tensor,
    x_scale_torch: torch.Tensor,
    w_torch: torch.Tensor,
    w_scale_torch: torch.Tensor,
    dtype,
):
    o_torch = torch.zeros(
        (x_fp8_torch.shape[0], w_torch.shape[0]), dtype=dtype, device=torch_device
    )
    for j in range(w_scale_torch.shape[0]):
        for k in range(w_scale_torch.shape[1]):
            o_torch[
                :,
                j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[0]),
            ] += (
                torch.matmul(
                    x_fp8_torch[
                        :,
                        k * block_size[1] : min((k + 1) * block_size[1], x_fp8_torch.shape[1]),
                    ].to(dtype),
                    w_torch[
                        j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[0]),
                        k * block_size[1] : min((k + 1) * block_size[1], w_torch.shape[1]),
                    ].T.to(dtype),
                )
                * x_scale_torch[:, k : k + 1]
                * w_scale_torch[j, k]
            )
    return o_torch


def blockwise_group_matmul(
    x_fp8_torch: torch.Tensor,
    x_scale_torch: torch.Tensor,
    w_torch: torch.Tensor,
    w_scale_torch: torch.Tensor,
    indptr: torch.Tensor,
    dtype,
):
    o_torch = torch.zeros(
        (x_fp8_torch.shape[0], w_torch.shape[1]), dtype=dtype, device=torch_device
    )
    for e in range(w_scale_torch.shape[0]):
        if indptr[e + 1] - indptr[e] == 0:
            continue
        indices = slice(indptr[e], indptr[e + 1])
        for j in range(w_scale_torch.shape[1]):
            for k in range(w_scale_torch.shape[2]):
                o_torch[
                    indices,
                    j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[1]),
                ] += (
                    torch.matmul(
                        x_fp8_torch.to(dtype)[
                            indices,
                            k * block_size[1] : min((k + 1) * block_size[1], x_fp8_torch.shape[1]),
                        ],
                        w_torch[
                            e,
                            j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[1]),
                            k * block_size[1] : min((k + 1) * block_size[1], w_torch.shape[2]),
                        ].T.to(dtype),
                    )
                    * x_scale_torch[indices, k : k + 1]
                    * w_scale_torch[e, j, k]
                )
    return o_torch


def blockwise_group_matmul_unquantized(
    x_torch: torch.Tensor,
    w_torch: torch.Tensor,
    w_scale_torch: torch.Tensor,
    expert_choices: torch.Tensor,
):
    o_torch = torch.zeros(
        (x_torch.shape[0], w_torch.shape[1]), dtype=x_torch.dtype, device=torch_device
    )
    for i, e in enumerate(expert_choices):
        for j in range(w_scale_torch.shape[1]):
            for k in range(w_scale_torch.shape[2]):
                o_torch[
                    i,
                    j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[1]),
                ] += torch.matmul(
                    x_torch[
                        i,
                        k * block_size[1] : min((k + 1) * block_size[1], x_torch.shape[1]),
                    ],
                    w_torch[
                        e,
                        j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[1]),
                        k * block_size[1] : min((k + 1) * block_size[1], w_torch.shape[2]),
                    ].T.to(x_torch.dtype)
                    * w_scale_torch[e, j, k].to(x_torch.dtype),
                )
    return o_torch


def blockwise_bmm(
    x_fp8_torch: torch.Tensor,
    x_scale_torch: torch.Tensor,
    w_torch: torch.Tensor,
    w_scale_torch: torch.Tensor,
    dtype,
):
    o_torch = torch.zeros(
        (x_fp8_torch.shape[0], x_fp8_torch.shape[1], w_torch.shape[1]),
        dtype=dtype,
        device=torch_device,
    )
    for j in range(w_scale_torch.shape[1]):
        for k in range(w_scale_torch.shape[2]):
            o_torch[
                ...,
                j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[1]),
            ] += (
                torch.bmm(
                    x_fp8_torch[
                        ...,
                        k * block_size[1] : min((k + 1) * block_size[1], x_fp8_torch.shape[2]),
                    ].to(dtype),
                    w_torch[
                        ...,
                        j * block_size[0] : min((j + 1) * block_size[0], w_torch.shape[1]),
                        k * block_size[1] : min((k + 1) * block_size[1], w_torch.shape[2]),
                    ]
                    .transpose(1, 2)
                    .to(dtype),
                )
                * x_scale_torch[..., k : k + 1]
                * w_scale_torch[..., j : j + 1, k : k + 1]
            )
    return o_torch


def blockwise_quant_fp8(
    w_full_torch: torch.Tensor, block_size: Tuple[int, int], quant_dtype: torch.dtype
):
    w_scale_shape = (
        *w_full_torch.shape[:-2],
        (w_full_torch.shape[-2] + block_size[0] - 1) // block_size[0],
        (w_full_torch.shape[-1] + block_size[1] - 1) // block_size[1],
    )
    # For each (block_size[0], block_size[1]) block, compute the max abs value of `w_full_torch`
    w_max_abs_torch = torch.zeros(w_scale_shape, dtype=torch.float32, device=torch_device)
    for i in range(w_scale_shape[-2]):
        for j in range(w_scale_shape[-1]):
            w_max_abs_torch[..., i, j] = torch.max(
                torch.abs(
                    w_full_torch[
                        ...,
                        i * block_size[0] : min((i + 1) * block_size[0], w_full_torch.shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], w_full_torch.shape[-1]),
                    ]
                ).flatten(-2, -1),
                dim=-1,
            )[0]
    # Scale is the `w_max_abs_torch` divided by the max value of quant_dtype in ml_dtypes
    fp8_max = float(ml_dtypes.finfo(fp8_dtype).max)
    w_scale_torch = w_max_abs_torch / fp8_max
    # `w_torch` is the `w_full_torch` divided by the `w_scale_torch` (with block awareness),
    # clamped to (-fp8_max, fp8_max), and cast to `quant_dtype`
    w_torch = torch.zeros_like(w_full_torch, dtype=quant_dtype, device=torch_device)
    if len(w_scale_shape) == 2:
        for i in range(w_scale_shape[-2]):
            for j in range(w_scale_shape[-1]):
                w_torch[
                    i * block_size[0] : min((i + 1) * block_size[0], w_full_torch.shape[-2]),
                    j * block_size[1] : min((j + 1) * block_size[1], w_full_torch.shape[-1]),
                ] = torch.clamp(
                    w_full_torch[
                        i * block_size[0] : min((i + 1) * block_size[0], w_full_torch.shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], w_full_torch.shape[-1]),
                    ]
                    / w_scale_torch[..., i, j],
                    -fp8_max,
                    fp8_max,
                )
    else:
        for e in range(w_scale_shape[0]):
            for i in range(w_scale_shape[-2]):
                for j in range(w_scale_shape[-1]):
                    w_torch[
                        e,
                        i * block_size[0] : min((i + 1) * block_size[0], w_full_torch.shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], w_full_torch.shape[-1]),
                    ] = torch.clamp(
                        w_full_torch[
                            e,
                            i
                            * block_size[0] : min((i + 1) * block_size[0], w_full_torch.shape[-2]),
                            j
                            * block_size[1] : min((j + 1) * block_size[1], w_full_torch.shape[-1]),
                        ]
                        / w_scale_torch[e, i, j],
                        -fp8_max,
                        fp8_max,
                    )

    w_scale_torch = (
        torch.rand(w_scale_torch.shape, dtype=torch.float32, device=torch_device) / fp8_max
    )
    return w_torch, w_scale_torch


def rowwise_quant_fp8(
    x_full_torch: torch.Tensor, block_size: Tuple[int, int], quant_dtype: torch.dtype
):
    x_scale_shape = (
        *x_full_torch.shape[:-1],
        (x_full_torch.shape[-1] + block_size[1] - 1) // block_size[1],
    )
    # For each (block_size[1]) block, compute the max abs value of `w_full_torch`
    x_max_abs_torch = torch.zeros(x_scale_shape, dtype=torch.float32, device=torch_device)
    for i in range(x_scale_shape[-1]):
        x_max_abs_torch[..., i] = torch.max(
            torch.abs(
                x_full_torch[
                    ...,
                    i * block_size[1] : min((i + 1) * block_size[1], x_full_torch.shape[-1]),
                ]
            ),
            dim=-1,
        )[0]
    # Scale is the `x_max_abs_torch` divided by the max value of quant_dtype in ml_dtypes
    fp8_max = float(ml_dtypes.finfo(fp8_dtype).max)
    x_scale_torch = x_max_abs_torch / fp8_max
    # `x_torch` is the `x_full_torch` divided by the `x_scale_torch` (with block awareness),
    # clamped to (-fp8_max, fp8_max), and cast to `quant_dtype`
    x_torch = torch.zeros_like(x_full_torch, dtype=quant_dtype, device=torch_device)
    for i in range(x_scale_shape[-1]):
        x_torch[
            ...,
            i * block_size[1] : min((i + 1) * block_size[1], x_full_torch.shape[-1]),
        ] = torch.clamp(
            x_full_torch[
                ...,
                i * block_size[1] : min((i + 1) * block_size[1], x_full_torch.shape[-1]),
            ]
            / x_scale_torch[..., i : i + 1],
            -fp8_max,
            fp8_max,
        )

    x_scale_torch = (
        torch.rand(x_scale_torch.shape, dtype=torch.float32, device=torch_device) / fp8_max
    )
    for i in range(x_scale_shape[-1]):
        x_full_torch[
            ...,
            i * block_size[1] : min((i + 1) * block_size[1], x_full_torch.shape[-1]),
        ] = (
            x_torch[
                ...,
                i * block_size[1] : min((i + 1) * block_size[1], x_full_torch.shape[-1]),
            ].to(x_scale_torch.dtype)
            * x_scale_torch[..., i : i + 1]
        )
    return x_full_torch, x_torch, x_scale_torch


@pytest.mark.skip(reason="Test requiring SM90a")
def test_cutlass_gemm():
    # Cutlass GEMM
    for M, (N, K), dtype in product(
        [4, 128, 256, 1024, 2112],
        [
            (4608, 896),
            (896, 2304),
            (3072, 896),
            (512, 896),
            (3072, 512),
            (4096, 512),
            (896, 2048),
            (129280, 896),
        ],
        ["bfloat16"],
    ):
        print(f"Cutlass, M: {M}, N: {N}, K: {K}, dtype: {dtype}")
        test_fp8_block_matmul_cutlass(M, N, K, dtype)


@pytest.mark.skip(reason="Test requiring SM90a")
def test_triton_gemm():
    # Triton GEMM
    for M, (N, K), dtype in product(
        [1, 128, 256, 1024, 2111],
        [
            (4608, 896),
            (896, 576),
            (896, 2304),
        ],
        ["bfloat16"],
    ):
        print(f"Triton, M: {M}, N: {N}, K: {K}, dtype: {dtype}")
        test_fp8_block_matmul_triton(M, N, K, dtype)


@pytest.mark.skip(reason="Test requiring SM90a")
def test_cutlass_group_gemm():
    # Cutlass group GEMM
    for M, (N, K), dtype in product(
        [1, 128, 256, 1024, 2111],
        [
            (512, 896),
            (896, 256),
        ],
        ["bfloat16"],
    ):
        print(f"Cutlass group gemm, M: {M}, N: {N}, K: {K}, dtype: {dtype}")
        test_fp8_block_group_matmul_cutlass(M, N, K, dtype)


@pytest.mark.skip(reason="Test requiring SM90a")
def test_triton_group_gemm():
    # Triton group GEMM
    for M, (N, K), dtype in product(
        [1, 128, 256, 1024, 2111],
        [
            (512, 896),
            (896, 256),
        ],
        ["bfloat16"],
    ):
        print(f"Triton group gemm, M: {M}, N: {N}, K: {K}, dtype: {dtype}")
        test_fp8_block_group_matmul_triton(M, N, K, dtype)


@pytest.mark.skip(reason="Test requiring SM90a")
def test_cutlass_bmm():
    # Cutlass BMM
    for M, H, (N, K), dtype in product(
        [4, 128, 256, 1024, 2112],
        [16, 64, 128],
        [
            (512, 128),
            (128, 512),
        ],
        ["bfloat16"],
    ):
        print(f"Cutlass BMM, M: {M}, N: {N}, K: {K}, H: {H}, dtype: {dtype}")
        test_fp8_block_bmm_cutlass(M, N, K, H, dtype)


@pytest.mark.skip(reason="Test requiring SM90a")
def test_tir_moe_gemv():
    # TIR MoE GEMV
    for (N, K), up, dtype in product(
        [(512, 896), (896, 256)],
        [True, False],
        ["bfloat16"],
    ):
        print(f"TIR MoE GEMV, N: {N}, K: {K}, up: {up}, dtype: {dtype}")
        test_fp8_block_gemv_tir(N, K, up, dtype)


if __name__ == "__main__":
    test_cutlass_gemm()
    test_triton_gemm()
    test_cutlass_group_gemm()
    test_triton_group_gemm()
    test_cutlass_bmm()
    test_tir_moe_gemv()
