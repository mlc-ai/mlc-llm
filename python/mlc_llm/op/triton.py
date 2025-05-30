"""Operators enabled by external modules."""

# pylint: disable=invalid-name

from typing import List, Literal, Tuple

import tvm
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import tir as T

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# We use a wrapper function to avoid type annotation issue of "tl.constexpr" when
# triton is not installed.
def _get_triton_w8a8_block_fp8_gemm():
    # Triton kernel adapted from SGLang project
    # https://github.com/sgl-project/sglang/blob/v0.4.4/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py  # pylint: disable=line-too-long
    def _triton_w8a8_block_fp8_gemm(  # pylint: disable=too-many-arguments,too-many-locals
        # Pointers to inputs and output
        A,
        B,
        C,
        As,
        Bs,
        # Shape for matmul
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        # Stride for inputs and output
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_As_m: tl.constexpr,
        stride_As_k: tl.constexpr,
        stride_Bs_k: tl.constexpr,
        stride_Bs_n: tl.constexpr,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Triton-accelerated function used to perform linear operations (dot
        product) on input tensors `A` and `B` with block-wise quantization,
        and store the result in output tensor `C`.
        """

        pid = tl.program_id(axis=0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        As_ptrs = As + offs_am * stride_As_m
        offs_bsn = offs_bn // group_n
        Bs_ptrs = Bs + offs_bsn * stride_Bs_n

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
            b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if C.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif C.dtype.element_ty == tl.float16:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator.to(tl.float32)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    return _triton_w8a8_block_fp8_gemm


# We use a wrapper function to avoid type annotation issue of "tl.constexpr" when
# triton is not installed.
def _get_triton_w8a8_block_fp8_group_gemm():
    # Triton kernel adapted from SGLang project
    # https://github.com/sgl-project/sglang/blob/v0.4.4/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py  # pylint: disable=line-too-long
    def _triton_w8a8_block_fp8_group_gemm(  # pylint: disable=too-many-arguments,too-many-locals
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        expert_ids_ptr,
        indptr_ptr,
        # Matrix dimensions
        EM,
        N: tl.constexpr,
        K: tl.constexpr,
        num_experts: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_be: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_asm: tl.constexpr,
        stride_ask: tl.constexpr,
        stride_bse: tl.constexpr,
        stride_bsk: tl.constexpr,
        stride_bsn: tl.constexpr,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        even_Ks: tl.constexpr,
    ):
        """
        Implements the fused computation for a Mixture of Experts (MOE) using
        token and expert matrices.

        Key Parameters:
        - A: The input tensor representing tokens with shape (*, K), where '*' can
            be any shape representing batches and K is the feature dimension of
            each token.
        - B: The stacked MOE weight tensor with shape (E, N, K), where E is
            the number of experts, K is the input feature dimension, and N is
            the output feature dimension.
        - C: The output cache tensor with shape (*, N), where '*' means the
            same shape as the input tensor A, and N is the output feature dimension.
        - expert_ids: A tensor containing the indices of the expert for each
            block. It determines which expert matrix from B should be used for
            each block in A.
        This kernel performs the multiplication of a token by its corresponding
        expert matrix as determined by `expert_ids`.
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        pid = tl.program_id(axis=0).to(tl.int64)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M) + num_experts
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if expert_id == -1:
            return

        token_begin = tl.load(indptr_ptr + expert_id)
        token_end = tl.load(indptr_ptr + expert_id + 1)
        start_pid_m = tl.cdiv(token_begin, BLOCK_SIZE_M) + expert_id
        offs_token_id = (
            token_begin + (pid_m - start_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        )
        token_mask = offs_token_id < token_end

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_token_id[:, None] * stride_am + offs_k[None, :] * stride_ak

        b_ptrs = (
            b_ptr
            + expert_id * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
        offs_bsn = offs_bn // group_n
        b_scale_ptrs = b_scale_ptr + expert_id * stride_bse + offs_bsn * stride_bsn

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if even_Ks:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None],
                    other=0.0,
                )
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            # We accumulate along the K dimension.
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if c_ptr.dtype.element_ty == tl.bfloat16:
            accumulator = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float16:
            accumulator = accumulator.to(tl.float16)
        else:
            accumulator = accumulator.to(tl.float32)

        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    return _triton_w8a8_block_fp8_group_gemm


def get_tir_w8a8_block_fp8_matmul(  # pylint: disable=too-many-arguments,too-many-locals
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    in_dtype: Literal["float8_e4m3fn"],
    out_dtype: Literal["float16", "bfloat16"],
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    GROUP_SIZE_M: int,
    num_warps: int,
    num_stages: int,
    extern_mods: List[tvm.runtime.Module],
):
    """Get the TIR function for the w8a8_block_fp8_matmul kernel."""
    # NOTE: adding the type annotation of " -> Tuple[Optional[tvm.tir.PrimFunc], str]"
    # will cause the failure of the type resolution in mypy.
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    name_suffix = f"_N{N}_K{K}_block_n{block_n}_block_k{block_k}_in{in_dtype}_out{out_dtype}"
    kernel_name = f"triton_w8a8_block_fp8_gemm{name_suffix}"
    tir_name = f"tir_w8a8_block_fp8_matmul{name_suffix}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_w8a8_block_fp8_gemm()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class BlockFP8Matmul:  # pylint: disable=missing-class-docstring,too-few-public-methods
        @T.prim_func(private=True)
        def tir_w8a8_block_fp8_matmul(  # pylint: disable=missing-function-docstring
            var_A: T.handle,
            var_B: T.handle,
            var_As: T.handle,
            var_Bs: T.handle,
            var_C: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tir.is_scheduled": 1})
            M = T.SizeVar("M", "int32")
            A = T.match_buffer(var_A, (M, K), dtype=in_dtype)
            B = T.match_buffer(var_B, (N, K), dtype=in_dtype)
            As = T.match_buffer(var_As, (M, (K + block_k - 1) // block_k), "float32")
            Bs = T.match_buffer(
                var_Bs, ((N + block_n - 1) // block_n, (K + block_k - 1) // block_k), "float32"
            )
            C = T.match_buffer(var_C, (M, N), dtype=out_dtype)
            with T.block("root"):
                T.reads(
                    A[0:M, 0:K],
                    B[0:N, 0:K],
                    As[0:M, 0 : (K + block_k - 1) // block_k],
                    Bs[0 : (N + block_n - 1) // block_n, 0 : (K + block_k - 1) // block_k],
                )
                T.writes(C[0:M, 0:N])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(M, BLOCK_SIZE_M) * T.ceildiv(N, BLOCK_SIZE_N),),
                    A.data,
                    B.data,
                    C.data,
                    As.data,
                    Bs.data,
                    M,
                    N,
                    K,
                    K,  # stride_am
                    1,  # stride_ak
                    1,  # stride_bk
                    K,  # stride_bn
                    N,  # stride_cm
                    1,  # stride_cn
                    (K + block_k - 1) // block_k,  # stride_As_m
                    1,  # stride_As_k
                    1,  # stride_Bs_k
                    (K + block_k - 1) // block_k,  # stride_Bs_n
                    block_n,
                    block_k,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    GROUP_SIZE_M,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = BlockFP8Matmul.attrs["external_mods"]  # type: ignore  # pylint: disable=no-member
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return BlockFP8Matmul["tir_w8a8_block_fp8_matmul"], tir_name  # type: ignore


def get_tir_w8a8_block_fp8_group_matmul(  # pylint: disable=too-many-arguments,too-many-locals
    N: int,
    K: int,
    num_experts: int,
    block_n: int,
    block_k: int,
    in_dtype: Literal["float8_e4m3fn"],
    out_dtype: Literal["float16", "bfloat16"],
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    GROUP_SIZE_M: int,
    num_warps: int,
    num_stages: int,
    extern_mods: List[tvm.runtime.Module],
):
    """Get the TIR function for the w8a8_block_fp8_group_gemm kernel."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    name_suffix = (
        f"_N{N}_K{K}_num_experts{num_experts}_block_n{block_n}"
        f"_block_k{block_k}_in{in_dtype}_out{out_dtype}"
    )
    kernel_name = f"triton_w8a8_block_fp8_group_gemm{name_suffix}"
    tir_name = f"tir_w8a8_block_fp8_group_gemm{name_suffix}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_w8a8_block_fp8_group_gemm()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class BlockFP8GroupMatmul:  # pylint: disable=missing-class-docstring,too-few-public-methods
        @T.prim_func(private=True)
        def tir_w8a8_block_fp8_group_gemm(  # pylint: disable=missing-function-docstring,too-many-arguments
            var_A: T.handle,
            var_B: T.handle,
            var_As: T.handle,
            var_Bs: T.handle,
            var_expert_ids: T.handle,
            var_indptr: T.handle,
            var_C: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tir.is_scheduled": 1})
            EM = T.SizeVar("EM", "int32")
            A = T.match_buffer(var_A, (EM, K), dtype=in_dtype)
            B = T.match_buffer(var_B, (num_experts, N, K), dtype=in_dtype)
            As = T.match_buffer(var_As, (EM, (K + block_k - 1) // block_k), "float32")
            Bs = T.match_buffer(
                var_Bs,
                (num_experts, (N + block_n - 1) // block_n, (K + block_k - 1) // block_k),
                "float32",
            )
            expert_ids = T.match_buffer(
                var_expert_ids, ((EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,), "int32"
            )
            indptr = T.match_buffer(var_indptr, (num_experts + 1,), "int32")
            C = T.match_buffer(var_C, (EM, N), dtype=out_dtype)

            with T.block("root"):
                T.reads(
                    A[0:EM, 0:K],
                    B[0:num_experts, 0:N, 0:K],
                    As[0:EM, 0 : (K + block_k - 1) // block_k],
                    Bs[
                        0:num_experts,
                        0 : (N + block_n - 1) // block_n,
                        0 : (K + block_k - 1) // block_k,
                    ],
                    expert_ids[0 : (EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts],
                    indptr[0 : num_experts + 1],
                )
                T.writes(C[0:EM, 0:N])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    ((T.ceildiv(EM, BLOCK_SIZE_M) + num_experts) * T.ceildiv(N, BLOCK_SIZE_N),),
                    A.data,
                    B.data,
                    C.data,
                    As.data,
                    Bs.data,
                    expert_ids.data,
                    indptr.data,
                    EM,
                    N,
                    K,
                    num_experts,
                    K,  # stride_am
                    1,  # stride_ak
                    N * K,  # stride_be
                    1,  # stride_bk
                    K,  # stride_bn
                    N,  # stride_cm
                    1,  # stride_cn
                    (K + block_k - 1) // block_k,  # stride_asm
                    1,  # stride_ask
                    ((N + block_n - 1) // block_n) * ((K + block_k - 1) // block_k),  # stride_bse
                    1,  # stride_bsk
                    (K + block_k - 1) // block_k,  # stride_Bs_n
                    block_n,
                    block_k,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    GROUP_SIZE_M,
                    K % BLOCK_SIZE_K == 0,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = BlockFP8GroupMatmul.attrs["external_mods"]  # type: ignore  # pylint: disable=no-member
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return BlockFP8GroupMatmul["tir_w8a8_block_fp8_group_gemm"], tir_name  # type: ignore


def _compute_expert_id_per_block(
    indptr: nn.Tensor,
    num_experts: int,
    M: nn.IntExpr,
    BLOCK_SIZE_M: int,
) -> nn.Tensor:
    """Compute the expert id for each threadblock (CTA).
    We assign an expert id to each threadblock, and the threadblock will
    compute the gemm with regard to the specified expert.

    Parameters
    ----------
    indptr : nn.Tensor
        The indptr tensor of group gemm, with shape of [num_experts + 1,].

    num_experts : int
        The number of total experts.

    M : nn.IntExpr
        The number of tokens.

    BLOCK_SIZE_M : int
        The block size of the threadblock along the batch dimension.

    Returns
    -------
    expert_ids : nn.Tensor
        The expert id for each threadblock, with shape of
        [(M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,].
    """

    @T.prim_func
    def tir_compute_expert_id_per_block(
        var_indptr: T.handle,
        var_expert_ids: T.handle,
        M: T.int64,
    ):
        T.func_attr({"op_pattern": 8, "tir.is_scheduled": 1})
        indptr = T.match_buffer(var_indptr, (num_experts + 1,), "int32")
        expert_ids = T.match_buffer(
            var_expert_ids, ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,), "int32"
        )
        with T.block("root"):
            for eid in T.thread_binding(0, num_experts, thread="threadIdx.x"):
                start_block_id: T.int32 = (indptr[eid] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + eid
                num_blocks: T.int32 = (
                    indptr[eid + 1] - indptr[eid] + BLOCK_SIZE_M - 1
                ) // BLOCK_SIZE_M
                start_block_id_next_expert: T.int32 = (
                    (indptr[eid + 1] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + eid + 1
                )
                for block_id in T.serial(num_blocks):
                    expert_ids[start_block_id + block_id] = eid
                for block_id in T.serial(
                    start_block_id_next_expert - (start_block_id + num_blocks)
                ):
                    expert_ids[start_block_id + num_blocks + block_id] = -1

    assert num_experts <= 1024
    return nn.tensor_ir_op(
        tir_compute_expert_id_per_block,
        "tir_compute_expert_id_per_block",
        args=[indptr, M],
        out=nn.Tensor.placeholder(
            ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,), dtype="int32"
        ),
    )


def fp8_block_scale_gemm(  # pylint: disable=too-many-arguments,too-many-locals
    x: nn.Tensor,
    x_scale: nn.Tensor,
    weight: nn.Tensor,
    weight_scale: nn.Tensor,
    block_size: Tuple[int, int],
    out_dtype: str,
) -> nn.Tensor:
    """Triton block-scale fp8 gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    x_scale : nn.Tensor
        The scale tensor, with shape of [m, k // block_size].

    weight : nn.Tensor
        The weight tensor, with shape of [n, k].

    weight_scale : nn.Tensor
        The scale tensor, with shape of [n // block_size, k // block_size].

    block_size : Tuple[int, int]
        The block size.

    out_dtype : str
        The data type of the output tensor.

    Returns
    -------
    out : nn.Tensor
        The output tensor, with shape of [m, n] and dtype of `out_dtype`.
    """
    assert x.ndim >= 2
    assert weight.ndim == 2
    assert x_scale.ndim == x.ndim
    assert weight_scale.ndim == weight.ndim
    assert x.shape[-1] == weight.shape[1]
    assert x.shape[:-1] == x_scale.shape[:-1]
    assert (x.shape[-1] + block_size[1] - 1) // block_size[1] == x_scale.shape[-1]
    assert (weight.shape[1] + block_size[1] - 1) // block_size[1] == weight_scale.shape[1]
    assert (weight.shape[0] + block_size[0] - 1) // block_size[0] == weight_scale.shape[0]

    if x.dtype != "float8_e4m3fn" or weight.dtype != "float8_e4m3fn":
        raise ValueError(
            f"x and weight must be float8_e4m3fn, but got x={x.dtype}, weight={weight.dtype}"
        )
    if x_scale.dtype != "float32" and weight_scale.dtype != "float32":
        raise ValueError(
            "x_scale and weight_scale must be float32, but got "
            f"x_scale={x_scale.dtype}, weight_scale={weight_scale.dtype}"
        )
    if out_dtype not in ["float16", "bfloat16"]:
        raise ValueError(f"out_dtype must be float16 or bfloat16, but got {out_dtype}")

    M = x.shape[0]
    for i in range(1, x.ndim - 1):
        M *= x.shape[i]
    N = weight.shape[0]
    K = x.shape[-1]

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = block_size[0]
    BLOCK_SIZE_K = block_size[1]
    GROUP_SIZE_M = 32
    num_warps = 4
    num_stages = 3

    x_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(M, K)
    x_scale = x_scale.reshape(M, x_scale.shape[-1])

    out = nn.extern(
        "mlc.triton.w8a8_block_fp8_matmul",
        args=[
            x,
            weight,
            x_scale,
            weight_scale,
            N,
            K,
            block_size[0],
            block_size[1],
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            str(x.dtype),
            str(out_dtype),
        ],
        out=nn.Tensor.placeholder((M, N), dtype=out_dtype),
    )
    return out.reshape(*x_shape[:-1], N) if len(x_shape) > 2 else out


def fp8_block_scale_group_gemm(  # pylint: disable=too-many-arguments,too-many-locals
    x: nn.Tensor,
    x_scale: nn.Tensor,
    weight: nn.Tensor,
    weight_scale: nn.Tensor,
    indptr: nn.Tensor,
    block_size: Tuple[int, int],
    out_dtype: str,
):
    """Triton block-scale fp8 group gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    x_scale : nn.Tensor
        The scale tensor, with shape of [m, k // block_size].

    weight : nn.Tensor
        The weight tensor, with shape of [num_experts, n, k].

    weight_scale : nn.Tensor
        The scale tensor, with shape of [num_experts, n // block_size, k // block_size].

    indptr : nn.Tensor
        The indptr tensor of group gemm, with shape of [num_experts + 1,].

    block_size : Tuple[int, int]
        The block size.

    out_dtype : str
        The data type of the output tensor.

    Returns
    -------
    out : nn.Tensor
        The output tensor, with shape of [m, n] and dtype of `out_dtype`.
    """
    assert x.ndim >= 2
    assert weight.ndim == 3
    assert x_scale.ndim == x.ndim
    assert weight_scale.ndim == weight.ndim
    assert x.shape[-1] == weight.shape[2]
    assert (x.shape[-1] + block_size[1] - 1) // block_size[1] == x_scale.shape[-1]
    assert (weight.shape[2] + block_size[1] - 1) // block_size[1] == weight_scale.shape[2]
    assert (weight.shape[1] + block_size[0] - 1) // block_size[0] == weight_scale.shape[1]

    num_experts = weight.shape[0]
    M = x.shape[0]
    for i in range(1, x.ndim - 1):
        M *= x.shape[i]
    N = weight.shape[1]
    K = x.shape[-1]
    assert weight_scale.shape[0] == num_experts
    assert indptr.ndim == 1
    assert indptr.shape[0] == num_experts + 1

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = block_size[0]
    BLOCK_SIZE_K = block_size[1]
    GROUP_SIZE_M = 32
    num_warps = 4
    num_stages = 3

    x_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(M, K)
    x_scale = x_scale.reshape(M, x_scale.shape[-1])
    expert_ids = _compute_expert_id_per_block(indptr, num_experts, M, BLOCK_SIZE_M)

    out = nn.extern(
        "mlc.triton.w8a8_block_fp8_group_matmul",
        args=[
            x,
            weight,
            x_scale,
            weight_scale,
            expert_ids,
            indptr,
            N,
            K,
            num_experts,
            block_size[0],
            block_size[1],
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            str(x.dtype),
            str(out_dtype),
        ],
        out=nn.Tensor.placeholder((M, N), dtype=out_dtype),
    )
    return out.reshape(*x_shape[:-1], N) if len(x_shape) > 2 else out
