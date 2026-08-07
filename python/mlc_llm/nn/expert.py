"""An nn.Module that represents MoE experts"""

from mlc_llm.op import cutlass, extern, ft_gemm, moe_matmul
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


class MixtralExperts(nn.Module):
    """Mixtral experts"""

    def __init__(self, num_local_experts, in_features, out_features, tensor_parallel_shards=1):
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((num_local_experts, out_features, in_features))
        self.dtype = "float32"
        self.tensor_parallel_shards = tensor_parallel_shards

    def forward(self, x: Tensor, indptr: Tensor):
        assert x.ndim == 2
        if indptr.ndim == 2:
            assert indptr.shape[0] == 1
            return moe_matmul.gemv(x, self.weight, indptr)
        assert indptr.ndim == 1
        if extern.get_store().cutlass_group_gemm and self.dtype in [
            "float16",
            "bfloat16",
        ]:
            return cutlass.group_gemm(x, self.weight, indptr)
        if extern.get_store().faster_transformer and self.dtype == "float16":
            return ft_gemm.faster_transformer_moe_gemm(x, self.weight, indptr)
        return moe_matmul.group_gemm(x, self.weight, indptr)


class GptOssMxfp4Experts(nn.Module):
    """MXFP4-quantized MoE expert projection layer for GptOss.
    Mirrors the interface of ``MixtralExperts`` so it can be used as a
    drop-in replacement inside ``GptOssMLP``.  Weights are stored in the
    GptOss MXFP4 format:
    * ``weight_blocks`` - uint32, shape ``(num_experts, in_features//8, out_features)``.
      Each uint32 packs 8 x FP4 nibbles along the K (in_features) axis.
    * ``weight_scales`` - float16, shape ``(num_experts, in_features//group_size, out_features)``.
      One scale per quantisation group along K.
    * ``bias``          - float16, shape ``(num_experts, out_features)``.
      Per-expert output bias added after the matmul.
    The weight layout is **KN** (K = in_features packed, N = out_features),
    matching the GptOss checkpoint convention.
    """

    no_quantization: bool = True  # tell MLC not to re-quantize these weights

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        group_size: int = 32,
    ):
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = "float16"

        # MXFP4 packed weights: 8 FP4 nibbles per uint32 along K (KN layout)
        self.weight_blocks = nn.Parameter(
            (num_local_experts, in_features // 8, out_features), "uint32"
        )
        # Per-group scales along K
        self.weight_scales = nn.Parameter(
            (num_local_experts, in_features // group_size, out_features), "float16"
        )
        # Per-expert output bias
        self.bias = nn.Parameter((num_local_experts, out_features), dtype="float16")

    def forward(self, x: Tensor, indptr: Tensor) -> Tensor:  # pylint: disable=missing-docstring
        if x.ndim == 3 and indptr.ndim == 2:
            # ------------------------------------------------------------------
            # Decode path
            #   x      : (batch, 1_or_experts_per_tok, in_features)
            #   indptr : (batch, experts_per_tok)   -- expert indices
            # Returns  : (batch, experts_per_tok, out_features)
            # ------------------------------------------------------------------
            out = moe_matmul.dequantize_mxfp4_gemv(
                x,
                self.weight_blocks,
                self.weight_scales,
                indptr,
                self.group_size,
            )
            # Add per-expert bias: gather bias[indptr[b, e]] for each slot
            # indptr : (batch, experts_per_tok)
            # bias   : (num_experts, out_features)
            batch, experts_per_tok = indptr.shape
            bias_gathered = op.take(
                self.bias, indptr.reshape(-1), axis=0
            )  # (batch * experts_per_tok, out_features)
            bias_gathered = bias_gathered.reshape(batch, experts_per_tok, self.out_features)
            return out + bias_gathered

        assert indptr.ndim == 1, (
            "GptOssMxfp4Experts.forward: expected indptr.ndim==1 for prefill path, "
            f"got {indptr.ndim}"
        )
        # ------------------------------------------------------------------
        # Prefill path
        #   x      : (num_tokens * experts_per_tok, in_features)
        #   indptr : (num_experts + 1,) int32  or  (num_experts,) int64
        # Returns  : (num_tokens * experts_per_tok, out_features)
        # Bias is fused directly into the group-GEMM store epilogue.
        # ------------------------------------------------------------------
        indptr_dtype = indptr.dtype
        return moe_matmul.dequantize_mxfp4_group_gemm(
            x,
            self.weight_blocks,
            self.weight_scales,
            self.bias,
            indptr,
            indptr_dtype,
            self.group_size,
        )
