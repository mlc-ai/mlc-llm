"""An nn.Module that represents MoE experts"""

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor

from mlc_llm.op import cutlass, extern, ft_gemm, moe_matmul


class MixtralExperts(nn.Module):
    """Mixtral experts"""

    def __init__(self, num_local_experts, in_features, out_features, tensor_parallel_shards=1):
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((num_local_experts, out_features, in_features))
        self.dtype = "float32"
        self.tensor_parallel_shards = tensor_parallel_shards

    def forward(self, x: Tensor, indptr: Tensor):  # pylint: disable=invalid-name,missing-docstring
        assert x.ndim == 2
        if indptr.ndim == 2:
            assert indptr.shape[0] == 1
            return moe_matmul.gemv(x, self.weight, indptr)
        assert indptr.ndim == 1
        if extern.get_store().cutlass_group_gemm and self.dtype == "float16":
            return cutlass.group_gemm(x, self.weight, indptr)
        if extern.get_store().faster_transformer and self.dtype == "float16":
            return ft_gemm.faster_transformer_moe_gemm(x, self.weight, indptr)
        return moe_matmul.group_gemm(x, self.weight, indptr)
