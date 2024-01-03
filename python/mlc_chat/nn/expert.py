"""An nn.Module that represents MoE experts"""
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor

from mlc_chat.op import moe_matmul


class MixtralExperts(nn.Module):
    """Mixtral experts"""

    def __init__(self, num_local_experts, in_features, out_features):
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((num_local_experts, out_features, in_features))
        self.dtype = "float32"

    def forward(self, x: Tensor, indptr: Tensor):  # pylint: disable=invalid-name,missing-docstring
        assert x.ndim == 2
        if indptr.ndim == 2:
            assert indptr.shape[0] == 1
            return moe_matmul.gemv(x, self.weight, indptr)
        assert indptr.ndim == 1
        return moe_matmul.group_gemm(x, self.weight, indptr)
