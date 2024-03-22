""" Quantization techniques for FP8 """
import numpy as np
from tvm import nd, relax
from tvm.relax.frontend import nn

from mlc_llm.nn import MixtralExperts

from ..op import cutlass, extern, moe_matmul
from . import per_tensor_quantization as ptq
from .utils import apply_sharding


class FP8PerTensorQuantizeMixtralExperts(
    ptq.PerTensorQuantizeMixtralExperts
):  # pylint: disable=too-many-instance-attributes
    """ MixtralExperts with per-tensor quantization in FP8. """
    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        config: ptq.PerTensorQuantize,
        tensor_parallel_shards=1,
    ):  # pylint: disable=too-many-arguments
        super().__init__(num_local_experts, in_features, out_features, config)
        self.tensor_parallel_shards = tensor_parallel_shards

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts",
        config: ptq.PerTensorQuantize,
    ) -> "FP8PerTensorQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a per-tensor quantized MixtralExperts.

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        weight_config : GroupQuantize
            The group quantization weight_config.

        Returns
        -------
        ret : MixtralExpertsFP8
            The per-tensor quantized MixtralExperts.
        """
        quantized_mistral_experts = FP8PerTensorQuantizeMixtralExperts(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
            tensor_parallel_shards=src.tensor_parallel_shards,
        )

        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_mistral_experts.q_weight)
            # scale doesn't need to be sharded since it's the same for all shards

        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        w = self.q_weight
        if indptr.ndim == 2:
            assert indptr.shape[0] == 1
            return moe_matmul.dequantize_float8_gemv(
                x, w, self.q_scale, indptr, self.config.weight_dtype
            )

        if extern.get_store().cutlass_group_gemm:
            # TODO: calibration scale should be used to convert x to fp8
            x = nn.op.astype(x, dtype=self.config.activation_dtype)
            scale = (
                self.q_scale.astype("float32")
                if self.q_scale is not None
                else nn.wrap_nested(
                    relax.Constant(nd.array(np.array([1.0]).astype("float32"))), "scale"
                )
            )
            return cutlass.group_gemm(
                x, w, indptr, scale, self.config.weight_dtype, self.config.model_dtype
            )
        else:
            # Note: convert_weight is target agnostic, so a fallback must be provided
            w =nn.tensor_expr_op(self.config.dequantize_float8, "dequantize", args=[w, self.q_scale, self.config.weight_dtype])
            return moe_matmul.group_gemm(x, w, indptr)
