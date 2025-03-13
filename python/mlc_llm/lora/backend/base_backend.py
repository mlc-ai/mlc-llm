from typing import Tuple, Union
from tvm.relax.frontend.nn import Tensor


class BaseLoraBackend:
    """Base class for different lora backends.
    Each backend has its own implementation of Lora kernels.
    """

    def __init__(self):
        pass

    def run_lora_a_agent(
        self, x: Tensor, weights: Tensor, *args, **kwargs
    ) -> Tensor:
        """Run segment Gemm of lora a modules with current backend.

        Args:
            x: input matrix with shape (b, s, input_dim), here b is batch size and
            s is the sequence length
            weights: a set of lora weights with shape (num_lora, r, input_dim), here r is lora rank
            usually input_dim is much larger than r

        Returns:
            result with shape (b, s, r)
        """
        pass

    def gate_up_lora_b_fwd(
        self, x: Tensor, weights: Tensor, *args, **kwargs
    ) -> Tensor:
        """Run segment Gemm of lora b modules with current backend.

        Args:
            x: input matrix with shape (b, s, r), here b is batch size and
            s is the sequence length, r is lora rank
            weights: a set of lora weights with shape (num_lora, output_dim, r)
            usually output_dim is much larger than r

        Returns:
            result with shape (b, s, output_dim)
        """
        pass

    def qkv_lora_b_fwd(
        self,
        x: Tensor,
        qkv_lora_b: Union[Tensor, Tuple[Tensor, Tensor]],
        *args,
        **kwargs
    ) -> Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (b, s, input_dim), here b is batch size and
            s is the sequence length
            qkv_lora_b: lora_b module for qkv.
            If passed in as a tensor, its shape should be (num_lora, output_dim_q + 2 * output_dim_kv, r)
            If passed in as a tuple of two tensors containing:
            a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
            and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)

        Returns:
            result with shape (b, s, output_dim_q + 2 * output_dim_kv)
        """
        pass