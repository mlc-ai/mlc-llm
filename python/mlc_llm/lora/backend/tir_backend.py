from tvm.relax.frontend.nn import Tensor, op
from mlc_llm import op as op_exit
from mlc_llm.lora.backend.base_backend import BaseLoraBackend


class TIRLoraBackend(BaseLoraBackend):  # pylint: disable=too-few-public-methods
    def __init__(self):
        pass

    def sgemm_lora_a_fwd(
        self, x: Tensor, weight: Tensor, seq_lens: Tensor, seq_indptr: Tensor, weight_indices: Tensor
    ) -> Tensor:
        """Run segment Gemm of lora a modules with current backend.

        Args:
            x: input matrix with shape (b, s, input_din) or (s, input_din) for lm_head
            weight: a set of lora weights with shape (num_lora, r, input_din)
            output: (b, s, r)
            when called by run_dkv_lora, the weights.shape[-2] will be 3 * r
            input_din is much larger than r
        """
        # pylint: disable=protected-access
        s, in_features = x.shape[-2], x.shape[-1]
        b = x.shape[0] if len(x.shape) == 3 else 1
        x = x.reshape(b * s, in_features)
        return op_exit.segment_gemm(x, weight, seq_lens, weight_indices).reshape(b, s, weight.shape[1])

    def gate_up_lora_b_fwd(
        self, gate_up: Tensor, weight: Tensor, seq_lens: Tensor, seq_indptr: Tensor, weight_indices: Tensor
    ) -> Tensor:
        """Run segment Gemm of lora b modules with current backend.

        Args:
            gate_up: (b, s, r)
            weights: (num_lora, output_din, r)
            output: (b, s, output_din) output_din is much larger than r
        """
        # pylint: disable=protected-access
        gate, up = op.split(gate_up, 2, axis=-1)
        gate_weight, up_weight = op.split(weight, 2, axis=-2)
        gate_out = self.sgemm_lora_a_fwd(gate, gate_weight, seq_lens, seq_indptr, weight_indices)
        up_out = self.sgemm_lora_a_fwd(up, up_weight, seq_lens, seq_indptr, weight_indices)
        return op.concat([gate_out, up_out], dim=-1)

    def qkv_lora_b_fwd(
        self,
        qkv: Tensor,
        qkv_weight: Tensor,
        q_output_din: int,
        kv_output_din: int,
        seq_lens: Tensor,
        seq_indptr: Tensor,
        weight_indices: Tensor,
    ) -> Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            qkv: (b, s, 3 * r)
            qkv_weight: (num_lora, q_output_din + 2 * kv_output_din, r)
            output: (b, s, q_output_din + 2 * kv_output_din)
        """
        # pylint: disable=protected-access
        q, k, v = op.split(qkv, 3, axis=-1)
        q_weight, k_weight, v_weight = op.split(qkv_weight, [q_output_din, q_output_din + kv_output_din], axis=-2)
        q_out = self.sgemm_lora_a_fwd(q, q_weight, seq_lens, seq_indptr, weight_indices)
        k_out = self.sgemm_lora_a_fwd(k, k_weight, seq_lens, seq_indptr, weight_indices)
        v_out = self.sgemm_lora_a_fwd(v, v_weight, seq_lens, seq_indptr, weight_indices)
        return op.concat([q_out, k_out, v_out], dim=-1)