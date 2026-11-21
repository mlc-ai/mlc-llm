"""Expert routing and dispatch for large Gemma 4 text models."""

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn.expert import MixtralExperts

from .gemma4_audio import Gemma4RMSNorm


class Gemma4Router(nn.Module):
    def __init__(self, config):
        self.config = config
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter((config.hidden_size,))
        self.per_expert_scale = nn.Parameter((config.num_experts,))

    def forward(self, hidden: Tensor):
        values = self.norm(hidden) * self.scale * self.config.hidden_size**-0.5
        probabilities = op.softmax(op.astype(self.proj(values), "float32"), axis=-1)
        weights, indices = op.topk(probabilities, k=self.config.top_k_experts, dtype="int32")
        weights = weights / op.sum(weights, axis=-1, keepdims=True)
        weights = weights * op.astype(
            op.take(self.per_expert_scale, indices, axis=0), weights.dtype
        )
        return probabilities, weights, indices


class Gemma4LargeExperts(nn.Module):
    def __init__(self, config):
        self.config = config
        self.gate_up_proj = MixtralExperts(
            config.num_experts, config.hidden_size, 2 * config.moe_intermediate_size
        )
        self.down_proj = MixtralExperts(
            config.num_experts, config.moe_intermediate_size, config.hidden_size
        )

    def _expert_forward(self, hidden, indptr):
        gate, up = op.split(self.gate_up_proj(hidden, indptr), 2, axis=-1)
        return self.down_proj(op.gelu(gate, approximate="tanh") * up, indptr)

    def forward(self, hidden, indices, weights):
        op_ext.configure()
        tokens = hidden.shape[0]
        c = self.config
        if tokens == 1:
            # A single token can dispatch directly to its selected experts.
            result = self._expert_forward(hidden, indices)
        else:
            cumsum = op_ext.moe_misc.moe_cumsum(indices, c.num_experts)
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, indices)
            indptr = op_ext.moe_misc.get_indptr(
                cumsum, c.num_experts, tokens, inclusive=False, out_dtype="int32"
            )
            dispatched = op.take(hidden, token_indices, axis=0)
            result = self._expert_forward(dispatched, indptr)
            result = op_ext.moe_misc.scatter_output(result, reverse_indices)
        result = op.reshape(result, (tokens, c.top_k_experts, c.hidden_size))
        result = op.astype(result, weights.dtype) * op.reshape(
            weights, (tokens, c.top_k_experts, 1)
        )
        result = op.astype(result, hidden.dtype)
        return op.astype(op_ext.moe_misc.moe_sum(result, dim=1), hidden.dtype)
