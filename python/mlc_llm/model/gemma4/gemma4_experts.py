"""Gemma4 expert dispatch using MLC's existing grouped and decode MoE kernels.

Inputs contain router-selected expert indices and already-scaled routing weights.
This component does not enable the full decoder or change cache semantics.
"""

from dataclasses import dataclass

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.loader import ExternMapping
from mlc_llm.nn.expert import MixtralExperts


@dataclass
class Gemma4ExpertsConfig:
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    top_k_experts: int

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in vars(self).values()):
            raise ValueError("Expert dimensions must be positive integers")
        if self.top_k_experts > self.num_experts:
            raise ValueError("Selected experts cannot exceed expert count")


class Gemma4Experts(nn.Module):
    """Float32 expert component; indices must be unique per row and in range."""

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

    def _combine(self, hidden, indices, weights, decode):
        op_ext.configure()
        tokens = hidden.shape[0]
        c = self.config
        if decode:
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
        return op_ext.moe_misc.moe_sum(result, dim=1)

    def prefill(self, hidden: Tensor, indices: Tensor, weights: Tensor):
        return self._combine(hidden, indices, weights, False)

    def decode(self, hidden: Tensor, indices: Tensor, weights: Tensor):
        return self._combine(hidden, indices, weights, True)

    def to(self, dtype=None):
        if dtype not in (None, "float32"):
            raise ValueError("Standalone expert verification supports float32 only")
        return super().to(dtype)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                name: {
                    "hidden": nn.spec.Tensor((tokens, self.config.hidden_size), "float32"),
                    "indices": nn.spec.Tensor((tokens, self.config.top_k_experts), "int32"),
                    "weights": nn.spec.Tensor((tokens, self.config.top_k_experts), "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
                for name, tokens in (("prefill", "tokens"), ("decode", 1))
            },
            self,
        )


def huggingface(config, layer_idx):
    if type(layer_idx) is not int or layer_idx < 0:
        raise ValueError("Invalid expert layer index")
    model = Gemma4Experts(config)
    _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, _ in params:
        source = f"model.language_model.layers.{layer_idx}.experts.{name.removesuffix('.weight')}"
        mapping.add_mapping(name, [source], lambda value: value.astype("float32", copy=False))
    return mapping
