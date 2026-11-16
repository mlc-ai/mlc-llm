"""Gemma4 MoE routing and parallel feed-forward composition components.

Expert dispatch/kernels and the full decoder are intentionally not enabled here.
Source: Transformers e7200cd9a98f0c874edcdf8f7263b7d19530b403 Gemma4TextRouter
and Gemma4TextDecoderLayer. All component arithmetic is float32.
"""

from dataclasses import dataclass

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.loader import ExternMapping

from .gemma4_audio import Gemma4RMSNorm


@dataclass
class Gemma4MoEComponentConfig:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k_experts: int
    rms_norm_eps: float = 1e-6
    hidden_activation: str = "gelu_pytorch_tanh"
    enable_moe_block: bool = True
    num_kv_shared_layers: int = 0

    def __post_init__(self):
        for value in (
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.top_k_experts,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("MoE dimensions must be positive integers")
        if self.top_k_experts > self.num_experts:
            raise ValueError("Top-k cannot exceed expert count")
        if self.enable_moe_block is not True or self.num_kv_shared_layers != 0:
            raise ValueError("Components require unshared MoE decoder layers")
        if self.hidden_activation != "gelu_pytorch_tanh" or not 0 < self.rms_norm_eps < 1:
            raise ValueError("Unsupported MoE activation or normalization")

    @classmethod
    def from_text_config(cls, config):
        return cls(**{key: config[key] for key in cls.__dataclass_fields__ if key in config})


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

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "hidden": nn.spec.Tensor(("tokens", self.config.hidden_size), "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )


class DenseMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(
            op.gelu(self.gate_proj(hidden), approximate="tanh") * self.up_proj(hidden)
        )


class Gemma4ParallelFeedForward(nn.Module):
    """Compose dense and already-routed expert outputs, before layer_scalar.

    `expert_output` must be the weighted expert result computed from
    `pre_feedforward_layernorm_2(hidden)`, with routing based on raw `hidden`.
    The separate `expert_input` entry point exposes that exact input transform.
    """

    def __init__(self, config):
        self.config = config
        self.mlp = DenseMLP(config)
        for name in (
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ):
            setattr(self, name, Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps))

    def expert_input(self, hidden: Tensor):
        return self.pre_feedforward_layernorm_2(hidden)

    def forward(self, hidden: Tensor, expert_output: Tensor):
        dense = self.post_feedforward_layernorm_1(self.mlp(self.pre_feedforward_layernorm(hidden)))
        experts = self.post_feedforward_layernorm_2(expert_output)
        return hidden + self.post_feedforward_layernorm(dense + experts)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "hidden": nn.spec.Tensor(("tokens", self.config.hidden_size), "float32"),
                    "expert_output": nn.spec.Tensor(("tokens", self.config.hidden_size), "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "expert_input": {
                    "hidden": nn.spec.Tensor(("tokens", self.config.hidden_size), "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
            },
            self,
        )


def component_huggingface(model, layer_idx: int) -> ExternMapping:
    if type(layer_idx) is not int or layer_idx < 0:
        raise ValueError("Invalid layer index")
    prefix = f"model.language_model.layers.{layer_idx}."
    if isinstance(model, Gemma4Router):
        prefix += "router."
    elif not isinstance(model, Gemma4ParallelFeedForward):
        raise ValueError("Unsupported MoE component")
    _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, _ in params:
        mapping.add_mapping(name, [prefix + name], _float32)
    return mapping


def _float32(value):
    return value.astype("float32", copy=False)
