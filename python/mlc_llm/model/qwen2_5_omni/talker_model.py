"""Standalone text/audio Talker decoder with explicit embedding projection.

The host supplies aligned Thinker states plus codec embeddings, including the
initial codec pad/BOS additions. Prefill/decode accept projected hidden states;
image/video MRoPE, reply alignment, and codec sampling are not implemented here.
"""

import dataclasses

from tvm.relax.frontend import nn

from .thinker_model import Qwen25OmniThinker, Qwen25OmniThinkerConfig


@dataclasses.dataclass
class Qwen25OmniTalkerConfig(Qwen25OmniThinkerConfig):
    """Talker configuration restricted to ordinary text/audio positions."""

    embedding_size: int = 2048

    @classmethod
    def from_dict(cls, source):
        source = dict(source)
        nested = source.get("talker_config", source)
        if not isinstance(nested, dict):
            raise ValueError("talker_config must be an object")
        flattened = dict(nested)
        for field in dataclasses.fields(cls):
            if field.name in source and field.name != "kwargs":
                flattened[field.name] = source[field.name]
        return super().from_dict(flattened)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.embedding_size, bool) or not isinstance(self.embedding_size, int):
            raise ValueError("Talker embedding_size must be a positive integer")
        if self.embedding_size <= 0 or self.tensor_parallel_shards != 1:
            raise ValueError(
                "Talker requires positive embedding_size and one tensor-parallel shard"
            )


class Qwen25OmniTalker(Qwen25OmniThinker):
    """Codec embeddings, Thinker projection and cached codec-logit decoder."""

    def __init__(self, config):
        super().__init__(config)
        self.embedding_size = config.embedding_size
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size)
        self.thinker_to_talker_proj = nn.Linear(
            config.embedding_size, config.hidden_size, bias=True, out_dtype="float32"
        )

    def project(self, embeddings):
        """Project host-aligned Thinker-plus-codec embeddings before cache use."""
        return self.thinker_to_talker_proj(embeddings).astype(self.dtype)

    def get_default_spec(self):
        inherited = super().get_default_spec()
        methods = dict(zip(inherited.method_names, inherited.method_specs))
        methods["project"] = {
            "embeddings": nn.spec.Tensor(["seq_len", self.embedding_size], self.dtype),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        }
        return nn.spec.ModuleSpec.from_raw(methods, self)
