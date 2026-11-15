"""Qwen2.5 Omni Thinker component for text and audio embeddings.

Audio-only multimodal positions repeat the ordinary position on all three MRoPE
axes. The existing Qwen2 cache therefore implements this restricted path. Image
and video positions are not supported by this component.
"""

import dataclasses

from mlc_llm.model.qwen2.qwen2_model import QWen2Config, QWen2LMHeadModel


@dataclasses.dataclass
class Qwen25OmniThinkerConfig(QWen2Config):
    """Dense Thinker configuration restricted to ordinary text/audio positions."""

    input_modalities: tuple[str, ...] = ("text", "audio")

    @classmethod
    def from_dict(cls, source):
        source = dict(source)
        thinker = source.get("thinker_config", source)
        if not isinstance(thinker, dict):
            raise ValueError("thinker_config must be an object")
        text = thinker.get("text_config", thinker)
        if not isinstance(text, dict):
            raise ValueError("Thinker text_config must be an object")
        flattened = dict(text)
        # Compilation overrides live outside the nested Hugging Face text config.
        for field in dataclasses.fields(cls):
            if field.name in source and field.name not in ("kwargs",):
                flattened[field.name] = source[field.name]
        return super().from_dict(flattened)

    def __post_init__(self):
        self.input_modalities = tuple(self.input_modalities)
        if self.input_modalities not in (("text",), ("text", "audio")):
            raise ValueError("Thinker supports only text and audio input modalities")
        if self.kwargs.get("use_sliding_window", False) or any(
            kind != "full_attention" for kind in self.kwargs.get("layer_types", [])
        ):
            raise ValueError("Thinker sliding-window attention is not supported")
        for name in ("rope_scaling", "rope_parameters"):
            rope = self.kwargs.get(name)
            if rope is None:
                continue
            if not isinstance(rope, dict):
                raise ValueError("Thinker RoPE settings must be an object")
            if any(rope.get(key, "default") != "default" for key in ("type", "rope_type")):
                raise ValueError("Thinker requires default RoPE for text/audio positions")
            if rope.get("factor", 1) != 1 or rope.get("partial_rotary_factor", 1) != 1:
                raise ValueError("Thinker scaled or partial RoPE is not supported")
            if "rope_theta" in rope:
                self.rope_theta = rope["rope_theta"]
        super().__post_init__()
        if self.kwargs.get("attention_bias", True) is not True:
            raise ValueError("Thinker requires Qwen2 query/key/value projection biases")


class Qwen25OmniThinker(QWen2LMHeadModel):
    """Token embeddings and stateful generation over text/audio embeddings.

    The caller must replace audio placeholder embeddings before prefill. This is
    not a full Omni model: it provides no media processing or Talker exports.
    """
