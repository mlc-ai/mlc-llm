"""Exact Hugging Face parameter mapping for the standalone Omni audio component."""

import functools

from mlc_llm.loader import ExternMapping

from .audio_model import Qwen25OmniAudioConfig, Qwen25OmniAudioEncoder, audio_positions


def huggingface(config: Qwen25OmniAudioConfig, dtype: str = "float32") -> ExternMapping:
    """Map audio tower weights; positional constants are generated, not checkpoint tensors."""
    model = Qwen25OmniAudioEncoder(config)
    model.to(dtype)
    _, parameters, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, parameter in parameters:
        if name == "positions":
            mapping.add_mapping(name, [], functools.partial(_positions, config=config, dtype=dtype))
        else:
            mapping.add_mapping(
                name,
                [f"thinker.audio_tower.{name}"],
                functools.partial(_cast, dtype=parameter.dtype),
            )
    # The checkpoint carries these tokens, but HF audio_encoder.forward never uses them.
    mapping.add_unused("thinker.audio_tower.audio_bos_eos_token.weight")
    return mapping


def _positions(*, config, dtype):
    return audio_positions(config).astype(dtype)


def _cast(value, *, dtype):
    return value.astype(dtype)
