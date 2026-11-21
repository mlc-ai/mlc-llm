"""Exact DiT flow weights; the separate ECAPA conditioner is intentionally excluded."""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping

from .dit_model import Qwen25OmniDiT, Qwen25OmniDiTConfig


def huggingface(config: Qwen25OmniDiTConfig) -> ExternMapping:
    model = Qwen25OmniDiT(config)
    _, parameters, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, _ in parameters:
        if name == "rotary_embed.inv_freq":
            mapping.add_mapping(name, [], functools.partial(rotary_frequencies, config))
            continue
        mapping.add_mapping(
            name,
            [f"token2wav.code2wav_dit_model.{name}"],
            lambda value: value.astype("float32", copy=False),
        )
    return mapping


def rotary_frequencies(config: Qwen25OmniDiTConfig):
    """HF recomputes this non-persistent buffer rather than loading legacy BF16 values."""
    return np.reciprocal(
        np.power(
            np.float32(config.rope_theta),
            np.arange(0, config.head_dim, 2, dtype="float32") / config.head_dim,
        )
    ).astype("float32")
