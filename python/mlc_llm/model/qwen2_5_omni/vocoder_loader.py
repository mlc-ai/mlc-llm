"""Exact checkpoint mapping for the standalone Qwen2.5 Omni waveform decoder."""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping

from .vocoder_model import Qwen25OmniVocoder, Qwen25OmniVocoderConfig, vocoder_filter


def huggingface(config: Qwen25OmniVocoderConfig) -> ExternMapping:
    model = Qwen25OmniVocoder(config)
    _, parameters, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, parameter in parameters:
        if name.endswith(".filter"):
            mapping.add_mapping(name, [], functools.partial(_filter, channels=parameter.shape[0]))
        else:
            mapping.add_mapping(name, [f"token2wav.code2wav_bigvgan_model.{name}"], _cast)
    return mapping


def _filter(*, channels):
    return np.repeat(vocoder_filter(), channels, axis=0)


def _cast(value):
    return value.astype("float32", copy=False)
