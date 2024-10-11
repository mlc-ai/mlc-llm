"""Global namespace of conversation template registry"""

# TODO(mlc-team): move conversation template apply to this namespace
# decouple conversation template apply from the conversation protocol
# data structure


# model preset templates
from . import (
    cohere,
    dolly,
    gemma,
    glm,
    gorrilla,
    gpt,
    hermes,
    llama,
    llava,
    mistral,
    oasst,
    orion,
    phi,
    qwen2,
    redpajama,
    rwkv,
    stablelm,
    tinyllama,
    wizardlm,
    deepseek_v2,
)
from .registry import ConvTemplateRegistry
