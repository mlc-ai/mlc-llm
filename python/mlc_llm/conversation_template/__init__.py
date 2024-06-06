"""Global namespace of conversation template registry"""

# TODO(mlc-team): move conversation template apply to this namespace
# decouple conversation template apply from the conversation protocol
# data structure


# model preset templates
from . import (
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
    redpajama,
    rwkv,
    stablelm,
    tinyllama,
    wizardlm,
)
from .registry import ConvTemplateRegistry
