"""Global namespace of conversation template registry"""

# TODO(mlc-team): move conversation template apply to this namespace
# decouple conversation template apply from the conversation protocol
# data structure

# model preset templates
from . import (
    cohere,
    deepseek,
    dolly,
    gemma,
    glm,
    gorrilla,
    gpt,
    hermes,
    llama,
    llava,
    llm_jp,
    ministral3,
    ministral3_reasoning,
    mistral,
    nemotron,
    oasst,
    olmo,
    orion,
    phi,
    qwen2,
    redpajama,
    rwkv,
    stablelm,
    tinyllama,
    wizardlm,
)
from .registry import ConvTemplateRegistry
