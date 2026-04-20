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
    gorilla,
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
    olmo2,
    orion,
    phi,
    qwen2,
    qwen3,
    qwen3_5,
    redpajama,
    rwkv,
    stablelm,
    tinyllama,
    wizardlm,
)
from .registry import ConvTemplateRegistry
