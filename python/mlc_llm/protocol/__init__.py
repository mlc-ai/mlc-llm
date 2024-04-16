"""The protocols for MLC LLM server"""
from . import openai_api_protocol

RequestProtocol = openai_api_protocol.CompletionRequest
