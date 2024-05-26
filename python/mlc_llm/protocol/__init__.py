"""Definitions of pydantic models for API entry points and configurations"""
from . import openai_api_protocol

RequestProtocol = openai_api_protocol.CompletionRequest
