"""A centralized registry of all existing loaders."""

from typing import Any, Dict

from .huggingface_loader import HuggingFaceLoader

Loader = Any

LOADER: Dict[str, Any] = {
    "huggingface-torch": HuggingFaceLoader,
    "huggingface-safetensor": HuggingFaceLoader,
    "awq": HuggingFaceLoader,
}
