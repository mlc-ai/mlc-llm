"""A centralized registry of all existing loaders."""

from typing import Any, Dict  # noqa: UP035

from .huggingface_loader import HuggingFaceLoader

Loader = Any

LOADER: Dict[str, Any] = {  # noqa: UP006
    "huggingface-torch": HuggingFaceLoader,
    "huggingface-safetensor": HuggingFaceLoader,
    "awq": HuggingFaceLoader,
}
