"""Tests for malformed image data URL handling in ``ImageData.from_url``."""

import pytest

from mlc_llm.protocol.error_protocol import BadRequestError
from mlc_llm.serve.data import ImageData

# A minimal valid 1x1 RGB PNG, base64-encoded. Used as the regression
# fixture for the valid-URL case.
VALID_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def test_malformed_data_url_no_comma_raises_bad_request():
    """A ``data:image`` URL without a comma cannot be split.

    On master this raises ``IndexError``; the fix surfaces a
    ``BadRequestError`` (HTTP 400) instead of a 500 traceback.
    """
    with pytest.raises(BadRequestError):
        ImageData.from_url("data:image/png", {"model_type": "llava"})


def test_malformed_data_url_invalid_base64_raises_bad_request():
    """An undecodable base64 payload in a ``data:image`` URL surfaces 400."""
    with pytest.raises(BadRequestError):
        ImageData.from_url("data:image/png;base64,@@@", {"model_type": "llava"})


def test_valid_data_url_does_not_raise_bad_request():
    """A valid ``data:image`` URL must not hit the new BadRequestError path."""
    url = f"data:image/png;base64,{VALID_PNG_B64}"
    image = ImageData.from_url(url, {"model_type": "llava"})
    assert image is not None
