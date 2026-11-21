"""Tests for malformed image data URL handling in ``ImageData.from_url``."""

import pytest

from mlc_llm.protocol.error_protocol import BadRequestError
from mlc_llm.serve.data import ImageData

# A minimal valid 1x1 RGB PNG, base64-encoded. Used as the regression
# fixture for the valid-URL case.
VALID_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

# The same 1x1 PNG truncated to its 41-byte header: ``Image.open`` succeeds
# because only the header is parsed, while ``convert("RGB")`` raises a plain
# ``OSError("image file is truncated")`` -- not ``UnidentifiedImageError``.
TRUNCATED_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQ="


def test_malformed_data_url_no_comma_raises_bad_request():
    """A ``data:image`` URL without a comma cannot be split.

    On master this raises ``IndexError``; the fix surfaces a
    ``BadRequestError`` (HTTP 400) instead of a 500 traceback.
    """
    with pytest.raises(BadRequestError):
        ImageData.from_url("data:image/png", {"model_type": "llava"})


def test_malformed_data_url_invalid_base64_raises_bad_request():
    """An undecodable base64 payload in a ``data:image`` URL surfaces 400.

    The payload has incorrect base64 padding (length 5), so
    ``base64.b64decode`` raises ``binascii.Error`` before ``Image.open``
    is reached.
    """
    with pytest.raises(BadRequestError):
        ImageData.from_url("data:image/png;base64,AAAAA", {"model_type": "llava"})


def test_valid_data_url_does_not_raise_bad_request():
    """A valid ``data:image`` URL must not hit the new BadRequestError path."""
    url = f"data:image/png;base64,{VALID_PNG_B64}"
    image = ImageData.from_url(url, {"model_type": "llava"})
    assert image is not None


def test_truncated_data_url_raises_bad_request():
    """A truncated but header-valid image in a ``data:image`` URL surfaces 400.

    ``Image.open`` only parses the header, so ``UnidentifiedImageError`` is
    never raised here; the truncation surfaces as a plain ``OSError`` from
    ``convert("RGB")``. On master that propagates as an HTTP 500.
    """
    url = f"data:image/png;base64,{TRUNCATED_PNG_B64}"
    with pytest.raises(BadRequestError):
        ImageData.from_url(url, {"model_type": "llava"})


def test_corrupt_http_image_raises_bad_request(monkeypatch):
    """A fetched ``http`` image whose body is not a valid image surfaces 400.

    On master, a fetched-but-corrupt response body makes ``Image.open``
    raise ``UnidentifiedImageError`` and propagate as an HTTP 500; the fix
    surfaces a ``BadRequestError`` (HTTP 400) instead. The network is faked
    by patching ``requests.get`` to return a corrupt body; the function
    under fix (``from_url``) itself is not mocked.
    """
    import requests

    class _FakeCorruptResponse:
        content = b"not a valid image"

    def _fake_get(*args, **kwargs):
        return _FakeCorruptResponse()

    monkeypatch.setattr(requests, "get", _fake_get)
    with pytest.raises(BadRequestError):
        ImageData.from_url("http://example.com/corrupt.png", {"model_type": "llava"})


def test_truncated_http_image_raises_bad_request(monkeypatch):
    """A fetched truncated-but-header-valid image surfaces 400.

    Companion of ``test_truncated_data_url_raises_bad_request`` for the
    ``http`` branch: the body parses far enough for ``Image.open`` to open
    it, then fails decoding with a plain ``OSError``.
    """
    import base64

    import requests

    class _FakeTruncatedResponse:
        content = base64.b64decode(TRUNCATED_PNG_B64)

    def _fake_get(*args, **kwargs):
        return _FakeTruncatedResponse()

    monkeypatch.setattr(requests, "get", _fake_get)
    with pytest.raises(BadRequestError):
        ImageData.from_url("http://example.com/truncated.png", {"model_type": "llava"})
