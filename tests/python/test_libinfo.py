"""Unit tests for :mod:`mlc_llm.libinfo`.

``libinfo.py`` is intentionally standalone (it is also ``exec``-ed by ``setup.py``
before the package is importable), so it is loaded here directly from its file path.
This keeps the test independent of a fully built ``mlc_llm`` / ``tvm`` install.
"""

import importlib.util
import os

import pytest

pytestmark = [pytest.mark.unittest]

_LIBINFO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "python", "mlc_llm", "libinfo.py"
)


def _load_libinfo():
    spec = importlib.util.spec_from_file_location("mlc_llm_libinfo_standalone", _LIBINFO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_lib_reports_missing_dependency(monkeypatch):
    """A failed dependency resolution should surface an actionable RuntimeError."""
    libinfo = _load_libinfo()

    def _raise_missing_dep(_path):
        raise OSError("libtvm.so: cannot open shared object file: No such file or directory")

    monkeypatch.setattr(libinfo.ctypes, "CDLL", _raise_missing_dep)

    with pytest.raises(RuntimeError) as exc_info:
        libinfo.load_lib("/some/path/libmlc_llm.so")

    message = str(exc_info.value)
    # The original loader error is preserved for debugging...
    assert "libtvm.so" in message
    # ...and the message points at the real fix: a matching mlc-ai install.
    assert "mlc-ai" in message
    assert isinstance(exc_info.value.__cause__, OSError)


def test_load_lib_success(monkeypatch):
    """On success ``load_lib`` returns the loaded handle unchanged."""
    libinfo = _load_libinfo()
    sentinel = object()
    monkeypatch.setattr(libinfo.ctypes, "CDLL", lambda path: sentinel)

    assert libinfo.load_lib("/some/path/libmlc_llm.so") is sentinel
