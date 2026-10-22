"""The native libraries of MLC LLM have to be loadable by the process.

On macOS, a library with a broken code signature is not mapped into the
process. The kernel kills the process instead, so the interpreter stops
without any message. The check in ``mlc_llm.libinfo`` has to turn that into an
error before the libraries are loaded.
"""

import platform
import shutil
import subprocess
import sys

import pytest

from mlc_llm import libinfo

pytestmark = [
    pytest.mark.unittest,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() not in ("arm64", "aarch64"),
        reason="macOS on Apple Silicon only",
    ),
]


def _signed_copy(tmp_path, name):
    """A copy of a signed Mach-O file, signed with an ad-hoc signature."""
    target = tmp_path / name
    shutil.copy(sys.executable, target)
    subprocess.run(
        ["codesign", "--force", "--sign", "-", str(target)],
        check=True,
        capture_output=True,
    )
    return target


def _broken_copy(tmp_path, name):
    """A copy of a signed Mach-O file whose signature does not match anymore."""
    target = tmp_path / name
    shutil.copy(sys.executable, target)
    content = bytearray(target.read_bytes())
    content[len(content) // 2] ^= 0xFF
    target.write_bytes(bytes(content))
    return target


def test_signed_library_is_accepted(tmp_path):
    """A library with a valid code signature can be mapped into the process."""
    library = _signed_copy(tmp_path, "libvalid.dylib")
    assert libinfo.find_invalid_code_signatures([str(library)]) == []


def test_broken_library_is_reported(tmp_path):
    """A library with a broken code signature is reported as unloadable."""
    valid = _signed_copy(tmp_path, "libvalid.dylib")
    broken = _broken_copy(tmp_path, "libbroken.dylib")
    assert libinfo.find_invalid_code_signatures([str(broken)]) == [str(broken)]
    assert libinfo.find_invalid_code_signatures([str(valid), str(broken)]) == [str(broken)]


def test_check_native_libraries_accepts_signed_libraries(tmp_path, monkeypatch):
    """The check passes when all native libraries are signed properly."""
    library = _signed_copy(tmp_path, "libvalid.dylib")
    monkeypatch.setattr(libinfo, "find_native_libraries", lambda: [str(library)])
    libinfo.check_native_libraries()


def test_check_native_libraries_reports_broken_libraries(tmp_path, monkeypatch):
    """The check reports the libraries that cannot be mapped into the process."""
    broken = _broken_copy(tmp_path, "libbroken.dylib")
    monkeypatch.setattr(libinfo, "find_native_libraries", lambda: [str(broken)])
    with pytest.raises(ImportError) as error:
        libinfo.check_native_libraries()
    message = str(error.value)
    assert str(broken) in message
    assert "codesign --force --sign -" in message


def test_find_native_libraries_returns_existing_files():
    """The libraries that are reported exist on disk."""
    libraries = libinfo.find_native_libraries()
    assert isinstance(libraries, list)
    for library in libraries:
        assert library.endswith(".dylib")
        assert shutil.os.path.isfile(library)
