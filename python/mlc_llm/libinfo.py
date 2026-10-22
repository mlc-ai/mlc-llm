"""Library information. This is a standalone file that can be used to get various info"""

#! pylint: disable=protected-access
import importlib.util
import os
import platform
import subprocess
import sys

__version__ = "0.1.dev0"
MLC_LIBRARY_PATH = os.environ.get("MLC_LIBRARY_PATH", None)

# The packages that ship the native libraries which are loaded together with
# the MLC LLM runtime. They are imported lazily, so that the locations of the
# libraries can be looked up without importing (and therefore loading) them.
NATIVE_LIBRARY_PACKAGES = ("tvm", "tvm_ffi")


def get_env_paths(env_var, splitter):
    """Get path in env variable"""
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(splitter)]
    return []


def get_dll_directories():
    """Get extra mlc llm dll directories"""
    curr_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.abspath(os.path.join(curr_dir, "..", ".."))
    dll_path = [
        curr_dir,
        os.path.join(source_dir, "build"),
        os.path.join(source_dir, "build", "Release"),
    ]
    if MLC_LIBRARY_PATH:
        dll_path.append(MLC_LIBRARY_PATH)
    if "CONDA_PREFIX" in os.environ:
        dll_path.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(get_env_paths("LD_LIBRARY_PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(get_env_paths("DYLD_LIBRARY_PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(get_env_paths("PATH", ";"))
    return [os.path.abspath(p) for p in dll_path if os.path.isdir(p)]


def find_lib_path(name, optional=False):
    """Find mlc llm library

    Parameters
    ----------
    name : str
        The name of the library

    optional: boolean
        Whether the library is required
    """
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_name = f"lib{name}.so"
    elif sys.platform.startswith("win32"):
        lib_name = f"{name}.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = f"lib{name}.dylib"
    else:
        lib_name = f"lib{name}.so"

    dll_paths = get_dll_directories()
    lib_dll_path = [os.path.join(p, lib_name) for p in dll_paths]
    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_found:
        if not optional:
            message = (
                f"Cannot find libraries: {lib_name}\n"
                + "List of candidates:\n"
                + "\n".join(lib_dll_path)
            )
            raise RuntimeError(message)
    return lib_found


def find_native_libraries() -> list[str]:
    """Find the native libraries that are loaded together with MLC LLM.

    The MLC LLM runtime maps the libraries shipped inside the ``mlc_llm``
    package as well as the TVM runtime libraries shipped by the ``tvm`` and
    ``tvm_ffi`` packages into the process. All of them are loaded, so all of
    them have to be loadable.

    Returns
    -------
    libraries : list of str
        The paths of the native libraries.
    """
    package_dir = os.path.dirname(os.path.realpath(__file__))
    directories = [package_dir, os.path.join(package_dir, "lib")]
    for package in NATIVE_LIBRARY_PACKAGES:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            directories.append(os.path.join(location, "lib"))
    libraries = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(".dylib"):
                libraries.append(os.path.join(directory, name))
    return libraries


def find_invalid_code_signatures(libraries: list[str]) -> list[str]:
    """Find the libraries whose macOS code signature is not valid.

    macOS refuses to map a library with a broken code signature and kills the
    whole process while doing so, which turns a broken installation into a
    silent crash. ``codesign`` reports the very same problem without mapping
    the library, so the check can run before the libraries are loaded.

    Parameters
    ----------
    libraries : list of str
        The paths of the native libraries to check.

    Returns
    -------
    invalid : list of str
        The paths of the libraries that cannot be mapped into a process.
    """
    if sys.platform != "darwin":
        return []
    if not libraries:
        return []
    try:
        result = subprocess.run(
            ["codesign", "--verify", *libraries],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        # `codesign` is not available, so there is nothing to check.
        return []
    if result.returncode == 0:
        return []
    # `codesign` stops at the first broken library, so check them one by one
    # to report all of them.
    invalid = []
    for library in libraries:
        try:
            result = subprocess.run(
                ["codesign", "--verify", library],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return []
        if result.returncode != 0:
            invalid.append(library)
    return invalid


def check_native_libraries() -> None:
    """Check that the native libraries of MLC LLM can be loaded.

    On macOS, a library with a broken code signature is not mapped into the
    process. The kernel kills the process instead, so the interpreter stops
    without any message and without a traceback. Checking the signatures
    before the libraries are loaded turns that silent crash into an error
    that can be acted upon.

    Raises
    ------
    ImportError
        When one of the native libraries has a broken code signature.
    """
    if sys.platform != "darwin" or platform.machine() not in ("arm64", "aarch64"):
        return
    if os.environ.get("SKIP_CODE_SIGNATURE_CHECK", "0") == "1":
        return
    invalid = find_invalid_code_signatures(find_native_libraries())
    if not invalid:
        return
    libraries = "\n".join(f"  {library}" for library in invalid)
    commands = "\n".join(f"  codesign --force --sign - {library}" for library in invalid)
    raise ImportError(
        "The native libraries of MLC LLM have an invalid macOS code signature:\n"
        f"{libraries}\n"
        "macOS does not map a library with a broken code signature into the "
        "process, it kills the process instead. That is why importing mlc_llm "
        "ends the interpreter without any message.\n"
        "Re-sign the libraries to make them loadable:\n"
        f"{commands}"
    )
