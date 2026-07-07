"""Library information. This is a standalone file that can be used to get various info"""

#! pylint: disable=protected-access
import ctypes
import os
import sys

__version__ = "0.1.dev0"
MLC_LIBRARY_PATH = os.environ.get("MLC_LIBRARY_PATH", None)


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


def load_lib(path):
    """Load a shared library, raising a clear error when a dependency is missing.

    ``ctypes.CDLL`` raises a bare ``OSError`` (e.g. ``libtvm.so: cannot open shared
    object file``) when the library itself is present but one of its shared
    dependencies cannot be resolved. ``libtvm.so`` is provided by the ``mlc-ai`` /
    TVM package rather than by ``mlc-llm`` itself, so this typically points at a
    missing or mismatched ``mlc-ai`` installation. Wrap the load to turn that opaque
    failure into an actionable message.

    Parameters
    ----------
    path : str
        The full path to the shared library to load.

    Returns
    -------
    lib : ctypes.CDLL
        The loaded library handle.
    """
    try:
        return ctypes.CDLL(path)
    except OSError as error:
        raise RuntimeError(
            f"Failed to load the MLC LLM library at '{path}'.\n"
            f"Underlying error: {error}\n"
            "This usually means one of its shared dependencies (for example "
            "libtvm.so, which is provided by the `mlc-ai` / TVM package) could not "
            "be found or is missing from the installation. Please make sure a "
            "matching `mlc-ai` package is installed, and when installing pip wheels "
            "ensure its CUDA variant matches (e.g. install `mlc-ai-nightly-cuXYZ` "
            "alongside `mlc-llm-nightly-cuXYZ`). See "
            "https://llm.mlc.ai/docs/install/mlc_llm.html for details."
        ) from error
