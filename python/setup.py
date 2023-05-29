# pylint: disable=invalid-name, exec-used
"""Setup MLC LLM package."""
import os
import shutil
import sys

from setuptools import find_packages
from setuptools.dist import Distribution
from setuptools import setup


CURRENT_DIR = os.path.dirname(__file__)
CONDA_BUILD = os.getenv("CONDA_BUILD") is not None


def get_lib_path():
    """Get library path, name and version"""
    # Directly exec libinfo to get the right setup
    libinfo_py = os.path.join(CURRENT_DIR, "./mlc_chat/libinfo.py")
    libinfo = {"__file__": libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, "exec"), libinfo, libinfo)
    version = libinfo["__version__"]

    # conda installs libraries into env instead of packaging with pip
    if not CONDA_BUILD:
        libs = [
            libinfo["find_lib_path"]("mlc_llm")[0],
            libinfo["find_lib_path"]("mlc_llm_module")[0],
        ]
    else:
        libs = None

    return libs, version


def git_describe_version(original_version):
    """Get git describe version."""
    ver_py = os.path.join(CURRENT_DIR, "..", "version.py")
    libver = {"__file__": ver_py}
    exec(compile(open(ver_py, "rb").read(), ver_py, "exec"), libver, libver)
    _, gd_version = libver["git_describe_version"]()
    if gd_version is not None and gd_version != original_version:
        print("Use git describe based version %s" % gd_version)
    return gd_version


LIB_LIST, __version__ = get_lib_path()
__version__ = git_describe_version(__version__)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup_kwargs = {}
if not CONDA_BUILD:
    with open("MANIFEST.in", "w") as fo:
        for path in LIB_LIST:
            if os.path.isfile(path):
                shutil.copy(path, os.path.join(CURRENT_DIR, "mlc_chat"))
                _, libname = os.path.split(path)
                fo.write(f"include mlc_chat/{libname}\n")
    setup_kwargs = {"include_package_data": True}


setup(
    name="mlc_chat",
    version=__version__,
    description="MLC Chat: an universal runtime running LLMs",
    url="https://mlc.ai/mlc-llm/",
    author="MLC LLM Contributors",
    license="Apache 2.0",
    # See https://pypi.org/classifiers/
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
    keywords="machine learning",
    zip_safe=False,
    packages=find_packages(),
    package_dir={"mlc_chat": "mlc_chat"},
    distclass=BinaryDistribution,
    **setup_kwargs,
)


def _remove_path(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


if not CONDA_BUILD:
    # Wheel cleanup
    os.remove("MANIFEST.in")
    for path in LIB_LIST:
        _, libname = os.path.split(path)
        _remove_path(f"mlc_chat/{libname}")
