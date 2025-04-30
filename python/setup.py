# pylint: disable=invalid-name, exec-used
"""Setup MLC LLM package."""
import os
import shutil

from setuptools import find_packages, setup
from setuptools.dist import Distribution

CURRENT_DIR = os.path.dirname(__file__)
CONDA_BUILD = os.getenv("CONDA_BUILD") is not None


def get_lib_path():
    """Get library path, name and version"""
    # Directly exec libinfo to get the right setup
    libinfo_py = os.path.join(CURRENT_DIR, "./mlc_llm/libinfo.py")
    libinfo = {"__file__": libinfo_py}
    with open(libinfo_py, "rb") as f:
        exec(compile(f.read(), libinfo_py, "exec"), libinfo, libinfo)
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
    with open(ver_py, "rb") as f:
        exec(compile(f.read(), ver_py, "exec"), libver, libver)
    _, gd_version = libver["git_describe_version"]()
    if gd_version is not None and gd_version != original_version:
        print(f"Use git describe based version {gd_version}")
    if gd_version is None:
        print(f"Use original version {original_version}")
        return original_version
    return gd_version


LIB_LIST, __version__ = get_lib_path()
__version__ = git_describe_version(__version__)


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        """Return True for binary distribution."""
        return True

    def is_pure(self):
        """Return False for binary distribution."""
        return False


def main():
    """The main entrypoint."""
    setup_kwargs = {}
    if not CONDA_BUILD:
        with open("MANIFEST.in", "w", encoding="utf-8") as fo:
            for path in LIB_LIST:
                if os.path.isfile(path):
                    shutil.copy(path, os.path.join(CURRENT_DIR, "mlc_llm"))
                    _, libname = os.path.split(path)
                    fo.write(f"include mlc_llm/{libname}\n")
        setup_kwargs = {"include_package_data": True}

    setup(
        name="mlc_llm",
        version=__version__,
        description="MLC LLM: an universal LLM deployment engine via ML compilation.",
        url="https://llm.mlc.ai/",
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
        entry_points={
            "console_scripts": ["mlc_llm = mlc_llm.__main__:main"],
        },
        package_dir={"mlc_llm": "mlc_llm"},
        install_requires=[
            "fastapi",
            "uvicorn",
            "shortuuid",
            "torch",
            "safetensors",
            "requests",
            "tqdm",
            "sentencepiece",
            "tiktoken",
            "prompt_toolkit",
            "openai",
            "transformers",
            "pandas",
            "datasets",
            "ml_dtypes>=0.5.1",
            "flashinfer-python==0.2.5",
        ],
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
            _remove_path(f"mlc_llm/{libname}")


main()
