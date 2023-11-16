from distutils.core import setup
from setuptools.dist import Distribution
from setuptools import find_packages
import os

# Note there is no need to setup when
# running locally.

CURRENT_DIR = os.path.dirname(__file__)


def git_describe_version(original_version):
    """Get git describe version."""
    ver_py = os.path.join(CURRENT_DIR, "version.py")
    libver = {"__file__": ver_py}
    exec(compile(open(ver_py, "rb").read(), ver_py, "exec"), libver, libver)
    _, gd_version = libver["git_describe_version"]()
    if gd_version is not None and gd_version != original_version:
        print("Use git describe based version %s" % gd_version)
    return gd_version


__version__ = git_describe_version(None)

setup(
    name="mlc_llm",
    version=__version__,
    description="MLC LLM: Universal Compilation of Large Language Models",
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
    package_dir={"mlc_llm": "mlc_llm"},
    install_requires=["numpy", "torch", "transformers", "scipy", "timm"],
    entry_points={"console_scripts": ["mlc_llm_build = mlc_llm.build:main"]},
    distclass=Distribution,
)
