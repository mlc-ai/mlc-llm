from distutils.core import setup

from setuptools import find_packages

# Note there is no need to setup when
# running locally in this folder.

setup(
    name="mlc-llm",
    version="0.1.0",
    license="Apache-2.0",
    description="LLM universal compilation",
    author="MLC LLM contributors",
    url="https://github.com/mlc-ai/mlc-llm",
    keywords=[],
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
