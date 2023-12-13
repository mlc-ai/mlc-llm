(DEPRECATED) Python API for Model Compilation
=============================================

.. contents:: Table of Contents
   :local:
   :depth: 2

We expose Python API for compiling/building models in the package :py:mod:`mlc_llm`, so
that users may build a model in any directory in their program (i.e. not just
within the mlc-llm repo).

Install MLC-LLM as a Package
----------------------------

To install, we first clone the repository (as mentioned in
:ref:`compile-models-via-MLC`):

.. code:: bash

    # clone the repository
    git clone git@github.com:mlc-ai/mlc-llm.git --recursive

Afterwards, we use ``pip`` to install :py:mod:`mlc_llm` as a package so that we can
use it in any directory:

.. code:: bash

    # enter to root directory of the repo
    cd mlc-llm
    # install the package
    pip install -e .

To verify installation, you are expected to see information about the package
:py:mod:`mlc_llm` even if you are not in the directory of ``mlc-llm``:

.. code:: bash

    python -c "import mlc_llm; print(mlc_llm)"

Compiling Model in Python
-------------------------

After installing the package, you can build the model using :meth:`mlc_llm.build_model`,
which takes in an instance of :class:`BuildArgs` (a dataclass that represents
the arguments for building a model).

For detailed instructions with code, please refer to `the Python notebook
<https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_compile_llama2_with_mlc_llm.ipynb>`_
(executable in Colab), where we walk you through compiling Llama-2 with :py:mod:`mlc_llm`
in Python.

.. _api-reference-compile-model:

API Reference
-------------

In order to use the python API :meth:`mlc_llm.build_model`, users need to create
an instance of the dataclass :class:`BuildArgs`. The corresponding arguments in
the command line shown in :ref:`compile-command-specification` are automatically
converted from the definition of :class:`BuildArgs` and are equivalent.

Then with an instantiated :class:`BuildArgs`, users can call the build API
:meth:`mlc_llm.build_model`.

.. currentmodule:: mlc_llm

.. autoclass:: BuildArgs
    :members:

.. autofunction:: build_model
