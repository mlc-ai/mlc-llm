Install Conda
=============

MLC LLM does not depend on, but generally recommends conda as a generic dependency manager, primarily because it creates unified cross-platform experience to make windows/Linux/macOS development equally easy. Moreover, conda is python-friendly and provides all the python packages needed for MLC LLM, such as numpy.

.. contents:: Table of Contents
    :depth: 2


Install Miniconda
-----------------

**Use installer.** Miniconda, a minimal distribution of conda, comes with out-of-box installer across Windows/macOS/Linux. Please refer to its `official website <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_ link for detailed instructions.

**Set libmamba as the dependency solver.** The default dependency solver in conda could be slow in certain scenarios, and it is always recommended to upgrade it to libmamba, a faster solver.

.. code-block:: bash
   :caption: Set libmamba as the default solver

   # update conda
   conda update --yes -n base -c defaults conda
   # install `conda-libmamba-solver`
   conda install --yes -n base conda-libmamba-solver
   # set it as the default solver
   conda config --set solver libmamba

.. note::
    Conda is a generic dependency manager, which is not necessarily related to any Python distributions.
    In fact, some of our tutorials recommends to use conda to install cmake, git and rust for its unified experience across OS platforms.


Validate installation
---------------------

**Step 1. Check conda-arch mismatch.** Nowadays macOS runs on two different architectures: arm64 and x86_64, which could particularly lead to many misuses in MLC LLM, where the error message hints about "architecture mismatch". Use the following command to make sure particular conda architecture is installed accordingly:

.. code-block:: bash
   :caption: Check conda architecture

   >>> conda info | grep platform
   # for arm mac
   platform : osx-arm64
   # for x86 mac
   platform : osx-64

**Step 2. Check conda virtual environment.** If you have installed python in your conda virtual environment, make sure conda, Python and pip are all from this environment:

.. code-block:: bash
   :caption: Check conda virtual environment (macOS, Linux)

   >>> echo $CONDA_PREFIX
   /.../miniconda3/envs/mlc-doc-venv
   >>> which python
   /.../miniconda3/envs/mlc-doc-venv/bin/python
   >>> which pip
   /.../miniconda3/envs/mlc-doc-venv/bin/pip

.. code-block:: bat
   :caption: Check conda virtual environment (Windows)

   >>> echo $Env:CONDA_PREFIX
   \...\miniconda3\envs\mlc-doc-venv
   >>> Get-Command python.exe
   \...\miniconda3\envs\mlc-doc-venv\bin\python.exe
   >>> Get-Command pip.exe
   \...\miniconda3\envs\mlc-doc-venv\bin\pip.exe
