🚧 Use Compiled LLMs in C++ CLI
===============================

.. contents:: Table of Contents
    :depth: 2


MLC-LLM CLI is a command-line interface for MLC-LLM, which enables user to chat with the bot in terminal. Please refer to :ref:`prepare-weight-library` for installation instructions.
We have released the `prebuilt CLI Conda package <https://anaconda.org/mlc-ai/mlc-chat-nightly>`_, which you can directly :ref:`install via Conda commands <CLI-install-from-Conda>`.
You can also :ref:`build CLI from source <CLI-build-from-source>`.


Option 1: Install from Conda
----------------------------

The easiest way to install the CLI from Conda, we can follow the instructions below to create a Conda environment and then install.

.. note::
    The prebuilt CLI **does not** support CUDA. Please :ref:`build CLI from source <CLI-build-from-source>` if you want to deploy models to CUDA backend.

.. code:: shell

    # Create a new conda environment and activate the environment.
    conda create -n mlc-chat
    conda activate mlc-chat
    # Install the chat CLI app from Conda.
    conda install -c mlc-ai -c conda-forge mlc-chat-nightly --force-reinstall

.. note::
    After installation, you can run ``mlc_chat_cli --help`` to verify that the CLI is installed correctly.

Option 2: Build from source
---------------------------

If you are a MLC-LLM developer and you add some functionalities to the CLI, you can build the CLI from source by running the following command:

.. code:: shell

    # create build directory
    mkdir -p build
    # prepare dependencies
    bash scripts/prep_deps.sh
    source "$HOME/.cargo/env"
    # generation cmake config
    python3 cmake/gen_cmake_config.py
    cp config.cmake build
    # build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    # Refresh shared library cache
    ldconfig  
    cd -

.. note::
    The ``make`` commands above is expected to end with ``[100%] Built target mlc_chat_cli`` on Linux and macOS.

    In the case that user do not have sudo privilege, user can customize the install prefix by adding ``-DCMAKE_INSTALL_PREFIX=/path/to/install`` to the ``cmake`` command. For example, if you want to install MLC-LLM CLI to ``~/.local``, you can run the following command:

    .. code-block:: bash
    
        export LOCAL_PATH=~/.local
        cmake .. -DCMAKE_INSTALL_PREFIX=$LOCAL_PATH

    Please also remember to add ``$LOCAL_PATH/bin`` to your ``$PATH`` environment variable and ``$LOCAL_PATH/lib`` to your ``$LD_LIBRARY_PATH`` environment variable:

    .. code-block:: bash
        
        export PATH=$LOCAL_PATH/bin:$PATH
        export LD_LIBRARY_PATH=$LOCAL_PATH/lib:$LD_LIBRARY_PATH
        ldconfig # Refresh shared library cache

Validate Installation
---------------------

You can validate the CLI build by executing the command:

.. code:: bash

   mlc_chat_cli --help

You are expected to see the help documentation of ``mlc_chat_cli``,
which means the installation is successful.
