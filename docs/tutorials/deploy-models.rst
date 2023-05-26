.. _How to Deploy Models:

How to Deploy Models
====================

In this tutorial, we will guide you on how to **deploy** models built by MLC LLM to different backends and devices. We support four options of device categories for deployment: laptop/desktop, web browser, iPhone/iPad, and Android phones.
.. Before starting this tutorial, you are expected to have completed model build in :ref:`How to Compile Models`, and this page will not cover the model build part. After finishing this page, you will be able to run large language models on the device you want.

This page contains the following sections. We first introduce how to prepare the (pre)built model libraries and weights, and one section for each backend device will then follow. Every section contains a Frequently Asked Questions (FAQ) subsection for troubleshooting.

.. contents:: Table of Contents
    :depth: 1
    :local:


.. _prepare-weight-library:

Prepare model weight and library
--------------------------------

In order to load the model correctly in deployment, we need to put the model libraries and weights to the right location before deployment.
You can select from the panel below for the preparation steps you will need.

.. tabs::

    .. tab:: Desktop/laptop

        .. tabs::

            .. tab:: Use prebuilt model

                MLC LLM provides a list of prebuilt models (check our :doc:`/model-zoo` for the list).
                If you want to use them, run the commands below under the root directory of MLC LLM to download the libraries and weights to the target location.

                .. code:: shell

                    # Make sure you have installed git-lfs.
                    mkdir -p dist/prebuilt
                    cd dist/prebuilt
                    git lfs install
                    git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git lib

                    # Choose one or both commands from below according to the model(s) you want to deploy.
                    git clone https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0
                    # and/or
                    git clone https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0

            .. tab:: Use model you built

                You are all good. There is no further action to prepare the model libraries and weights.

    .. tab:: Web browser

        .. tabs::

            .. tab:: Use prebuilt model

                .. .. tabs::

                ..     .. tab:: Local deployment

                ..         Please clone Web LLM

                TBA.

            .. tab:: Use model you built

                TBA.

    .. tab:: iPhone/iPad/Android phones

        .. tabs::

            .. tab:: Use prebuilt model

                You are all good. There is no further action to prepare the model libraries and weights.

            .. tab:: Use model you built

                TBA.


.. _deploy-on-laptop-desktop:

Deploy Models on Your Laptop/Desktop
------------------------------------

This section goes through the process of deploying prebuilt model or the model you built on your laptop or desktop.
MLC LLM provides a Command-Line Interface (CLI) application to deploy and help interact with the model.
After :ref:`preparing the CLI <CLI-prepare-CLI>`, you can :ref:`deploy and interact <CLI-run-model>` with the model on your machine through CLI.

.. _CLI-prepare-CLI:

Prepare the CLI
~~~~~~~~~~~~~~~

We have released the `prebuilt CLI Conda package <https://anaconda.org/mlc-ai/mlc-chat-nightly>`_, which you can directly :ref:`install via Conda commands <CLI-install-from-Conda>`. You can also :ref:`build CLI from source <CLI-build-from-source>`.


.. _CLI-install-from-Conda:

Option 1: Install from Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. _CLI-build-from-source:

Option 2: Build from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are a MLC-LLM developer and you add some functionalities to the CLI, you can build the CLI from source by running the following command:

.. code:: shell

    mkdir -p build
    python3 cmake/gen_cmake_config.py
    cp cmake/config.cmake build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    ldconfig  # Refresh shared library cache
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
        ldconfig # Re
    

.. _CLI_validate-installation:

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

You can validate the CLI build by executing the command:

.. code:: bash

   mlc_chat_cli --help

You are expected to see the help documentation of ``mlc_chat_cli``,
which means the installation is successful.

.. _CLI-run-model:

Run the Models Through CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the model, we need to know the model's "id" which can be recognized by the CLI.
Model id is in the format ``MODEL_NAME-QUANTIZATION_MODE`` (for example, ``vicuna-v1-7b-q3f16_0``, ``RedPajama-INCITE-Chat-3B-v1-q4f16_0``, etc.).
You can find the model id by checking the directory names under ``dist`` (if you built model on your own) or ``dist/prebuilt`` if you use prebuilt models:

.. code:: shell

    # Check id for models manually built.
    ~/mlc-llm > ls dist
    RedPajama-INCITE-Chat-3B-v1-q4f16_0     models                vicuna-v1-7b-q3f16_0
    RedPajama-INCITE-Chat-3B-v1-q4f32_0     prebuilt              vicuna-v1-7b-q4f32_0

    # Check id for prebuilt models.
    # Note: Model ids start with the model name after `mlc-chat-`.
    ~ > ls dist/prebuilt
    mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0    mlc-chat-vicuna-v1-7b-q3f16_0

After confirming the model id, we can run the model in CLI by

.. code:: shell

    # If CLI is installed from Conda:
    mlc_chat_cli --local-id MODEL_ID
    # example:
    mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_0

    # If CLI is built from source:
    ./build/mlc_chat_cli --local-id MODEL_ID
    # example:
    ./build/mlc_chat_cli --local-id vicuna-v1-7b-q3f16_0


Troubleshooting FAQ
~~~~~~~~~~~~~~~~~~~

TBA.


Deploy Models on Your Web Browser
---------------------------------

TBA.

Troubleshooting FAQ
~~~~~~~~~~~~~~~~~~~

TBA.


Deploy Models on Your iPhone/iPad
---------------------------------

This section introduces how to deploy model you built or prebuilt by us on your iPhone/iPad devices.
The iOS/iPadOS application supports chatting with prebuilt Vicuna or RedPajama models, and also supports using the model you manually built.

MLC LLM has released an iOS/iPadOS application which you can directly download and use.
You can also build the application on your own.


Troubleshooting FAQ
~~~~~~~~~~~~~~~~~~~


Deploy Model on Your Android Phone
----------------------------------

Troubleshooting FAQ
~~~~~~~~~~~~~~~~~~~

