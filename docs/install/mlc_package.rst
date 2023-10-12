.. _install-mlc-packages:

Install MLC-AI & MLC-Chat
=========================

.. .. contents:: Table of Contents
..     :local:
..     :depth: 2

We provide pip wheels for MLC-AI and MLC-Chat nightly build, which can be installed with pip. Select your operating system/compute platform and run the command in your terminal:

.. tabs::

    .. tab:: Linux

        .. tabs::

            .. tab:: CPU

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly mlc-chat-nightly -f https://mlc.ai/wheels

            .. tab:: CUDA 11.6

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly-cu116 mlc-chat-nightly-cu116 -f https://mlc.ai/wheels

            .. tab:: CUDA 11.7

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly-cu117 mlc-chat-nightly-cu117 -f https://mlc.ai/wheels

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly-cu118 mlc-chat-nightly-cu118 -f https://mlc.ai/wheels

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly-cu121 mlc-chat-nightly-cu121 -f https://mlc.ai/wheels
            
            .. tab:: CUDA 12.2

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly-cu122 mlc-chat-nightly-cu122 -f https://mlc.ai/wheels
            
            .. tab:: ROCm 5.6

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly-rocm mlc-chat-nightly-rocm -f https://mlc.ai/wheels
    
    .. tab:: Windows

        .. tabs::

            .. tab:: CPU

                .. code-block:: bash

                    pip install --pre --force-reinstall mlc-ai-nightly mlc-chat-nightly -f https://mlc.ai/wheels
            
            .. tab:: CUDA

                .. code-block:: bash

                    Windows package for CUDA is not available yet. Install from source if needed.
                
            .. tab:: ROCm

                .. code-block:: bash

                    Windows package for ROCm is not available yet. Install from source if needed.

    .. tab:: Mac

        .. tabs::

            .. tab:: CPU

                .. code-block:: bash

                pip install --pre --force-reinstall mlc-ai-nightly mlc-chat-nightly -f https://mlc.ai/wheels
                
            .. tab:: Metal

                .. code-block:: 

                    pip install --pre --force-reinstall mlc-ai-nightly mlc-chat-nightly -f https://mlc.ai/wheels


All Linux/Windows packages (both CPU/CUDA versions) supports Vulkan.

.. note::
    If you install the pip wheel under a Conda environment, please also install the latest gcc in Conda to resolve possible libstdc++.so issue:

        .. code-block::

            conda install -c conda-forge gcc


.. note:: 
    We provide conda packages for MLC-Chat-CLI nightly build, which can be installed with conda:

        .. code-block::

            conda create -n mlc-chat-venv -c mlc-ai -c conda-forge mlc-chat-nightly
            conda activate mlc-chat-venv