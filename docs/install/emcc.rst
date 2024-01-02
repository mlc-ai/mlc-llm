.. _install-web-build:

Install Wasm Build Environment
==============================

This page describes the steps to setup build environment for WebAssembly and WebGPU builds.

Step 1: Install EMSDK
---------------------

Emscripten is an LLVM-based compiler that compiles C/C++ source code to WebAssembly.
We need to install emscripten for webgpu build.

- Please follow the installation instruction `here <https://emscripten.org/docs/getting_started/downloads.html#installation-instructions-using-the-emsdk-recommended>`__
  to install the latest emsdk.
- Source path/to/emsdk_env.sh so emcc is reachable from PATH and the command emcc works.

Validate that emcc is accessible in shell

.. code:: bash

    emcc --version

Step 2: Set TVM_HOME
--------------------

We need to set a path to a tvm source in order to build tvm runtime.
Note that you do not need to build tvm unity from the source. The source here is only used to build the web runtime component.
Set environment variable in your shell startup profile in to point to ``3rdparty/tvm``

.. code:: bash

    export TVM_HOME=/path/to/3rdparty/tvm


Step 3: Prepare Wasm Runtime
----------------------------

First, we need to obtain a copy of the mlc-llm source code for the setup script

.. code:: bash
    
    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm

Now we can prepare wasm runtime using the script in mlc-llm repo

.. code:: bash
    
    ./scripts/prep_emcc_deps.sh

We can then validate the outcome

.. code:: bash

    >>> echo ${TVM_HOME}

    /path/set/in/step2

    >>> ls -l ${TVM_HOME}/web/dist/wasm/*.bc

    tvmjs_support.bc
    wasm_runtime.bc
    webgpu_runtime.bc
