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

.. note::
    We recently found that using the latest ``emcc`` version may run into issues during runtime. Use
    ``./emsdk install 3.1.56`` instead of ``./emsdk install latest`` for now as a workaround.

    The error may look like

    .. code:: text

        Init error, LinkError: WebAssembly.instantiate(): Import #6 module="wasi_snapshot_preview1"
        function="proc_exit": function import requires a callable


Step 2: Set TVM_SOURCE_DIR and MLC_LLM_SOURCE_DIR
-------------------------------------------------

We need to set a path to a tvm source in order to build tvm runtime.
Note that you do not need to build tvm unity from the source. The source here is only used to build the web runtime component.
Set environment variable in your shell startup profile in to point to ``3rdparty/tvm`` (if preferred, you could also
point to your own TVM address if you installed TVM from source).

Besides, we also need to set ``MLC_LLM_SOURCE_DIR`` so that we can locate ``mlc_wasm_runtime.bc`` when compiling a model library wasm.

.. code:: bash

    export TVM_SOURCE_DIR=/path/to/3rdparty/tvm
    export MLC_LLM_SOURCE_DIR=/path/to/mlc-llm


Step 3: Prepare Wasm Runtime
----------------------------

First, we need to obtain a copy of the mlc-llm source code for the setup script

.. code:: bash

    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm

Now we can prepare wasm runtime using the script in mlc-llm repo

.. code:: bash

    ./web/prep_emcc_deps.sh

We can then validate the outcome

.. code:: bash

    >>> echo ${TVM_SOURCE_DIR}

    /path/set/in/step2

    >>> ls -l ${TVM_SOURCE_DIR}/web/dist/wasm/*.bc

    tvmjs_support.bc
    wasm_runtime.bc
    webgpu_runtime.bc
