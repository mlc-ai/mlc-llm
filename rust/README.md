# MLC-LLM Rust Package

This folder contains the source code of MLC-LLM Rust package.

# Installations
To set up the MLC-LLM Rust package, please follow these steps:

**Step 1:** Begin by following the detailed installation [instructions](https://llm.mlc.ai/docs/deploy/rest.html#optional-build-from-source) for TVM Unity and MLC-LLM.

**Step 2:** Define the environment variables for TVM and MLC-LLM by running the following commands in your terminal:
```bash
export TVM_HOME=/path/to/tvm
export MLC_HOME=/path/to/mlc-llm
```

**Step 3:** Update your `LD_LIBRARY_PATH` to include the `libtvm_runtime` and `libmlc_llm_module` libraries. These can typically be found within the build directories of your TVM and MLC-LLM installations.

# How to run it?
To start using the package, you can refer to the example code provided in the examples directory. This code demonstrates how to create a chat_module and serve prompts effectively.

Execute the example with Cargo using the following command:
```bash
cargo run --example mlc_chat
```

