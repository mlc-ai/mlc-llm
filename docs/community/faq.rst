.. _FAQ:

Frequently Asked Questions
==========================

This is a list of Frequently Asked Questions (FAQ) about the MLC-LLM. Feel free to suggest new entries!

... How can I customize the temperature, and repetition penalty of models?
   Please check our :ref:`configure-mlc-chat-json` tutorial.

... What's the quantization algorithm MLC-LLM using?
   Please check our :doc:`/compilation/configure_quantization` tutorial.

... Why do I encounter an error ``free(): invalid pointer, Aborted (core dumped)`` at the end of model compilation?
   This happens if you compiled TVM-Unity from source and didn't hide LLVM symbols in cmake configurations.
   Please follow our instructions in :ref:`Building TVM Unity from Source  <tvm-unity-build-from-source>` tutorial to compile TVM-Unity which hides LLVM symbols, or use our pre-built MLC-LLM :doc:`pip wheels <../install/mlc_llm>`.
