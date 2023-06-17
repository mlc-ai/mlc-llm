.. _FAQ:

Frequently Asked Questions (FAQ)
================================

This is a list of Frequently Asked Questions (FAQ) about the MLC-LLM. Feel free to suggest new entries!

... How can I customize the temperature, repetition penalty of models?
   Please check our :doc:`/get_started/mlc_chat_config` tutorial.

... What's the quantization algorithm MLC-LLM using?
   Please check our :doc:`/tutorials/quantization` tutorial.

... Why do I encounter an error ``free(): invalid pointer, Aborted (core dumped)`` at the end of model compilation?
   This happens if you compiled TVM-Unity from source and didn't hide LLVM symbols in cmake configurations.
   Please follow our instructions in :ref:`Building TVM Unity from Source  <tvm-unity-build-from-source>` tutorial to compile TVM-Unity which hides LLVM symbols,
   or use our pre-builc MLC-AI pip wheels from `MLC Packages <https://mlc.ai/package/>`__.
