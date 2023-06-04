.. _FAQ:

Frequently Asked Questions (FAQ)
================================

This is a list of Frequently Asked Questions (FAQ) about the MLC-LLM. Feel free to suggest new entries!

... How can I customize the temperature, repetition penalty of models?
   There is a ``mlc-chat-config.json`` file under your model directory, where you can modify the parameters.

... What's the quantization algorithm MLC-LLM using?
   MLC-LLM does not impose any restrictions on the choice of the quantization algorithm, and the design should be compatible with all quantization algorithms. By default, we utilize the grouping quantization method discussed in the paper `The case for 4-bit precision: k-bit Inference Scaling Laws <https://arxiv.org/abs/2212.09720>`__.

... Why do I encounter an error ``free(): invalid pointer, Aborted (core dumped)`` at the end of model compilation?
   This happens if you compiled TVM-Unity from source and didn't hide LLVM symbols in cmake configurations.
   Please follow our instructions in :ref:`tvm-unity-build-from-source` to compile TVM-Unity which hides LLVM symbols,
   or use our pre-builc MLC-AI pip wheels from `MLC Packages <https://mlc.ai/package/>`__.
