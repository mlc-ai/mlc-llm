🚧 Configure Quantization
=========================

Quantization Algorithm
----------------------

The default quantization algorithm used in MLC-LLM is grouping quantization method discussed in the papers `The case for 4-bit precision: k-bit Inference Scaling Laws <https://arxiv.org/abs/2212.09720>`__ and `LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models <https://arxiv.org/abs/2206.09557>`__.

.. _quantization_mode:

Quantization Mode
-----------------

In MLC-LLM we use a short code that indicates the quantization mode to use.    

The format of the code is ``qAfB(_id)``, where ``A`` represents the number
of bits for storing weights and ``B`` represents the number of bits for storing activations.
The ``_id`` is an integer identifier to distinguish different quantization algorithms (e.g. symmetric, non-symmetric, AWQ, etc).

Currently, available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and ``q4f16_awq`` (not stable).

More details to come.