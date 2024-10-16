Configure Quantization
======================

Quantization Algorithm
----------------------

The default quantization algorithm used in MLC-LLM is grouping quantization method discussed in the papers `The case for 4-bit precision: k-bit Inference Scaling Laws <https://arxiv.org/abs/2212.09720>`__ and `LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models <https://arxiv.org/abs/2206.09557>`__.

.. _quantization_mode:

Quantization Mode
-----------------

In MLC-LLM we use a short code that indicates the quantization mode to use. MLC-LLM supports both
weight-only quantization and weight-activation quantization.

For the weight-only quantization, he format of the code is ``qAfB(_id)``, where ``A`` represents the number
of bits for storing weights and ``B`` represents the number of bits for storing activations.
The ``_id`` is an integer identifier to distinguish different quantization algorithms (e.g. symmetric, non-symmetric, AWQ, etc).

Currently, available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and ``q4f16_awq`` (not stable).

For the weight-activation quantization, currently MLC-LLM supports FP8 quantization on CUDA.
The available options are: ``e4m3_e4m3_f16`` and ``e5m2_e5m2_f16``. In these modes, both weights and activations are quantized to FP8 format.
The output of each layer is in higher precision (FP16) and then requantized to FP8.

.. _calibration:

Calibration
-----------

For ``e4m3_e4m3_f16`` quantization, we need to calibrate the quantization parameters for the activations.
The calibration process is done by running the following command:

1. Compile the calibration model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the same compilation workflow to compile the model in calibration mode.
The only difference is that we need to specify the quantization mode as ``e4m3_e4m3_f16_calibrate``.

.. code-block:: bash

    mlc_llm gen_config \
        <model-path> \
        --quantization e4m3_e4m3_f16_max_calibrate \
        --output <output-path>

    mlc_llm convert_weights \
        <model-path> \
        --quantization e4m3_e4m3_f16_max_calibrate \
        --output <output-path>

    mlc_llm compile \
        <config-path> \
        --output <output-path>

2. Run the calibration model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will run the calibration model on the dataset such as ShareGPT to collect the statistics of the
activations. The calibration model will updates the quantization parameters in the weights file
in-place. We turn off the cuda graph as it is not yet supported in the calibration process.

.. code-block:: bash

   mlc_llm calibrate \
       <model-path> \
       --model-lib <model-lib-path> \
       --dataset <dataset-path> \
       --num-calibration-samples <num-samples> \
       --opt "cudagraph=0"
       --output <output-path>

3. Compile the quantized model for inference.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the calibration process, we can compile the model for inference. In this step, we only need
to generate the configuration file using the desired quantization format and compile the model.
Weights are already quantized and calibrated in the previous steps and do not need to be converted again.

.. code-block:: bash

    mlc_llm gen_config \
        <model-path> \
        --quantization e4m3_e4m3_f16 \
        --output <output-path>
    mlc_llm compile \
        <config-path> \
        --output <output-path>
