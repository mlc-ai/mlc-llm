# QAT Training for MLC-LLM

This directory contains scripts and utilities for Quantization Aware Training (QAT) that are compatible with MLC-LLM's q4f16_1 format.

## Overview

- **Base Model**: Llama3.2-1B (after SFT training)
- **Training Data**: ShareGPT format, distributed across multiple files
- **Target Quantization**: q4f16_1 format for MLC-LLM
- **Output**: QAT-trained model ready for MLC-LLM conversion

## Directory Structure

```
qat_training/
├── README.md                    # This file
├── config/
│   ├── training_config.py       # Training configuration
│   └── model_config.py         # Model-specific settings
├── data/
│   ├── data_loader.py          # ShareGPT data loading utilities
│   ├── data_processor.py       # Data preprocessing and sampling
│   └── data_sampler.py         # Smart sampling from large datasets
├── training/
│   ├── qat_trainer.py          # Main QAT training script
│   ├── qat_model.py            # QAT model wrapper
│   └── metrics_logger.py       # Training progress and metrics logging
├── conversion/
│   ├── weight_converter.py     # Convert QAT weights to q4f16_1 format
│   └── mlc_formatter.py        # Format weights for MLC-LLM
├── scripts/
│   ├── train_qat.py            # Main training entry point
│   ├── convert_to_mlc.py       # Conversion script
│   └── validate_model.py       # Model validation
└── examples/
    ├── sample_config.yaml       # Example configuration
    └── run_training.sh          # Example training script
```

## Quick Start

1. **Prepare your training data**:
   ```bash
   python scripts/prepare_data.py --input_dir /path/to/sharegpt/files --output_dir ./data/processed --sample_count 30000
   ```

2. **Start QAT training**:
   ```bash
   python scripts/train_qat.py --config examples/sample_config.yaml --model_path /path/to/your/sft/model
   ```

3. **Convert to MLC format**:
   ```bash
   python scripts/convert_to_mlc.py --qat_model ./outputs/qat_trained --output_dir ./outputs/mlc_ready
   ```

4. **Use with MLC-LLM**:
   ```bash
   mlc_llm convert_weight ./outputs/mlc_ready --quantization q4f16_1 --output ./final_model
   ```

## Features

- **Multi-file ShareGPT support**: Automatically loads and processes ShareGPT data from multiple files
- **Smart data sampling**: Intelligent sampling strategies to select representative data from large datasets
- **Progress monitoring**: Comprehensive logging of training progress, loss, and metrics
- **MLC-LLM compatible**: Direct conversion to q4f16_1 format without external dependencies
- **Llama3.2-1B optimized**: Pre-configured for Llama3.2-1B architecture