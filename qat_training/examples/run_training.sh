#!/bin/bash

# QAT Training Script for Llama3.2-1B
# This script demonstrates how to run QAT training with your ShareGPT data

# Configuration
MODEL_PATH="/path/to/your/llama3.2-1b-sft-model"  # Replace with your SFT model path
DATA_PATHS=(
    "/path/to/sharegpt/file1.json"
    "/path/to/sharegpt/file2.jsonl" 
    "/path/to/sharegpt/directory"
)  # Replace with your ShareGPT data paths

OUTPUT_DIR="./qat_training_outputs"
SAMPLE_COUNT=30000
BATCH_SIZE=2
EPOCHS=3
LEARNING_RATE=1e-4

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "QAT Training for MLC-LLM"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data files: ${#DATA_PATHS[@]}"
echo "Sample count: $SAMPLE_COUNT"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

# Check if at least one data path exists
data_exists=false
for path in "${DATA_PATHS[@]}"; do
    if [ -e "$path" ]; then
        data_exists=true
        break
    fi
done

if [ "$data_exists" = false ]; then
    echo "Error: No valid data paths found"
    echo "Please update DATA_PATHS in this script"
    exit 1
fi

# Run QAT training
echo "Starting QAT training..."
python3 ../scripts/train_qat.py \
    --model_path "$MODEL_PATH" \
    --data_paths "${DATA_PATHS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --sample_count $SAMPLE_COUNT \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --sampling_strategy balanced \
    --conversation_template llama3 \
    --validation_ratio 0.1 \
    --preview_data \
    --convert_to_mlc \
    --mlc_output_dir "$OUTPUT_DIR/mlc_format"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "QAT Training Completed Successfully!"
    echo "========================================"
    echo "Trained model saved to: $OUTPUT_DIR"
    echo "MLC format saved to: $OUTPUT_DIR/mlc_format"
    echo ""
    echo "Next steps:"
    echo "1. Convert to MLC-LLM:"
    echo "   mlc_llm convert_weight $OUTPUT_DIR/mlc_format --quantization q4f16_1 --output ./final_model"
    echo ""
    echo "2. Generate config:"
    echo "   mlc_llm gen_config ./final_model --quantization q4f16_1 --output ./mlc_config"
    echo ""
    echo "3. Compile model:"
    echo "   mlc_llm compile ./mlc_config/mlc-chat-config.json --output ./compiled_model"
    echo ""
    echo "4. Test inference:"
    echo "   mlc_llm chat ./compiled_model --quantization q4f16_1"
else
    echo "========================================"
    echo "Training failed! Check the logs above."
    echo "========================================"
    exit 1
fi