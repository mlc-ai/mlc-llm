#!/usr/bin/env python3
"""
Convert QAT-trained model to MLC-LLM format
"""

import os
import sys
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qat_training.conversion.weight_converter import convert_qat_to_mlc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert QAT model to MLC-LLM format")
    
    parser.add_argument("--qat_model_path", type=str, required=True,
                       help="Path to QAT-trained model")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for MLC format")
    parser.add_argument("--group_size", type=int, default=32,
                       help="Group size for quantization")
    
    return parser.parse_args()


def main():
    """Main conversion function"""
    args = parse_arguments()
    
    logger.info("=" * 50)
    logger.info("Converting QAT Model to MLC-LLM Format")
    logger.info("=" * 50)
    
    try:
        # Load QAT model
        logger.info(f"Loading QAT model from: {args.qat_model_path}")
        qat_model = AutoModelForCausalLM.from_pretrained(
            args.qat_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Convert to MLC format
        logger.info(f"Converting to MLC format: {args.output_dir}")
        convert_qat_to_mlc(qat_model, args.output_dir, args.group_size)
        
        logger.info("Conversion completed successfully!")
        logger.info(f"Use with MLC-LLM:")
        logger.info(f"  mlc_llm convert_weight {args.output_dir} --quantization q4f16_1 --output ./final_model")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()