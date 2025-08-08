#!/usr/bin/env python3
"""
Validate QAT-trained model and conversion
"""

import os
import sys
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Validate QAT model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to QAT-trained model")
    parser.add_argument("--prompt", type=str, 
                       default="Hello, how are you?",
                       help="Test prompt for generation")
    parser.add_argument("--max_length", type=int, default=200,
                       help="Maximum generation length")
    
    return parser.parse_args()


def validate_model(model_path: str, prompt: str, max_length: int):
    """Validate model by running inference"""
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model type: {model.config.model_type}")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test generation
        logger.info(f"Testing generation with prompt: '{prompt}'")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info("Generation successful!")
        logger.info("=" * 50)
        logger.info("GENERATED TEXT:")
        logger.info("=" * 50)
        logger.info(generated_text)
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation function"""
    args = parse_arguments()
    
    logger.info("=" * 50)
    logger.info("QAT Model Validation")
    logger.info("=" * 50)
    
    # Validate model
    success = validate_model(args.model_path, args.prompt, args.max_length)
    
    if success:
        logger.info("Model validation completed successfully!")
        logger.info("The QAT-trained model is working correctly.")
    else:
        logger.error("Model validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()