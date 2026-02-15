#!/usr/bin/env python3
"""
Main training script for QAT training
"""

import os
import sys
import argparse
import logging
from typing import Optional

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qat_training.config.training_config import QATTrainingConfig, LLAMA_1B_CONFIG
from qat_training.data.data_loader import ShareGPTDataLoader
from qat_training.data.data_processor import create_qat_dataset
from qat_training.data.data_sampler import sample_conversations_for_qat
from qat_training.training.qat_trainer import QATTrainer
from qat_training.conversion.weight_converter import convert_qat_to_mlc

from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="QAT Training for MLC-LLM")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the base SFT-trained model")
    parser.add_argument("--model_type", type=str, default="llama",
                       help="Model type (llama, etc.)")
    
    # Data arguments
    parser.add_argument("--data_paths", type=str, nargs="+", required=True,
                       help="Paths to ShareGPT data files or directories")
    parser.add_argument("--sample_count", type=int, default=30000,
                       help="Number of samples for QAT training")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--sampling_strategy", type=str, default="balanced",
                       choices=["random", "diverse", "quality", "stratified", "balanced"],
                       help="Data sampling strategy")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./qat_outputs",
                       help="Output directory for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    
    # Advanced arguments
    parser.add_argument("--conversation_template", type=str, default="llama3",
                       help="Conversation template format")
    parser.add_argument("--system_message", type=str, default=None,
                       help="Optional system message")
    parser.add_argument("--validation_ratio", type=float, default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Conversion arguments
    parser.add_argument("--convert_to_mlc", action="store_true",
                       help="Convert trained model to MLC format")
    parser.add_argument("--mlc_output_dir", type=str, default=None,
                       help="Output directory for MLC conversion")
    
    # Utility arguments
    parser.add_argument("--preview_data", action="store_true",
                       help="Preview data samples before training")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Load configuration from file")
    parser.add_argument("--save_config", type=str, default=None,
                       help="Save configuration to file")
    
    return parser.parse_args()


def load_and_sample_data(data_paths, sample_count, sampling_strategy, seed=42):
    """Load and sample training data"""
    logger.info("Loading ShareGPT data...")
    
    # Load data
    data_loader = ShareGPTDataLoader(data_paths, validate_format=True)
    all_conversations = data_loader.load_all_conversations()
    
    # Show data statistics
    stats = data_loader.get_data_statistics()
    logger.info(f"Data statistics: {stats}")
    
    # Sample data for QAT
    logger.info(f"Sampling {sample_count} conversations using '{sampling_strategy}' strategy...")
    sampled_conversations = sample_conversations_for_qat(
        all_conversations, 
        target_count=sample_count,
        strategy=sampling_strategy,
        seed=seed
    )
    
    logger.info(f"Sampled {len(sampled_conversations)} conversations for QAT training")
    return sampled_conversations


def create_config_from_args(args) -> QATTrainingConfig:
    """Create training configuration from arguments"""
    if args.config_file:
        logger.info(f"Loading configuration from: {args.config_file}")
        config = QATTrainingConfig.load(args.config_file)
        # Override with command line arguments
        config.base_model_path = args.model_path
        config.data_paths = args.data_paths
    else:
        # Create config from scratch
        config = QATTrainingConfig(
            base_model_path=args.model_path,
            model_type=args.model_type,
            data_paths=args.data_paths,
            sample_count=args.sample_count,
            max_length=args.max_length,
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            conversation_template=args.conversation_template,
            system_message=args.system_message,
            validation_ratio=args.validation_ratio,
        )
    
    return config


def main():
    """Main training function"""
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("QAT Training for MLC-LLM")
    logger.info("=" * 60)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
        
        logger.info(f"Training configuration:")
        logger.info(f"  Model: {config.base_model_path}")
        logger.info(f"  Data files: {len(config.data_paths)}")
        logger.info(f"  Sample count: {config.sample_count}")
        logger.info(f"  Max length: {config.max_length}")
        logger.info(f"  Batch size: {config.per_device_train_batch_size}")
        logger.info(f"  Epochs: {config.num_train_epochs}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Output dir: {config.output_dir}")
        
        # Load and sample data
        sampled_conversations = load_and_sample_data(
            config.data_paths,
            config.sample_count,
            args.sampling_strategy,
            args.seed
        )
        
        # Preview data if requested
        if args.preview_data:
            logger.info("Previewing data samples...")
            for i, conv in enumerate(sampled_conversations[:3]):
                logger.info(f"Sample {i+1}:")
                conversations = conv.get("conversations", [])
                for turn in conversations[:4]:
                    speaker = turn.get("from", "unknown")
                    content = turn.get("value", "")[:100]
                    logger.info(f"  {speaker}: {content}...")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)
        
        # Create datasets
        logger.info("Processing conversations into datasets...")
        train_dataset, eval_dataset = create_qat_dataset(
            sampled_conversations,
            tokenizer,
            max_length=config.max_length,
            conversation_template=config.conversation_template,
            system_message=config.system_message,
            validation_ratio=config.validation_ratio
        )
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        # Initialize trainer
        logger.info("Initializing QAT trainer...")
        trainer = QATTrainer(config)
        
        # Start training
        logger.info("Starting QAT training...")
        train_result = trainer.train(train_dataset, eval_dataset)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Convert to MLC format if requested
        if args.convert_to_mlc:
            mlc_output_dir = args.mlc_output_dir or os.path.join(config.output_dir, "mlc_format")
            logger.info(f"Converting to MLC-LLM format: {mlc_output_dir}")
            
            # Get the trained model
            trained_model = trainer.get_model_for_conversion()
            
            # Convert to MLC format
            convert_qat_to_mlc(trained_model, mlc_output_dir)
            
            logger.info("Conversion to MLC format completed!")
            logger.info(f"Use with MLC-LLM: mlc_llm convert_weight {mlc_output_dir} --quantization q4f16_1")
        
        logger.info("=" * 60)
        logger.info("QAT Training Completed Successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()