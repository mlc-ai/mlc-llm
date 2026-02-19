"""
QAT Training implementation for MLC-LLM compatibility
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from ..config.training_config import QATTrainingConfig
from ..config.model_config import get_model_config
from .metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


class QATTrainer:
    """Quantization Aware Training implementation"""
    
    def __init__(self, config: QATTrainingConfig):
        """
        Initialize QAT trainer
        
        Args:
            config: QAT training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.metrics_logger = MetricsLogger(config.output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def setup_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Setup model and tokenizer for QAT training
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from: {self.config.base_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_path,
            trust_remote_code=True,
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # BitsAndBytes configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(**self.config.quantization_config)
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        logger.info(f"Model loaded with quantization: {self.model.config}")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        lora_config = LoraConfig(**self.config.lora_config)
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return self.model, self.tokenizer
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Trainer:
        """
        Setup HuggingFace Trainer for QAT
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Configured Trainer
        """
        # Convert config to TrainingArguments
        training_args = self.config.to_training_args()
        
        # Custom data collator for causal LM
        def data_collator(batch):
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'labels': torch.stack([item['labels'] for item in batch])
            }
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[self.metrics_logger]
        )
        
        logger.info("Trainer setup completed")
        return self.trainer
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Execute QAT training
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Training results
        """
        logger.info("Starting QAT training...")
        
        # Setup model and tokenizer if not done
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Log training info
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation samples: {len(eval_dataset)}")
        
        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {self.config.num_train_epochs}")
        logger.info(f"  - Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"  - Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"  - Learning rate: {self.config.learning_rate}")
        logger.info(f"  - Max length: {self.config.max_length}")
        
        # Start training
        try:
            train_result = self.trainer.train()
            
            # Log training results
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            # Save final model
            self.save_model()
            
            # Save training metrics
            self.metrics_logger.save_final_metrics(train_result)
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Save the trained model
        
        Args:
            output_dir: Optional custom output directory
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Save model and tokenizer
        final_model_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        
        # Save training config
        config_path = os.path.join(final_model_dir, "training_config.json")
        self.config.save(config_path)
        
        logger.info(f"Model saved to: {final_model_dir}")
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        logger.info("Evaluating model...")
        eval_results = self.trainer.evaluate(eval_dataset)
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_results
    
    def get_model_for_conversion(self):
        """
        Get the trained model ready for MLC conversion
        
        Returns:
            Model ready for weight conversion
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Merge LoRA weights with base model
        logger.info("Merging LoRA weights for conversion...")
        merged_model = self.model.merge_and_unload()
        
        return merged_model
    
    def export_for_mlc(self, output_dir: str) -> None:
        """
        Export model in format ready for MLC-LLM conversion
        
        Args:
            output_dir: Directory to save the exported model
        """
        logger.info(f"Exporting model for MLC-LLM to: {output_dir}")
        
        # Get merged model
        merged_model = self.get_model_for_conversion()
        
        # Save merged model
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Create a marker file indicating this is QAT-trained
        marker_file = os.path.join(output_dir, "qat_trained.txt")
        with open(marker_file, 'w') as f:
            f.write("This model was trained with Quantization Aware Training (QAT)\n")
            f.write(f"Original base model: {self.config.base_model_path}\n")
            f.write(f"Training samples: {self.config.sample_count}\n")
            f.write(f"Target quantization: q4f16_1\n")
        
        logger.info(f"Model exported successfully to: {output_dir}")
        logger.info("Ready for MLC-LLM conversion with: mlc_llm convert_weight --quantization q4f16_1")


def create_qat_trainer(config: QATTrainingConfig) -> QATTrainer:
    """
    Convenience function to create QAT trainer
    
    Args:
        config: QAT training configuration
        
    Returns:
        QATTrainer instance
    """
    return QATTrainer(config)


def train_qat_model(config: QATTrainingConfig, 
                   train_dataset: Dataset, 
                   eval_dataset: Optional[Dataset] = None) -> QATTrainer:
    """
    Convenience function to run complete QAT training
    
    Args:
        config: QAT training configuration
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        
    Returns:
        Trained QATTrainer instance
    """
    trainer = QATTrainer(config)
    trainer.train(train_dataset, eval_dataset)
    return trainer