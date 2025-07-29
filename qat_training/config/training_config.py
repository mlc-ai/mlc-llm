"""
Training configuration for QAT training
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class QATTrainingConfig:
    """Configuration for QAT Training"""
    
    # Model Configuration
    base_model_path: str = ""  # Path to SFT-trained Llama3.2-1B model
    model_type: str = "llama"
    model_size: str = "1b"
    
    # Data Configuration
    data_paths: List[str] = field(default_factory=list)  # List of ShareGPT data files
    data_format: str = "sharegpt"
    max_length: int = 2048
    sample_count: int = 30000  # Number of samples for QAT
    validation_ratio: float = 0.1
    
    # QAT Configuration
    quantization_config: Dict[str, Any] = field(default_factory=lambda: {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_storage_dtype": "uint8",
    })
    
    # LoRA Configuration for QAT
    lora_config: Dict[str, Any] = field(default_factory=lambda: {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    })
    
    # Training Arguments
    output_dir: str = "./qat_outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Logging and Saving
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    
    # Hardware Configuration
    fp16: bool = True
    bf16: bool = False
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 4
    
    # Advanced Options
    remove_unused_columns: bool = False
    label_smoothing_factor: float = 0.1
    report_to: Optional[str] = None  # "wandb", "tensorboard", None
    
    # Conversation Template
    conversation_template: str = "llama3"
    system_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.base_model_path:
            raise ValueError("base_model_path must be specified")
        
        if not self.data_paths:
            raise ValueError("data_paths must be specified")
        
        # Validate data files exist
        for path in self.data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Adjust batch size based on available memory
        if self.model_size == "1b":
            # For 1B models, we can use larger batch sizes
            if self.per_device_train_batch_size < 4:
                print("Note: You might be able to increase batch size for 1B model")
        
        # Validate conversation template
        valid_templates = ["llama3", "default", "alpaca", "vicuna"]
        if self.conversation_template not in valid_templates:
            print(f"Warning: Unknown conversation template '{self.conversation_template}'. "
                  f"Valid options: {valid_templates}")

    def to_training_args(self):
        """Convert to HuggingFace TrainingArguments format"""
        from transformers import TrainingArguments
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            evaluation_strategy=self.evaluation_strategy,
            fp16=self.fp16,
            bf16=self.bf16,
            dataloader_pin_memory=self.dataloader_pin_memory,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            label_smoothing_factor=self.label_smoothing_factor,
            report_to=self.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving"""
        return {
            "base_model_path": self.base_model_path,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "data_paths": self.data_paths,
            "data_format": self.data_format,
            "max_length": self.max_length,
            "sample_count": self.sample_count,
            "validation_ratio": self.validation_ratio,
            "quantization_config": self.quantization_config,
            "lora_config": self.lora_config,
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "conversation_template": self.conversation_template,
            "system_message": self.system_message,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QATTrainingConfig":
        """Create config from dictionary"""
        return cls(**config_dict)

    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "QATTrainingConfig":
        """Load configuration from file"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different scenarios
LLAMA_1B_CONFIG = QATTrainingConfig(
    model_type="llama",
    model_size="1b",
    max_length=2048,
    sample_count=30000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
)

LLAMA_1B_FAST_CONFIG = QATTrainingConfig(
    model_type="llama", 
    model_size="1b",
    max_length=1024,
    sample_count=10000,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=2,
)

LLAMA_1B_QUALITY_CONFIG = QATTrainingConfig(
    model_type="llama",
    model_size="1b", 
    max_length=2048,
    sample_count=50000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=5,
    warmup_ratio=0.05,
)