"""
Data preprocessing and formatting for QAT training
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


class ShareGPTProcessor:
    """Processor for ShareGPT data formatting and preparation"""
    
    def __init__(self, tokenizer, max_length: int = 2048, conversation_template: str = "llama3"):
        """
        Initialize ShareGPT processor
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            conversation_template: Template format for conversations
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversation_template = conversation_template
        
        # Set up pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Conversation templates
        self.templates = self._get_conversation_templates()
        
    def _get_conversation_templates(self) -> Dict[str, Dict[str, str]]:
        """Get conversation templates"""
        return {
            "llama3": {
                "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
                "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
                "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
            },
            "default": {
                "system": "<|system|>\n{content}\n",
                "user": "<|user|>\n{content}\n",
                "assistant": "<|assistant|>\n{content}\n",
            },
            "alpaca": {
                "system": "### System:\n{content}\n\n",
                "user": "### Human: {content}\n",
                "assistant": "### Assistant: {content}\n",
            },
            "vicuna": {
                "system": "SYSTEM: {content}\n",
                "user": "USER: {content}\n",
                "assistant": "ASSISTANT: {content}\n",
            }
        }
    
    def format_conversation(self, conversation: Dict[str, Any], system_message: Optional[str] = None) -> str:
        """
        Format a single conversation using the specified template
        
        Args:
            conversation: ShareGPT conversation dictionary
            system_message: Optional system message to prepend
            
        Returns:
            Formatted conversation string
        """
        conversations = conversation.get("conversations", [])
        if not conversations:
            return ""
        
        template = self.templates.get(self.conversation_template, self.templates["default"])
        formatted_text = ""
        
        # Add system message if provided
        if system_message:
            formatted_text += template["system"].format(content=system_message)
        
        # Process conversation turns
        for turn in conversations:
            speaker = turn.get("from", "")
            content = turn.get("value", "").strip()
            
            if not content:
                continue
            
            # Map speaker names
            if speaker in ["human", "user"]:
                formatted_text += template["user"].format(content=content)
            elif speaker in ["gpt", "assistant", "bot"]:
                formatted_text += template["assistant"].format(content=content)
            else:
                logger.warning(f"Unknown speaker: {speaker}")
                continue
        
        return formatted_text
    
    def clean_conversation(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Clean and validate a conversation
        
        Args:
            conversation: Raw conversation dictionary
            
        Returns:
            Cleaned conversation or None if invalid
        """
        conversations = conversation.get("conversations", [])
        if not conversations:
            return None
        
        cleaned_turns = []
        
        for turn in conversations:
            speaker = turn.get("from", "")
            content = turn.get("value", "").strip()
            
            # Skip empty content
            if not content or len(content) < 10:
                continue
            
            # Skip very long content (likely corrupted)
            if len(content) > 8192:
                content = content[:8192]
            
            # Clean content
            content = self._clean_content(content)
            
            if content:
                cleaned_turns.append({
                    "from": speaker,
                    "value": content
                })
        
        # Need at least one user-assistant pair
        if len(cleaned_turns) < 2:
            return None
        
        return {"conversations": cleaned_turns}
    
    def _clean_content(self, content: str) -> str:
        """Clean individual content string"""
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        # Remove common artifacts
        artifacts = [
            "I'm an AI assistant",
            "I'm sorry, but I can't",
            "I cannot provide",
            "I'm not able to",
        ]
        
        # Don't remove if entire content would be gone
        for artifact in artifacts:
            if artifact.lower() in content.lower() and len(content) > len(artifact) * 2:
                content = content.replace(artifact, "").strip()
        
        return content
    
    def tokenize_conversation(self, formatted_text: str) -> Dict[str, Any]:
        """
        Tokenize formatted conversation
        
        Args:
            formatted_text: Formatted conversation string
            
        Returns:
            Tokenized data dictionary
        """
        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Mask padding tokens in labels
        tokenized["labels"][tokenized["labels"] == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["labels"].squeeze()
        }
    
    def process_conversations(self, conversations: List[Dict[str, Any]], 
                            system_message: Optional[str] = None) -> List[str]:
        """
        Process multiple conversations
        
        Args:
            conversations: List of ShareGPT conversations
            system_message: Optional system message
            
        Returns:
            List of formatted conversation strings
        """
        formatted_texts = []
        processed_count = 0
        
        for conv in conversations:
            # Clean conversation
            cleaned_conv = self.clean_conversation(conv)
            if not cleaned_conv:
                continue
            
            # Format conversation
            formatted_text = self.format_conversation(cleaned_conv, system_message)
            if not formatted_text.strip():
                continue
            
            formatted_texts.append(formatted_text)
            processed_count += 1
        
        logger.info(f"Processed {processed_count}/{len(conversations)} conversations")
        return formatted_texts
    
    def filter_by_length(self, formatted_texts: List[str], 
                        min_tokens: int = 100, max_tokens: int = None) -> List[str]:
        """
        Filter conversations by token length
        
        Args:
            formatted_texts: List of formatted conversation strings
            min_tokens: Minimum token count
            max_tokens: Maximum token count (defaults to max_length)
            
        Returns:
            Filtered list of conversations
        """
        if max_tokens is None:
            max_tokens = self.max_length
        
        filtered_texts = []
        
        for text in formatted_texts:
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
            token_count = tokens.shape[1]
            
            if min_tokens <= token_count <= max_tokens:
                filtered_texts.append(text)
        
        logger.info(f"Filtered {len(filtered_texts)}/{len(formatted_texts)} conversations by length")
        return filtered_texts
    
    def create_dataset(self, formatted_texts: List[str]) -> Dataset:
        """
        Create HuggingFace Dataset from formatted texts
        
        Args:
            formatted_texts: List of formatted conversation strings
            
        Returns:
            HuggingFace Dataset object
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Add labels
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
        
        logger.info(f"Created dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def split_dataset(self, dataset: Dataset, validation_ratio: float = 0.1) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and validation sets
        
        Args:
            dataset: Full dataset
            validation_ratio: Fraction for validation set
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        split_dataset = dataset.train_test_split(test_size=validation_ratio, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} validation")
        return train_dataset, eval_dataset
    
    def get_data_statistics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed data"""
        if not conversations:
            return {}
        
        # Format conversations to get text lengths
        formatted_texts = self.process_conversations(conversations)
        
        token_lengths = []
        char_lengths = []
        
        for text in formatted_texts:
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
            token_lengths.append(tokens.shape[1])
            char_lengths.append(len(text))
        
        stats = {
            "total_conversations": len(conversations),
            "processed_conversations": len(formatted_texts),
            "processing_success_rate": len(formatted_texts) / len(conversations) if conversations else 0,
            "token_length_stats": {
                "min": min(token_lengths) if token_lengths else 0,
                "max": max(token_lengths) if token_lengths else 0,
                "avg": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
                "median": sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0,
            },
            "char_length_stats": {
                "min": min(char_lengths) if char_lengths else 0,
                "max": max(char_lengths) if char_lengths else 0,
                "avg": sum(char_lengths) / len(char_lengths) if char_lengths else 0,
            }
        }
        
        return stats


def create_qat_dataset(conversations: List[Dict[str, Any]], 
                      tokenizer,
                      max_length: int = 2048,
                      conversation_template: str = "llama3",
                      system_message: Optional[str] = None,
                      validation_ratio: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Convenience function to create QAT training dataset
    
    Args:
        conversations: List of ShareGPT conversations
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        conversation_template: Conversation format template
        system_message: Optional system message
        validation_ratio: Fraction for validation set
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    processor = ShareGPTProcessor(
        tokenizer=tokenizer,
        max_length=max_length,
        conversation_template=conversation_template
    )
    
    # Process conversations
    formatted_texts = processor.process_conversations(conversations, system_message)
    
    # Filter by length
    filtered_texts = processor.filter_by_length(formatted_texts)
    
    # Create dataset
    dataset = processor.create_dataset(filtered_texts)
    
    # Split dataset
    train_dataset, eval_dataset = processor.split_dataset(dataset, validation_ratio)
    
    return train_dataset, eval_dataset