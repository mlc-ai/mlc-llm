"""
Data loading utilities for ShareGPT format with multi-file support
"""

import json
import os
import glob
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ShareGPTDataLoader:
    """Data loader for ShareGPT format supporting multiple files"""
    
    def __init__(self, data_paths: List[str], validate_format: bool = True):
        """
        Initialize ShareGPT data loader
        
        Args:
            data_paths: List of file paths or directories containing ShareGPT data
            validate_format: Whether to validate ShareGPT format
        """
        self.data_paths = data_paths
        self.validate_format = validate_format
        self.file_list = self._discover_files()
        
        logger.info(f"Discovered {len(self.file_list)} data files")
        
    def _discover_files(self) -> List[str]:
        """Discover all data files from paths"""
        files = []
        
        for path in self.data_paths:
            if os.path.isfile(path):
                files.append(path)
            elif os.path.isdir(path):
                # Search for common data file extensions
                patterns = ['*.json', '*.jsonl', '*.txt']
                for pattern in patterns:
                    files.extend(glob.glob(os.path.join(path, pattern)))
                    files.extend(glob.glob(os.path.join(path, '**', pattern), recursive=True))
            else:
                logger.warning(f"Path not found: {path}")
        
        # Remove duplicates and sort
        files = sorted(list(set(files)))
        logger.info(f"Found data files: {[os.path.basename(f) for f in files[:10]]}")
        if len(files) > 10:
            logger.info(f"... and {len(files) - 10} more files")
            
        return files
    
    def load_all_conversations(self) -> List[Dict[str, Any]]:
        """Load all conversations from all files"""
        all_conversations = []
        
        for file_path in self.file_list:
            try:
                conversations = self._load_single_file(file_path)
                all_conversations.extend(conversations)
                logger.info(f"Loaded {len(conversations)} conversations from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Total conversations loaded: {len(all_conversations)}")
        return all_conversations
    
    def load_conversations_iterator(self) -> Iterator[Dict[str, Any]]:
        """Load conversations as iterator to save memory"""
        for file_path in self.file_list:
            try:
                conversations = self._load_single_file(file_path)
                for conv in conversations:
                    yield conv
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
    
    def _load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load conversations from a single file"""
        conversations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    # JSONL format - one JSON object per line
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if self._is_valid_conversation(data):
                                conversations.append(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at {file_path}:{line_num}: {e}")
                            continue
                else:
                    # Regular JSON format
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        # List of conversations
                        for conv in data:
                            if self._is_valid_conversation(conv):
                                conversations.append(conv)
                    elif isinstance(data, dict):
                        # Single conversation
                        if self._is_valid_conversation(data):
                            conversations.append(data)
                    else:
                        logger.warning(f"Unexpected data format in {file_path}")
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
            
        return conversations
    
    def _is_valid_conversation(self, conv: Dict[str, Any]) -> bool:
        """Validate ShareGPT conversation format"""
        if not self.validate_format:
            return True
        
        # Basic ShareGPT format validation
        if "conversations" not in conv:
            return False
        
        conversations = conv["conversations"]
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False
        
        # Check each turn
        for turn in conversations:
            if not isinstance(turn, dict):
                return False
            
            if "from" not in turn or "value" not in turn:
                return False
            
            speaker = turn["from"]
            content = turn["value"]
            
            # Check valid speakers
            if speaker not in ["human", "user", "gpt", "assistant", "bot"]:
                return False
            
            # Check content is not empty
            if not content or not content.strip():
                return False
        
        return True
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        stats = {
            "total_files": len(self.file_list),
            "total_conversations": 0,
            "total_turns": 0,
            "avg_turns_per_conversation": 0,
            "file_sizes": {},
            "conversation_lengths": [],
        }
        
        for file_path in self.file_list:
            try:
                # Get file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                stats["file_sizes"][os.path.basename(file_path)] = round(file_size, 2)
                
                # Load and analyze conversations
                conversations = self._load_single_file(file_path)
                stats["total_conversations"] += len(conversations)
                
                for conv in conversations:
                    turns = len(conv.get("conversations", []))
                    stats["total_turns"] += turns
                    stats["conversation_lengths"].append(turns)
                    
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                continue
        
        if stats["total_conversations"] > 0:
            stats["avg_turns_per_conversation"] = round(
                stats["total_turns"] / stats["total_conversations"], 2
            )
        
        # Calculate length distribution
        if stats["conversation_lengths"]:
            lengths = stats["conversation_lengths"]
            stats["length_distribution"] = {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(sum(lengths) / len(lengths), 2),
                "median": sorted(lengths)[len(lengths) // 2],
            }
        
        return stats
    
    def preview_data(self, num_samples: int = 3) -> None:
        """Preview sample conversations"""
        print("=== ShareGPT Data Preview ===")
        
        sample_count = 0
        for conv in self.load_conversations_iterator():
            if sample_count >= num_samples:
                break
            
            print(f"\n--- Conversation {sample_count + 1} ---")
            conversations = conv.get("conversations", [])
            
            for i, turn in enumerate(conversations[:4]):  # Show first 4 turns
                speaker = turn.get("from", "unknown")
                content = turn.get("value", "")[:200]  # First 200 chars
                print(f"{speaker}: {content}...")
            
            if len(conversations) > 4:
                print(f"... and {len(conversations) - 4} more turns")
            
            sample_count += 1


# Utility functions
def load_sharegpt_data(data_paths: List[str], validate: bool = True) -> List[Dict[str, Any]]:
    """Convenience function to load ShareGPT data"""
    loader = ShareGPTDataLoader(data_paths, validate_format=validate)
    return loader.load_all_conversations()


def get_data_info(data_paths: List[str]) -> Dict[str, Any]:
    """Get information about ShareGPT data files"""
    loader = ShareGPTDataLoader(data_paths, validate_format=False)
    return loader.get_data_statistics()


def preview_sharegpt_data(data_paths: List[str], num_samples: int = 3) -> None:
    """Preview ShareGPT data"""
    loader = ShareGPTDataLoader(data_paths, validate_format=True)
    loader.preview_data(num_samples)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <data_path1> [data_path2] ...")
        sys.exit(1)
    
    data_paths = sys.argv[1:]
    
    # Preview data
    preview_sharegpt_data(data_paths)
    
    # Show statistics
    stats = get_data_info(data_paths)
    print("\n=== Data Statistics ===")
    for key, value in stats.items():
        if key != "conversation_lengths":  # Skip the long list
            print(f"{key}: {value}")