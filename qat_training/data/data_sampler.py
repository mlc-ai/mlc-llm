"""
Smart data sampling strategies for QAT training
"""

import random
import json
import math
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class DataSampler:
    """Smart sampling strategies for QAT training data"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize data sampler
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
    def random_sample(self, conversations: List[Dict[str, Any]], 
                     target_count: int) -> List[Dict[str, Any]]:
        """
        Simple random sampling
        
        Args:
            conversations: List of all conversations
            target_count: Number of samples to return
            
        Returns:
            Randomly sampled conversations
        """
        if len(conversations) <= target_count:
            return conversations
        
        sampled = random.sample(conversations, target_count)
        logger.info(f"Random sampling: {len(sampled)} from {len(conversations)}")
        return sampled
    
    def diverse_sample(self, conversations: List[Dict[str, Any]], 
                      target_count: int) -> List[Dict[str, Any]]:
        """
        Diverse sampling based on conversation characteristics
        
        Args:
            conversations: List of all conversations
            target_count: Number of samples to return
            
        Returns:
            Diversely sampled conversations
        """
        if len(conversations) <= target_count:
            return conversations
        
        # Categorize conversations by characteristics
        categorized = self._categorize_conversations(conversations)
        
        # Sample from each category proportionally
        sampled = []
        total_categories = len(categorized)
        
        for category, convs in categorized.items():
            # Calculate proportion for this category
            category_ratio = len(convs) / len(conversations)
            category_target = max(1, int(target_count * category_ratio))
            
            # Sample from this category
            if len(convs) >= category_target:
                category_sample = random.sample(convs, category_target)
            else:
                category_sample = convs
            
            sampled.extend(category_sample)
            logger.info(f"Category '{category}': {len(category_sample)} samples")
        
        # If we haven't reached target, randomly sample more
        if len(sampled) < target_count:
            remaining = [c for c in conversations if c not in sampled]
            need = target_count - len(sampled)
            if remaining and need > 0:
                additional = random.sample(remaining, min(need, len(remaining)))
                sampled.extend(additional)
        
        # If we exceeded target, randomly reduce
        if len(sampled) > target_count:
            sampled = random.sample(sampled, target_count)
        
        logger.info(f"Diverse sampling: {len(sampled)} from {len(conversations)}")
        return sampled
    
    def quality_sample(self, conversations: List[Dict[str, Any]], 
                      target_count: int) -> List[Dict[str, Any]]:
        """
        Quality-based sampling using conversation scoring
        
        Args:
            conversations: List of all conversations
            target_count: Number of samples to return
            
        Returns:
            Quality-selected conversations
        """
        if len(conversations) <= target_count:
            return conversations
        
        # Score each conversation
        scored_conversations = []
        for conv in conversations:
            score = self._calculate_quality_score(conv)
            scored_conversations.append((score, conv))
        
        # Sort by quality score (descending)
        scored_conversations.sort(key=lambda x: x[0], reverse=True)
        
        # Take top conversations
        sampled = [conv for _, conv in scored_conversations[:target_count]]
        
        avg_score = sum(score for score, _ in scored_conversations[:target_count]) / target_count
        logger.info(f"Quality sampling: {len(sampled)} from {len(conversations)}, avg score: {avg_score:.2f}")
        
        return sampled
    
    def stratified_sample(self, conversations: List[Dict[str, Any]], 
                         target_count: int, 
                         stratify_by: str = "length") -> List[Dict[str, Any]]:
        """
        Stratified sampling to ensure representation across strata
        
        Args:
            conversations: List of all conversations
            target_count: Number of samples to return
            stratify_by: Stratification strategy ("length", "turns", "topics")
            
        Returns:
            Stratified sample of conversations
        """
        if len(conversations) <= target_count:
            return conversations
        
        # Create strata
        strata = self._create_strata(conversations, stratify_by)
        
        # Sample from each stratum
        sampled = []
        total_strata = len(strata)
        
        for stratum_name, stratum_conversations in strata.items():
            # Calculate target for this stratum
            stratum_ratio = len(stratum_conversations) / len(conversations)
            stratum_target = max(1, int(target_count * stratum_ratio))
            
            # Sample from stratum
            if len(stratum_conversations) >= stratum_target:
                stratum_sample = random.sample(stratum_conversations, stratum_target)
            else:
                stratum_sample = stratum_conversations
            
            sampled.extend(stratum_sample)
            logger.info(f"Stratum '{stratum_name}': {len(stratum_sample)} samples")
        
        # Adjust to exact target
        if len(sampled) > target_count:
            sampled = random.sample(sampled, target_count)
        elif len(sampled) < target_count:
            remaining = [c for c in conversations if c not in sampled]
            need = target_count - len(sampled)
            if remaining and need > 0:
                additional = random.sample(remaining, min(need, len(remaining)))
                sampled.extend(additional)
        
        logger.info(f"Stratified sampling: {len(sampled)} from {len(conversations)}")
        return sampled
    
    def balanced_sample(self, conversations: List[Dict[str, Any]], 
                       target_count: int) -> List[Dict[str, Any]]:
        """
        Balanced sampling combining multiple strategies
        
        Args:
            conversations: List of all conversations
            target_count: Number of samples to return
            
        Returns:
            Balanced sample using multiple strategies
        """
        if len(conversations) <= target_count:
            return conversations
        
        # Divide target among different strategies
        strategies = {
            "quality": 0.4,    # 40% from quality selection
            "diverse": 0.3,    # 30% from diverse selection
            "random": 0.3,     # 30% from random selection
        }
        
        sampled = []
        used_conversations = set()
        
        for strategy, ratio in strategies.items():
            strategy_target = int(target_count * ratio)
            
            # Get available conversations (not yet used)
            available = [c for c in conversations 
                        if id(c) not in used_conversations]
            
            if not available:
                break
            
            # Apply strategy
            if strategy == "quality":
                strategy_sample = self.quality_sample(available, strategy_target)
            elif strategy == "diverse":
                strategy_sample = self.diverse_sample(available, strategy_target)
            else:  # random
                strategy_sample = self.random_sample(available, strategy_target)
            
            # Add to final sample
            sampled.extend(strategy_sample)
            used_conversations.update(id(c) for c in strategy_sample)
            
            logger.info(f"Strategy '{strategy}': {len(strategy_sample)} samples")
        
        # Fill remaining slots with random sampling
        if len(sampled) < target_count:
            available = [c for c in conversations 
                        if id(c) not in used_conversations]
            need = target_count - len(sampled)
            if available and need > 0:
                additional = random.sample(available, min(need, len(available)))
                sampled.extend(additional)
        
        logger.info(f"Balanced sampling: {len(sampled)} from {len(conversations)}")
        return sampled
    
    def _categorize_conversations(self, conversations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize conversations by characteristics"""
        categories = defaultdict(list)
        
        for conv in conversations:
            conversations_list = conv.get("conversations", [])
            
            # Categorize by conversation length
            total_length = sum(len(turn.get("value", "")) for turn in conversations_list)
            
            if total_length < 500:
                length_category = "short"
            elif total_length < 2000:
                length_category = "medium"
            else:
                length_category = "long"
            
            # Categorize by number of turns
            num_turns = len(conversations_list)
            if num_turns <= 2:
                turn_category = "simple"
            elif num_turns <= 6:
                turn_category = "multi_turn"
            else:
                turn_category = "complex"
            
            # Combine categories
            category = f"{length_category}_{turn_category}"
            categories[category].append(conv)
        
        return dict(categories)
    
    def _calculate_quality_score(self, conversation: Dict[str, Any]) -> float:
        """Calculate quality score for a conversation"""
        conversations_list = conversation.get("conversations", [])
        if not conversations_list:
            return 0.0
        
        score = 0.0
        
        # Factor 1: Number of turns (more interaction = better)
        num_turns = len(conversations_list)
        score += min(num_turns * 0.1, 1.0)  # Cap at 1.0 for 10+ turns
        
        # Factor 2: Content quality
        for turn in conversations_list:
            content = turn.get("value", "")
            content_length = len(content)
            
            # Optimal length range
            if 50 <= content_length <= 1000:
                score += 0.3
            elif content_length < 20:
                score -= 0.2  # Too short
            elif content_length > 3000:
                score -= 0.1  # Too long
            
            # Check for quality indicators
            if any(indicator in content.lower() for indicator in 
                  ["explain", "describe", "analyze", "compare", "example"]):
                score += 0.1
            
            # Penalize low-quality indicators
            if any(indicator in content.lower() for indicator in 
                  ["i can't help", "i cannot", "i'm sorry", "i don't know"]):
                score -= 0.2
        
        # Factor 3: Balance between human and assistant
        human_turns = sum(1 for turn in conversations_list 
                         if turn.get("from") in ["human", "user"])
        assistant_turns = sum(1 for turn in conversations_list 
                            if turn.get("from") in ["gpt", "assistant", "bot"])
        
        balance_ratio = min(human_turns, assistant_turns) / max(human_turns, assistant_turns, 1)
        score += balance_ratio * 0.5
        
        # Factor 4: Uniqueness (simple check for repetition)
        all_content = " ".join(turn.get("value", "") for turn in conversations_list)
        words = all_content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.3
        
        return max(0.0, score)
    
    def _create_strata(self, conversations: List[Dict[str, Any]], 
                      stratify_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """Create strata for stratified sampling"""
        strata = defaultdict(list)
        
        for conv in conversations:
            conversations_list = conv.get("conversations", [])
            
            if stratify_by == "length":
                total_length = sum(len(turn.get("value", "")) for turn in conversations_list)
                if total_length < 800:
                    stratum = "short"
                elif total_length < 2500:
                    stratum = "medium"
                else:
                    stratum = "long"
            
            elif stratify_by == "turns":
                num_turns = len(conversations_list)
                if num_turns <= 2:
                    stratum = "few_turns"
                elif num_turns <= 6:
                    stratum = "medium_turns"
                else:
                    stratum = "many_turns"
            
            else:  # Default to length
                total_length = sum(len(turn.get("value", "")) for turn in conversations_list)
                stratum = f"length_{total_length // 1000}k"
            
            strata[stratum].append(conv)
        
        return dict(strata)


def sample_conversations_for_qat(conversations: List[Dict[str, Any]], 
                                target_count: int,
                                strategy: str = "balanced",
                                seed: int = 42) -> List[Dict[str, Any]]:
    """
    Convenience function to sample conversations for QAT training
    
    Args:
        conversations: List of all conversations
        target_count: Target number of samples
        strategy: Sampling strategy ("random", "diverse", "quality", "stratified", "balanced")
        seed: Random seed
        
    Returns:
        Sampled conversations
    """
    sampler = DataSampler(seed=seed)
    
    if strategy == "random":
        return sampler.random_sample(conversations, target_count)
    elif strategy == "diverse":
        return sampler.diverse_sample(conversations, target_count)
    elif strategy == "quality":
        return sampler.quality_sample(conversations, target_count)
    elif strategy == "stratified":
        return sampler.stratified_sample(conversations, target_count)
    elif strategy == "balanced":
        return sampler.balanced_sample(conversations, target_count)
    else:
        logger.warning(f"Unknown strategy '{strategy}', using 'balanced'")
        return sampler.balanced_sample(conversations, target_count)