"""
Sampling parameters for text generation.

based on https://github.com/vllm-project/vllm/blob/ac5cf86aa6aebbf9e42df51f7e377fbee85bc703/vllm/sampling_params.py
"""
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Dict, Optional


_SAMPLING_EPS = 1e-5


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1


@dataclass
class SamplingParams:
    """
    Sampling parameters for text generation.

    Args:
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Positive float that penalizes new tokens based on
            whether they appear in the generated text so far. Values > 1 encourage
            the model to use new tokens, while values < 1 encourage the model
            to repeat tokens. The penalty works as multiplication factor, it
            multiplys on logprob or divides the probabilities.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        logit_bias: The bias applied on the logit before sampling. Must be in
            [-100, 100].
    """

    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logit_bias: Optional[Dict[int, float]] = None
    appeared_tokens_freq: Dict[int, int] = None
    logit_bias_index: list[int] = None
    logit_bias_value: list[float] = None

    def __post_init__(self):
        self.appeared_tokens_freq = {}
        if self.logit_bias:
            self.logit_bias_index = list(self.logit_bias.keys())
            self.logit_bias_value = list(self.logit_bias.values())
        self._verify_args()
        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self._verify_greedy_sampling()

    def _verify_args(self) -> None:
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disable), or at least 1, " f"got {self.top_k}."
            )
        if self.logit_bias:
            for token, bias in self.logit_bias.items():
                if not -100 <= bias <= 100:
                    raise ValueError(
                        f"logit bias must be in [-100, 100], got {bias} for token {token}."
                    )

    def _verify_greedy_sampling(self) -> None:
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        return SamplingType.RANDOM
