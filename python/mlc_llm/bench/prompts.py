"""MLC LLM bench prompts generator"""
import random
from pathlib import Path
from typing import Optional


class PromptsGenerator:  # pylint: disable=too-few-public-methods
    """
    Generates prompts of a specified token length from a text file containing potential prompts.
    """

    def __init__(self, file_path: Optional[str] = None) -> None:
        """
        Initializes the PromptsGenerator with the file path and tokenizer.

        Parameters
        ----------
        file_path : Optional[str]
            The path to the text file containing the prompts.
        """
        from transformers import (  # pylint: disable=import-outside-toplevel,import-error
            LlamaTokenizerFast,
        )

        # TODO(yongwww): Add a default plain prompts source
        prompt_path = Path(file_path) if file_path else Path(__file__).parent / "prompts.txt"
        with prompt_path.open("r") as file:
            self.source_prompts = file.readlines()
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

    def _count_tokens(self, text: str) -> int:
        """Get the number of tokens.

        Parameters
        ----------
        text : str
            The text to tokenize.

        Returns
        -------
        output : int
            The number of tokens
        """
        return len(self.tokenizer.encode(text))

    def generate_prompt(
        self, tokens_mean: int, tokens_stddev: Optional[int] = 0, seed: Optional[int] = 11111
    ) -> str:
        """
        Generates a prompt that closely matches the desired token count.

        Parameters
        ----------
        token_mean : int
            The desired mean number of tokens in the prompt.

        token_stddev : Optional[int]
            The desired standard deviation of tokens in the prompt.

        seed : Optional[int]
            The seed for the random number generator.

        Returns
        -------
        out: str
            A prompt string with the specified number of tokens.
        """
        assert tokens_mean > 0, "The mean number of tokens must be greater than 0."
        random.seed(seed)
        num_out_tokens = (
            int(random.gauss(tokens_mean, tokens_stddev)) if tokens_stddev else tokens_mean
        )
        if num_out_tokens <= 0:
            num_out_tokens = tokens_mean
        random.shuffle(self.source_prompts)
        remaining_num_tokens = num_out_tokens
        out_prompt = ""
        while remaining_num_tokens > 0:
            for tokens in self.source_prompts:
                num_tokens = self._count_tokens(tokens)
                if remaining_num_tokens - num_tokens < 0:
                    out_prompt += tokens[:remaining_num_tokens]
                    remaining_num_tokens = 0
                    break
                out_prompt += tokens
                remaining_num_tokens -= num_tokens
        self._count_tokens(out_prompt)
        return out_prompt
