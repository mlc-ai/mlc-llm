"""MLC LLM bench prompts generator"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlc_llm.support import logging

logging.enable_logging()
logger = logging.getLogger(__name__)


class PromptsGenerator:  # pylint: disable=too-few-public-methods
    """
    Generates prompts of a specified token length from a text file containing potential prompts.
    """

    def __init__(
        self,
        prompts_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        seed: Optional[int] = 11111,
    ) -> None:
        """
        Initializes the PromptsGenerator with the file path and tokenizer.

        Parameters
        ----------
        prompts_path : Optional[str]
            The path to the file containing the source prompts. This file can be
            a plain text file, with each line representing a separate prompt str,
            or a .jsonl file where each line is a JSON object formatted as
            {"prompt": "prompt text", "prompt_tokens": 10}.

        tokenizer : Optional[Any]
            The tokenizer object to use for tokenizing the prompts.

        seed : Optional[int]
            The seed for the random number generator.
        """
        random.seed(seed)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            from transformers import (  # pylint: disable=import-outside-toplevel,import-error
                LlamaTokenizerFast,
            )

            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                "hf-internal-testing/llama-tokenizer"
            )
            logger.warning("No tokenizer provided. Using default tokenizer.")

        self.prompts: List[Dict] = []
        if prompts_path is not None and prompts_path.endswith(".jsonl"):
            with open(prompts_path, "r", encoding="utf-8") as file:
                for line in file:
                    json_line = json.loads(line)
                    assert "prompt" in json_line, "The prompt field is required in the JSONL file."
                    if "prompt_tokens" not in json_line:
                        json_line["prompt_tokens"] = self._count_tokens(json_line["prompt"])
                    self.prompts.append(json.loads(line))
                self.prompts = [json.loads(line) for line in file]
        else:
            if not prompts_path:
                prompts_path = Path(__file__).parent / "prompts.txt"  # type: ignore
            with open(prompts_path, "r", encoding="utf-8") as file:
                prompt_line = file.readline()
                prompt_tokens = self._count_tokens(prompt_line)
                self.prompts.append({"prompt": prompt_line, "prompt_tokens": prompt_tokens})

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

    def generate_prompt(self, tokens_mean: int, tokens_stddev: Optional[int] = 0) -> str:
        """
        Generates a prompt that closely matches the desired token count.

        Parameters
        ----------
        token_mean : int
            The desired mean number of tokens in the prompt.

        token_stddev : Optional[int]
            The desired standard deviation of tokens in the prompt.

        Returns
        -------
        out: str
            A prompt string with the specified number of tokens.
        """
        assert tokens_mean > 0, "The mean number of tokens must be greater than 0."
        out_prompt_tokens = (
            int(random.gauss(tokens_mean, tokens_stddev)) if tokens_stddev else tokens_mean
        )
        if out_prompt_tokens <= 0:
            out_prompt_tokens = tokens_mean
        remaining_prompt_tokens = out_prompt_tokens
        result_prompt = ""
        while remaining_prompt_tokens > 0:
            prompt_dict = random.choice(self.prompts)
            cur_prompt_tokens = prompt_dict["prompt_tokens"]
            cur_prompt = prompt_dict["prompt"]
            if remaining_prompt_tokens - cur_prompt_tokens < 0:
                result_prompt += cur_prompt[:remaining_prompt_tokens]
                remaining_prompt_tokens = 0
                break
            result_prompt += cur_prompt
            remaining_prompt_tokens -= cur_prompt_tokens
        self._count_tokens(result_prompt)
        return result_prompt
