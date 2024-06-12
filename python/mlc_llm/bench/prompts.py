"""MLC LLM bench prompts generator"""

import json
import random
from collections import defaultdict
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
        json_prompts_path: Optional[str] = None,
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

        json_prompts_path : Optional[str]
            The path to the file containing the source json prompts. This file a
            .jsonl file where each line is a JSON object formatted as
            {"messages": List[Dict[str, Any]], "response_format": Dict[str, Any]}.

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
                    self.prompts.append(json_line)
        else:
            if not prompts_path:
                prompts_path = Path(__file__).parent / "prompts.txt"  # type: ignore
            with open(prompts_path, "r", encoding="utf-8") as file:
                prompt_line = file.readline()
                prompt_tokens = self._count_tokens(prompt_line)
                self.prompts.append({"prompt": prompt_line, "prompt_tokens": prompt_tokens})
        if json_prompts_path:
            self.json_prompts = defaultdict(list)
            with open(json_prompts_path, "r", encoding="utf-8") as file:
                for line in file:
                    json_line = json.loads(line)
                    assert (
                        "messages" in json_line
                    ), "The messages field is required in the JSONL file."
                    assert (
                        "response_format" in json_line
                    ), "The response_format field is required in the JSONL file."
                    self.json_prompts[json.dumps(json_line["response_format"]["schema"])].append(
                        json_line["messages"]
                    )
        else:
            self.json_prompts = None

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

    def generate_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a prompt based on the params, e.g. prompt_tokens, response_format.

        Parameters
        ----------
        params : Dict[str, Any]
            The desired mean number of tokens in the prompt.

        Returns
        -------
        override_params: Dict[str, Any]
            The params to override the original request, e.g. messages, response_format.
        """
        if "response_format" in params:
            response_format = params["response_format"]
            if response_format.get("type") == "json_object":
                if response_format.get("schema") in self.json_prompts:
                    assert len(self.json_prompts[response_format["schema"]]) > 0
                    return {"messages": random.choice(self.json_prompts[response_format["schema"]])}
                schema, prompts = random.choice(list(self.json_prompts.items()))
                response_format["schema"] = schema
                return {"messages": random.choice(prompts), "response_format": response_format}
        tokens_mean = params.get("prompt_tokens", 128)
        assert tokens_mean > 0, "The mean number of tokens must be greater than 0."
        remaining_prompt_tokens = tokens_mean
        result_prompt = ""
        override_params = None
        while remaining_prompt_tokens > 0:
            prompt_dict = random.choice(self.prompts)
            cur_prompt_tokens = prompt_dict["prompt_tokens"]
            cur_prompt = prompt_dict["prompt"]
            if override_params is None:
                override_params = prompt_dict["override_params"]
            if remaining_prompt_tokens - cur_prompt_tokens < 0:
                result_prompt += cur_prompt[:remaining_prompt_tokens]
                remaining_prompt_tokens = 0
                break
            result_prompt += cur_prompt
            remaining_prompt_tokens -= cur_prompt_tokens
        return {"messages": [{"role": "system", "content": result_prompt}]}
