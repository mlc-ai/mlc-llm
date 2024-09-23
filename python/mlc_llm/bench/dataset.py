"""MLC LLM benchmark dataset classes"""

import argparse
import json
import random
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset  # pylint: disable=import-error
from transformers import AutoTokenizer  # pylint: disable=import-error

from mlc_llm.bench.request_record import Metrics, RequestRecord
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    DebugConfig,
)


class Dataset:  # pylint: disable=too-few-public-methods
    """The dataset base class."""

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        """Get the raw unprocessed request records of the dataset."""
        raise NotImplementedError()


class ShareGPTDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for ShareGPT dataset."""

    _tokenized_dataset: List[Tuple[str, List[int], int]]

    def __init__(self, dataset_path: str, tokenizer: AutoTokenizer) -> None:
        with open(dataset_path, encoding="utf-8") as f:
            raw_dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        _dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in raw_dataset
            if len(data["conversations"]) >= 2
        ]
        # Tokenize the prompts and completions.
        self.tokenizer = tokenizer
        prompts = [prompt for prompt, _ in _dataset]
        prompt_token_ids = list(
            tokenizer(
                prompts,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).input_ids
        )
        completions = [completion for _, completion in _dataset]
        completion_token_ids = tokenizer(
            completions,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        self._tokenized_dataset: List[Tuple[str, List[int], int]] = []
        for i in range(len(_dataset)):
            self._tokenized_dataset.append(
                (prompts[i], prompt_token_ids[i], len(completion_token_ids[i]))
            )

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        request_records = []
        for prompt, input_token_ids, output_length in self._tokenized_dataset:
            input_length = len(input_token_ids)
            # If the request does not have enough length, discard it.
            if input_len is not None and input_length < input_len + 4 * input_len_std:
                continue

            if input_len is not None:
                input_length = round(
                    float(np.random.normal(loc=input_len, scale=input_len_std, size=1)[0])
                )
                input_token_ids = input_token_ids[:input_length]
                input_truncated = True
            else:
                input_truncated = False
            if output_len is not None:
                output_length = round(
                    float(np.random.normal(loc=output_len, scale=output_len_std, size=1)[0])
                )
            elif output_length <= 1:
                continue
            request_records.append(
                RequestRecord(
                    chat_cmpl=ChatCompletionRequest(
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    self.tokenizer.decode(input_token_ids)
                                    if input_truncated
                                    else prompt
                                ),
                            }
                        ],
                        model="",
                        max_tokens=output_length,
                    ),
                    metrics=Metrics(
                        success=False,
                        start_time=0,
                        finish_time=0,
                        end_to_end_latency_s=0,
                        input_tokens=len(input_token_ids),
                    ),
                )
            )
        return request_records


class LLMPerfDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for LLMPerf dataset."""

    def __init__(self, dataset_path: str, num_requests: int, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.num_requests = num_requests

        with open(dataset_path, encoding="utf-8") as f:
            untokenized_data = f.readlines()
        # Tokenize the prompts and completions.
        tokenized_data = tokenizer(
            untokenized_data,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        tokenized_data_lengths = [len(tokens) for tokens in tokenized_data]
        self.dataset: List[Tuple[str, List[int], int]] = list(
            zip(untokenized_data, tokenized_data, tokenized_data_lengths)
        )

    def generate_request_records(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
        input_len_std: float = 250,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        if input_len is None or input_len < 40:
            input_len = 550
        if output_len is None:
            output_len = 150

        request_records = []
        for _ in range(self.num_requests):
            input_length = round(float(np.random.normal(loc=input_len, scale=input_len_std)))
            output_length = round(float(np.random.normal(loc=output_len, scale=output_len_std)))

            prompt = (
                "Randomly stream lines from the following text "
                f"with {output_length} output tokens. "
                "Don't generate eos tokens:\n\n"
            )

            remaining_token_length = input_length - len(self.tokenizer.encode(prompt))

            random.shuffle(self.dataset)

            while remaining_token_length > 0:
                for text, tokens, token_length in self.dataset:
                    if remaining_token_length < token_length:
                        prompt += self.tokenizer.decode(tokens[:remaining_token_length])
                    else:
                        prompt += text

                    remaining_token_length -= token_length
                    if remaining_token_length < 0:
                        break

            request_records.append(
                RequestRecord(
                    chat_cmpl=ChatCompletionRequest(
                        messages=[{"role": "user", "content": prompt}],
                        model="",
                        max_tokens=output_length,
                        debug_config=DebugConfig(ignore_eos=True),
                    ),
                    metrics=Metrics(
                        success=False,
                        start_time=0,
                        finish_time=0,
                        end_to_end_latency_s=0,
                        input_tokens=input_length,
                    ),
                )
            )
        return request_records


class JSONModeEvalDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for JSON dataset."""

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        raw_dataset = load_dataset("NousResearch/json-mode-eval")
        self.tokenizer = tokenizer
        self.dataset = []
        for data in raw_dataset["train"]:
            messages = data["prompt"]
            schema = {
                "type": "json_object",
                "schema": data["schema"],
            }
            num_tokens = 0
            for message in messages:
                num_tokens += len(self.tokenizer.encode(message["content"]))
            self.dataset.append((messages, schema, num_tokens))

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        request_records = []
        for messages, schema, num_tokens in self.dataset:
            # If the request does not have enough length, discard it.
            if input_len is not None and num_tokens < input_len + 4 * input_len_std:
                continue

            if output_len is not None:
                output_length = max(
                    round(np.random.normal(loc=output_len, scale=output_len_std)), 1
                )
            else:
                output_length = None
            request_records.append(
                RequestRecord(
                    chat_cmpl=ChatCompletionRequest(
                        messages=[
                            ChatCompletionMessage(content=message["content"], role=message["role"])
                            for message in messages
                        ],
                        model="",
                        max_tokens=output_length,
                        response_format=schema,
                    ),
                    metrics=Metrics(
                        success=False,
                        start_time=0,
                        finish_time=0,
                        end_to_end_latency_s=0,
                        input_tokens=num_tokens,
                    ),
                )
            )
        return request_records


# Todo: dataset of log replay  # pylint: disable=fixme
# NOTE: moved from the previous "python/mlc_llm/bench/prompts.py"
# class PromptsGenerator:  # pylint: disable=too-few-public-methods
#     """
#     Generates prompts of a specified token length from a text file containing potential prompts.
#     """

#     def __init__(
#         self,
#         prompts_path: Optional[str] = None,
#         json_prompts_path: Optional[str] = None,
#         tokenizer: Optional[Any] = None,
#         seed: Optional[int] = 11111,
#     ) -> None:
#         """
#         Initializes the PromptsGenerator with the file path and tokenizer.

#         Parameters
#         ----------
#         prompts_path : Optional[str]
#             The path to the file containing the source prompts. This file can be
#             a plain text file, with each line representing a separate prompt str,
#             or a .jsonl file where each line is a JSON object formatted as
#             {"prompt": "prompt text", "input_tokens": 10}.

#         json_prompts_path : Optional[str]
#             The path to the file containing the source json prompts. This file a
#             .jsonl file where each line is a JSON object formatted as
#             {"messages": List[Dict[str, Any]], "response_format": Dict[str, Any]}.

#         tokenizer : Optional[Any]
#             The tokenizer object to use for tokenizing the prompts.

#         seed : Optional[int]
#             The seed for the random number generator.
#         """
#         random.seed(seed)
#         self.tokenizer = tokenizer
#         if not self.tokenizer:
#             from transformers import (  # pylint: disable=import-outside-toplevel,import-error
#                 LlamaTokenizerFast,
#             )

#             self.tokenizer = LlamaTokenizerFast.from_pretrained(
#                 "hf-internal-testing/llama-tokenizer"
#             )
#             logger.warning("No tokenizer provided. Using default tokenizer.")

#         self.prompts: List[Dict] = []
#         if prompts_path is not None and prompts_path.endswith(".jsonl"):
#             with open(prompts_path, "r", encoding="utf-8") as file:
#                 for line in file:
#                     json_line = json.loads(line)
#                     assert "prompt" in json_line, "The prompt field is required in the JSONL file"
#                     if "input_tokens" not in json_line:
#                         json_line["input_tokens"] = self._count_tokens(json_line["prompt"])
#                     self.prompts.append(json_line)
#         else:
#             if not prompts_path:
#                 prompts_path = Path(__file__).parent / "prompts.txt"  # type: ignore
#             with open(prompts_path, "r", encoding="utf-8") as file:
#                 prompt_line = file.readline()
#                 input_tokens = self._count_tokens(prompt_line)
#                 self.prompts.append({"prompt": prompt_line, "input_tokens": input_tokens})
#         if json_prompts_path:
#             self.json_prompts = defaultdict(list)
#             with open(json_prompts_path, "r", encoding="utf-8") as file:
#                 for line in file:
#                     json_line = json.loads(line)
#                     assert (
#                         "messages" in json_line
#                     ), "The messages field is required in the JSONL file."
#                     assert (
#                         "response_format" in json_line
#                     ), "The response_format field is required in the JSONL file."
#                     self.json_prompts[json.dumps(json_line["response_format"]["schema"])].append(
#                         json_line["messages"]
#                     )
#         else:
#             self.json_prompts = None

#     def _count_tokens(self, text: str) -> int:
#         """Get the number of tokens.

#         Parameters
#         ----------
#         text : str
#             The text to tokenize.

#         Returns
#         -------
#         output : int
#             The number of tokens
#         """
#         return len(self.tokenizer.encode(text))

#     def generate_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Generates a prompt based on the params, e.g. input_tokens, response_format.

#         Parameters
#         ----------
#         params : Dict[str, Any]
#             The desired mean number of tokens in the prompt.

#         Returns
#         -------
#         override_params: Dict[str, Any]
#             The params to override the original request, e.g. messages, response_format.
#         """
#         if "response_format" in params:
#             response_format = params["response_format"]
#             if response_format.get("type") == "json_object":
#                 if response_format.get("schema") in self.json_prompts:
#                     assert len(self.json_prompts[response_format["schema"]]) > 0
#                     return {"messages":
#                       random.choice(self.json_prompts[response_format["schema"]])}
#                 schema, prompts = random.choice(list(self.json_prompts.items()))
#                 response_format["schema"] = schema
#                 return {"messages": random.choice(prompts), "response_format": response_format}
#         tokens_mean = params.get("input_tokens", 128)
#         assert tokens_mean > 0, "The mean number of tokens must be greater than 0."
#         remaining_input_tokens = tokens_mean
#         result_prompt = ""
#         override_params = None
#         while remaining_input_tokens > 0:
#             prompt_dict = random.choice(self.prompts)
#             cur_input_tokens = prompt_dict["input_tokens"]
#             cur_prompt = prompt_dict["prompt"]
#             if override_params is None:
#                 override_params = prompt_dict["override_params"]
#             if remaining_input_tokens - cur_input_tokens < 0:
#                 result_prompt += cur_prompt[:remaining_input_tokens]
#                 remaining_input_tokens = 0
#                 break
#             result_prompt += cur_prompt
#             remaining_input_tokens -= cur_input_tokens
#         return {"messages": [{"role": "system", "content": result_prompt}]}


# def load_replay_log(log_path: str) -> List[Dict]:
#     """
#     Load replay log from file

#     Parameters
#     ----------
#     log_path : str
#         The path to the event log CSV or JSONL file containing the events to replay.

#     Returns
#     -------
#     res: List[Dict]
#         A list of preprocessed event data for replay.
#     """
#     if log_path.endswith(".csv"):
#         import pandas as pd  # pylint: disable=import-outside-toplevel,import-error

#         df = pd.read_csv(log_path)
#         column_names = df.columns.values
#         assert (
#             ("Date" in column_names)
#             and ("@request" in column_names)
#             and ("Message" in column_names)
#         )
#         df["timestamp"] = pd.to_datetime(df["Date"])
#         df.sort_values("timestamp", inplace=True)
#         # Get the request params from the loaded CSV
#         params = []
#         for _, row in df.iterrows():
#             request = row["@request"]
#             payload = json.loads(str(request))
#             params.append(
#                 {
#                     "timestamp": row["timestamp"],
#                     "payload": payload,
#                 }
#             )
#         return params
#     if log_path.endswith(".jsonl"):
#         with open(log_path, "r", encoding="utf-8") as file:
#             data = [json.loads(line) for line in file]
#             for row in data:
#                 row["timestamp"] = datetime.fromisoformat(str(row["timestamp"]))
#         return data
#     raise ValueError("Unsupported file format. Please use .csv or .jsonl.")

SUPPORTED_DATASET = [
    "sharegpt",
    "llmperf",
    "json-mode-eval",
]


def create_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> "Dataset":
    """Create a dataset instance with regard to the specified dataset kind and file path."""
    if args.dataset is None:
        # Auto-detect the dataset kind by looking into the dataset path.
        if "sharegpt" in args.dataset_path.lower():
            args.dataset = "sharegpt"
        else:
            raise ValueError(
                f"Unable to detect the dataset kind from dataset path {args.dataset_path}. "
                'Please specify the dataset kind via "--dataset".'
            )
    if args.dataset == "sharegpt":
        return ShareGPTDataset(args.dataset_path, tokenizer)
    if args.dataset == "llmperf":
        return LLMPerfDataset(args.dataset_path, args.num_requests * 4, tokenizer)
    if args.dataset == "json-mode-eval":
        return JSONModeEvalDataset(tokenizer)
    raise ValueError(f"Unrecognized dataset {args.dataset}")
