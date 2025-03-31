"""MLC LLM benchmark dataset classes"""

import argparse
import json
import os
import requests
import random
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # pylint: disable=import-error
from datasets import load_dataset  # pylint: disable=import-error
from transformers import AutoTokenizer  # pylint: disable=import-error

from request_record import GroupedRequestRecord, Metrics, RequestRecord
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatToolCall,
    DebugConfig,
)


class Dataset:  # pylint: disable=too-few-public-methods
    """The dataset base class."""

    # We set a truncation limit of 100k.
    truncate_length = int(1e5)
    # For some that datasets (e.g., dataset that has shared common prefix),
    # we need fake warmup requests to avoid prefilling common prefixes to the engine.
    require_fake_warmup: bool = False
    # Whether the dataset contains timestamps already.
    # If the dataset comes with timestamps, the benchmark can just replay
    # the requests according to their timestamps.
    timestamp_available: bool = False

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        """Get the raw unprocessed request records of the dataset."""
        raise NotImplementedError()
    
GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

class GorillaDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for Gorilla dataset.
    Reference: https://github.com/ShishirPatil/gorilla
    """

    def __init__(self, dataset_path: str, tokenizer: AutoTokenizer, use_stag: bool) -> None:
        self.tokenizer = tokenizer
        self.require_fake_warmup = True
        self.gorilla_data = []
        file_patterns = [
            "BFCL_v3_simple.json",
        ]
        base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data"
        
        for filename in file_patterns:
            id = 0
            dataset_file = f"{dataset_path}/{filename}"
            if os.path.exists(dataset_file):
                with open(dataset_file, mode="r", encoding="utf-8") as file:
                    self.gorilla_data = json.load(file)
            else:
                function_url = f"{base_url}/{filename}"
                answer_url = f"{base_url}/possible_answer/{filename}"
                print(f"Downloading {filename} from GitHub...")
                functions_data = []
                answers_data = []
                try:
                    function_response = requests.get(function_url)
                    function_response.raise_for_status()
                    function_text = function_response.text
                    for line in function_text.strip().split("\n"):
                        if line.strip():
                            try:
                                functions_data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Error parsing function line in {filename}: {e}")
                    answer_response = requests.get(answer_url)
                    answer_response.raise_for_status()
                    answer_text = answer_response.text
                    for line in answer_text.strip().split("\n"):
                        if line.strip():
                            try:
                                answers_data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Error parsing answer line in {filename}: {e}")
                    print(
                        f"Successfully downloaded {filename}: {len(functions_data)} functions, {len(answers_data)} answers"
                    )
                except requests.RequestException as e:
                    print(f"Error downloading {filename}: {e}")
                    functions_data = []
                    answers_data = []
                if not functions_data or not answers_data:
                    print(f"Skipping {filename} - failed to download data")
                    continue
                print(f"Processing {filename}...")
                answers_by_id = {item["id"]: item for item in answers_data}
                for item in functions_data:
                    item_id = item["id"]
                    question = item["question"][0]
                    if item_id not in answers_by_id:
                        print(f"Warning: No answer found for item {item_id}")
                        continue
                    if "function" not in item or not item["function"]:
                        print(f"Warning: No function definition for item {item_id}")
                        continue
                    tool = [{"type": "function", "function": func} for func in item["function"]]
                    self.map_type_values(tool)                    
                    answer = answers_by_id[item_id]
                    if "ground_truth" not in answer or not answer["ground_truth"]:
                        print(f"Warning: No ground truth for item {item_id}")
                        continue
                    ideal_call = []
                    for ground_truth in answer["ground_truth"]:
                        function_name = list(ground_truth.keys())[0]
                        params = ground_truth[function_name]
                        ideal_call.append({"name": function_name, "arguments": params})
                    self.gorilla_data.append(
                        {
                            "id": id,
                            "question": question,
                            "tool": tool,
                            "ideal_call": ideal_call,
                            "source": filename,
                        }
                    )
                    id += 1
                with open(dataset_file, mode="w", encoding="utf-8") as file:
                    json.dump(self.gorilla_data, file, ensure_ascii=False, indent=4)
        if self.tokenizer is not None:
            for item in self.gorilla_data:
                num_tokens = 0
                for message in item["question"]:
                    num_tokens += len(
                        tokenizer.encode(message["content"], add_special_tokens=False)
                    )
                item["num_tokens"] = num_tokens
        if not use_stag:
            for item in self.gorilla_data:
                for tool in item["tool"]:
                    tool["function"]["strict"] = False

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        
        request_records = []
        for entry in self.gorilla_data:
            # If the request does not have enough length, discard it.
            # if input_len is not None and entry["num_tokens"] < input_len + 4 * input_len_std:
            #     continue

            if output_len is not None:
                output_length = max(
                    round(np.random.normal(loc=output_len, scale=output_len_std)), 1
                )
            else:
                output_length = 256
            request_records.append(
                RequestRecord(
                    request_id=entry["id"],
                    chat_cmpl=ChatCompletionRequest(
                        messages=[
                            ChatCompletionMessage(content=message["content"], role=message["role"])
                            for message in entry["question"]
                        ],
                        model="",
                        max_tokens=output_length,
                        tools=entry["tool"],
                    ),
                    metrics=Metrics(
                        success=False,
                        start_time=0,
                        finish_time=0,
                        end_to_end_latency_s=0,
                        input_tokens=entry["num_tokens"],
                    ),
                )
            )
        return request_records

    # Modified by https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/eval_checker/ast_eval/ast_checker.py
    def check_simple(self, tool_call: Dict[str, Any],
                     tool: Dict[str, Any], ideal: Dict[str, Any]) -> Tuple[bool, bool]:
        # check func name
        if ideal["name"] != tool_call["function"]["name"]:
            return True, False
        func = tool["function"]
        # check func args
        for arg in func["parameters"]["required"]:
            if arg not in tool_call["function"]["arguments"]:
                return True, False
        for arg in tool_call["function"]["arguments"].keys():
            ideal_arg: List = ideal["arguments"][arg] if arg in ideal["arguments"] else None
            real_arg = tool_call["function"]["arguments"][arg]
            if arg not in func["parameters"]["properties"]:
                return True, False
            info_arg = func["parameters"]["properties"][arg]
            if info_arg["type"] == "integer":
                if not self.check_integer(real_arg, ideal_arg):
                    return True, False
            elif info_arg["type"] == "number":
                if not self.check_number(real_arg, ideal_arg):
                    return True, False
            elif info_arg["type"] == "boolean":
                if not self.check_boolean(real_arg, ideal_arg):
                    return True, False
            elif info_arg["type"] == "string":
                enum = info_arg["enum"] if "enum" in info_arg else None
                if not self.check_string(real_arg, ideal_arg, enum):
                    return True, False
            elif info_arg["type"] == "array":
                if not self.check_list(real_arg, ideal_arg, info_arg["items"]):
                    return True, False
            elif info_arg["type"] == "dict":
                if not self.check_dict(real_arg, ideal_arg, info_arg["properties"]):
                    return True, False
        return True, True
            
                
                
    def check_integer(self, real_arg: Any, ideal_arg: Optional[List[Any]]) -> bool:
        try:
            if type(real_arg) != int:
                return False
            if ideal_arg is None:
                return True
            match = False
            for ideal in ideal_arg:
                if real_arg == ideal:
                    match = True
                    break
            return match
        except:
            return False
    
    def check_number(self, real_arg: Any, ideal_arg: Optional[List[Any]]) -> bool:
        if type(real_arg) != float and type(real_arg) != int:
            return False
        if ideal_arg is None:
            return True
        match = False
        for ideal in ideal_arg:
            if real_arg == ideal:
                match = True
                break
        return match
    
    def check_string(self, real_arg: Any, ideal_arg: Optional[List[Any]], enum: Optional[List[str]]) -> bool:
        
        def standardize_string(string: Any) -> str:
            if not isinstance(string, str):
                return "Error><><><><><>"
            regex_string = r"[ \,\.\/\-\_\*\^]"
            return re.sub(regex_string, "", string).lower().replace("'", '"')
        
        if type(real_arg) != str:
            return False
        match = False
        real_arg = standardize_string(real_arg)
        if ideal_arg is None:
            if enum is None:
                return True
            else:
                for ideal in enum:
                    if real_arg == standardize_string(ideal):
                        match = True
                        break
        else:
            for ideal in ideal_arg:
                if real_arg == standardize_string(ideal):
                    match = True
                    break
        return match
    
    def check_boolean(self, real_arg: bool, ideal_arg: Optional[List[bool]]) -> bool:
        if type(real_arg) != bool:
            return False
        if ideal_arg is None:
            return True
        match = False
        for ideal in ideal_arg:
            if real_arg == ideal:
                match = True
                break
        return match
                
    def check_list(self, real_arg: List, ideal_arg: Optional[List[List]], item: Dict[str, Any]) -> bool:
        if type(real_arg) != list:
            return False
        item_type = item["type"]
        if ideal_arg is None:
            if item_type == "integer":
                for i, integer in enumerate(real_arg):
                    if not self.check_integer(integer, None):
                        return False
            elif item_type == "number":
                for i, integer in enumerate(real_arg):
                    if not self.check_number(integer, None):
                        return False
            elif item_type == "boolean":
                for i, boolean in enumerate(real_arg):
                    if not self.check_boolean(boolean, None):
                        return False
            elif item_type == "string":
                for i, string in enumerate(real_arg):
                    enum = item["enum"] if "enum" in item else None
                    if not self.check_string(string, None, enum):
                        return False
            elif item_type == "array":
                for i, array in enumerate(real_arg):
                    if not self.check_list(array, None, item["items"]):
                        return False
            elif item_type == "dict":
                for i, dictionary in enumerate(real_arg):
                    if not self.check_dict(dictionary, None, item["properties"]):
                        return False
            return True
        else:
            for ideal in ideal_arg:
                if len(ideal) != len(real_arg):
                    continue
                match = True
                if item_type == "integer":
                    for i, integer in enumerate(real_arg):
                        if not self.check_integer(integer, [ideal[i]]):
                            match = False
                            break
                elif item_type == "number":
                    for i, integer in enumerate(real_arg):
                        if not self.check_number(integer, [ideal[i]]):
                            match = False
                            break
                elif item_type == "boolean":
                    for i, boolean in enumerate(real_arg):
                        if not self.check_boolean(boolean, [ideal[i]]):
                            match = False
                            break
                elif item_type == "string":
                    for i, string in enumerate(real_arg):
                        enum = item["enum"] if "enum" in item else None
                        if not self.check_string(string, [ideal[i]], enum):
                            match = False
                            break
                elif item_type == "array":
                    for i, array in enumerate(real_arg):
                        if not self.check_list(array, [ideal[i]], item["items"]):
                            match = False
                            break
                elif item_type == "dict":
                    for i, dictionary in enumerate(real_arg):
                        if not self.check_dict(dictionary, [ideal[i]], item["properties"]):
                            match = False
                            break
                if match:
                    return True
            return False
    
    def check_dict(self, real_arg: Dict[str, Any], ideal_arg: Optional[Dict[str, Any]], properties: Dict[str, Any]) -> bool:
        if type(real_arg) != dict:
            return False
        if ideal_arg is None:
            for key in properties.keys():
                if key not in real_arg:
                    return False
                item_type = properties[key]["type"]
                if item_type == "integer":
                    if not self.check_integer(real_arg[key], None):
                        return False
                elif item_type == "number":
                    if not self.check_number(real_arg[key], None):
                        return False
                elif item_type == "boolean":
                    if not self.check_boolean(real_arg[key], None):
                        return False
                elif item_type == "string":
                    enum = properties[key]["enum"] if "enum" in properties[key] else None
                    if not self.check_string(real_arg[key], None, enum):
                        return False
                elif item_type == "array":
                    if not self.check_list(real_arg[key], None, properties[key]["items"]):
                        return False
                elif item_type == "dict":
                    if not self.check_dict(real_arg[key], None, properties[key]["properties"]):
                        return False
            return True
        else:
            for ideal in ideal_arg:
                match = True
                for key in properties.keys():
                    if key not in real_arg:
                        match = False
                        break
                    item_type = properties[key]["type"]
                    if item_type == "integer":
                        if not self.check_integer(real_arg[key], [ideal[key]]):
                            match = False
                            break
                    elif item_type == "number":
                        if not self.check_number(real_arg[key], [ideal[key]]):
                            match = False
                            break
                    elif item_type == "boolean":
                        if not self.check_boolean(real_arg[key], [ideal[key]]):
                            match = False
                            break
                    elif item_type == "string":
                        enum = properties[key]["enum"] if "enum" in properties[key] else None
                        if not self.check_string(real_arg[key], [ideal[key]], enum):
                            match = False
                            break
                    elif item_type == "array":
                        if not self.check_list(real_arg[key], [ideal[key]], properties[key]["items"]):
                            match = False
                            break
                    elif item_type == "dict":
                        if not self.check_dict(real_arg[key], [ideal[key]], properties[key]["properties"]):
                            match = False
                            break
                if match:
                    return True
            return False

    def map_type_values(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self.map_type_values(value)
                elif key == "type" and value in GORILLA_TO_OPENAPI:
                    data[key] = GORILLA_TO_OPENAPI[value]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self.map_type_values(item)
                


SUPPORTED_DATASET = [
    "gorilla"
]


def create_dataset(  # pylint: disable=too-many-return-statements,too-many-branches
    args: argparse.Namespace, tokenizer: AutoTokenizer
) -> Dataset:
    """Create a dataset instance with regard to the specified dataset kind and file path."""
    if args.dataset_path is not None and not isinstance(args.dataset_path, str):
        raise TypeError(f"Invalid dataset path {args.dataset_path}. Please use a string.")
    if args.dataset == "gorilla":
        if args.dataset_path is None:
            raise ValueError(
                "Gorilla dataset requires dataset path. "
                'Please specify it with "--dataset-path".'
            )
        assert (
            args.apply_chat_template is False
        ), "Gorilla dataset does not support applying chat template"
        return GorillaDataset(args.dataset_path, tokenizer)
    raise ValueError(f"Unrecognized dataset {args.dataset}")
