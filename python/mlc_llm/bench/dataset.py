"""MLC LLM benchmark dataset classes"""

import argparse
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset  # pylint: disable=import-error
from transformers import AutoTokenizer  # pylint: disable=import-error

from mlc_llm.bench.request_record import GroupedRequestRecord, Metrics, RequestRecord
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    DebugConfig,
)


class Dataset:  # pylint: disable=too-few-public-methods
    """The dataset base class."""

    # We set a truncation limit of 100k.
    truncate_length = int(1e5)
    # For some that datasets (e.g., dataset that has shared common prefix),
    # we need fake warmup requests to avoid prefilling common prefixes to the engine.
    require_fake_warmup: bool = False

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
    apply_chat_template: bool

    def __init__(
        self, dataset_path: str, tokenizer: AutoTokenizer, apply_chat_template: bool
    ) -> None:
        self.apply_chat_template = apply_chat_template
        with open(dataset_path, encoding="utf-8") as f:
            raw_dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        _dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in raw_dataset
            if len(data["conversations"]) >= 2 and data["conversations"][0]["from"] == "human"
        ]
        # Tokenize the prompts and completions.
        self.tokenizer = tokenizer
        prompts = [prompt for prompt, _ in _dataset]
        if apply_chat_template:
            assert (
                getattr(tokenizer, "chat_template", None) is not None
            ), '"--apply-chat-template" is set but the tokenizer does not have chat template.'
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for prompt in prompts
            ]

        prompt_token_ids = list(
            tokenizer(
                prompts,
                truncation=True,
                max_length=min(tokenizer.model_max_length, self.truncate_length),
                add_special_tokens=False,
            ).input_ids
        )
        completions = [completion for _, completion in _dataset]
        completion_token_ids = tokenizer(
            completions,
            truncation=True,
            max_length=min(tokenizer.model_max_length, self.truncate_length),
            add_special_tokens=False,
        ).input_ids
        self._tokenized_dataset: List[Tuple[str, List[int], int]] = []
        for i in range(len(_dataset)):
            if (
                len(prompt_token_ids[i]) < 4
                or len(completion_token_ids[i]) < 4
                or len(prompt_token_ids[i]) + len(completion_token_ids[i])
                >= min(tokenizer.model_max_length, 8192)
            ):
                # Filter out sequences that are too short or too long
                continue
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
        if self.apply_chat_template:
            assert (
                input_len is None
            ), '"--apply-chat-template" is not supported when "--input-len" is specified.'

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


class LoogleDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for Loogle dataset."""

    # pylint: disable=line-too-long
    task2prompt = {
        "shortdep_qa": "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: ",
        "longdep_qa": "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: ",
        "longdep_summarization": "Please generate a summary of the below paper. \n{input}\n Summarization: ",
        "shortdep_cloze": "Please fill in the clozes based on the given long texts below. Each of the placeholder '<mask-n>' in the question could be an entity of Person, Location or Organiocation. The same masks represent the same entity. Output a json format answer, for example: {{'<mask-0>': 'Bob', '<mask-1>': 'Gorrosion Magazine','<mask-2>': 'Bethel Horizon'}}\n{input}\n Question: {Q} What are the masked entities? \nAnswer:",
    }
    # pylint: enable=line-too-long
    require_fake_warmup: bool = True

    def __init__(self, tokenizer: AutoTokenizer, testset_name: str) -> None:
        raw_dataset = load_dataset("bigainlco/LooGLE", testset_name, split="test")
        self.tokenizer = tokenizer
        self.dataset = []
        self.prompt_format = self.task2prompt[testset_name]
        prompts = []
        generate_lens = []
        questions = []
        for data in raw_dataset:
            prompt = data["input"]
            prompts.append(prompt)
            qa_pairs = eval(data["qa_pairs"])  # pylint: disable=eval-used
            questions.append([j["Q"] for j in qa_pairs])
            generate_lens.append(
                [len(tokenizer.encode(j["A"], add_special_tokens=False)) for j in qa_pairs]
            )
        prompt_token_ids = tokenizer(
            prompts,
            truncation=True,
            max_length=min(tokenizer.model_max_length, self.truncate_length),
            add_special_tokens=False,
        ).input_ids
        for prompt, prompt_token_id, question, generate_len in zip(
            prompts, prompt_token_ids, questions, generate_lens
        ):
            self.dataset.append((prompt, prompt_token_id, question, generate_len))

    def generate_request_records(  # pylint: disable=too-many-locals
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        request_records = []
        for prompt, input_token_ids, questions, generate_lens in self.dataset:
            input_length = round(float(np.random.normal(loc=input_len, scale=input_len_std)))
            if len(input_token_ids) > input_length:
                input_token_ids = input_token_ids[:input_length]
                prompt = self.tokenizer.decode(input_token_ids)
            grouped_request_records = []
            for question, generate_len in zip(questions, generate_lens):
                json_obj = {"input": prompt, "Q": question}
                full_prompt = self.prompt_format.format(**json_obj)

                output_length = (
                    round(float(np.random.normal(loc=output_len, scale=output_len_std, size=1)[0]))
                    if output_len is not None
                    else generate_len
                )
                grouped_request_records.append(
                    RequestRecord(
                        chat_cmpl=ChatCompletionRequest(
                            messages=[
                                {
                                    "role": "user",
                                    "content": full_prompt,
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
            request_records.append(
                GroupedRequestRecord(
                    # Create a dummy ChatCompletionRequest.
                    chat_cmpl=ChatCompletionRequest(messages=[]),
                    records=grouped_request_records,
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
            max_length=min(tokenizer.model_max_length, self.truncate_length),
            add_special_tokens=False,
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

            remaining_token_length = input_length - len(
                self.tokenizer.encode(prompt, add_special_tokens=False)
            )

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
                num_tokens += len(
                    self.tokenizer.encode(message["content"], add_special_tokens=False)
                )
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


class ReActDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for replaying a given ReAct trace for benchmark purpose.
    It is not an actual ReAct agent implementation.
    """

    _dataset: List[List[Tuple[str, int, int]]]
    require_fake_warmup: bool = True
    # pylint: disable=line-too-long
    prefix: str = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action 3: Finish[yes]
"""

    # pylint: enable=line-too-long
    def __init__(  # pylint: disable=too-many-locals
        self, dataset_path: str, tokenizer: AutoTokenizer
    ) -> None:
        raw_entries: List[Dict] = []
        with open(dataset_path) as fin:  # pylint: disable=unspecified-encoding
            for line in fin:
                line_content = json.loads(line)
                raw_entries += list({"question": k, "triplets": v} for k, v in line_content.items())

        self._dataset = []
        max_rounds = 0
        for raw_entry in raw_entries:
            processed_entry = []
            question = raw_entry["question"]
            triplets = raw_entry["triplets"]
            seq = self.prefix + question
            max_rounds = max(max_rounds, len(triplets) + 1)
            output_lengths: List[int] = []
            for i, triplet in enumerate(triplets):
                output_lengths.append(
                    len(
                        tokenizer(
                            triplet["thought"]
                            + "\nAction "
                            + str(i + 1)
                            + ": "
                            + triplet["action"]
                            + "\n",
                            truncation=True,
                            max_length=min(tokenizer.model_max_length, self.truncate_length),
                            add_special_tokens=False,
                        ).input_ids
                    )
                )

            for i in range(1, len(triplets) + 2):
                seq += "Thought " + str(i) + ":"
                input_len = len(
                    tokenizer(
                        seq,
                        truncation=True,
                        max_length=min(tokenizer.model_max_length, self.truncate_length),
                        add_special_tokens=False,
                    ).input_ids
                )
                output_length = (
                    output_lengths[i - 1]
                    if i <= len(triplets)
                    else int(sum(output_lengths) / len(triplets))
                )
                processed_entry.append((seq, input_len, output_length))
                if i != len(triplets) + 1:
                    seq += (
                        triplets[i - 1]["thought"]
                        + "\nAction "
                        + str(i)
                        + ": "
                        + triplets[i - 1]["action"]
                        + "\nObservation "
                        + str(i)
                        + ": "
                        + triplets[i - 1]["observation"]
                        + "\n"
                    )
            self._dataset.append(processed_entry)

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        if input_len is not None or output_len is not None:
            raise ValueError("ReAct dataset does not support specifying input/output length.")

        request_records = []
        for processed_entries in self._dataset:
            grouped_request_records = []
            for prompt, input_length, output_length in processed_entries:
                grouped_request_records.append(
                    RequestRecord(
                        chat_cmpl=ChatCompletionRequest(
                            messages=[{"role": "user", "content": prompt}],
                            model="",
                            max_tokens=output_length,
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
            request_records.append(
                GroupedRequestRecord(
                    # Create a dummy ChatCompletionRequest.
                    chat_cmpl=ChatCompletionRequest(messages=[]),
                    records=grouped_request_records,
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
    "loogle",
    "react",
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
        return ShareGPTDataset(args.dataset_path, tokenizer, args.apply_chat_template)
    if args.dataset == "llmperf":
        assert (
            args.apply_chat_template is False
        ), "LLMPerf dataset does not support applying chat template"
        return LLMPerfDataset(
            args.dataset_path, (args.num_requests + args.num_warmup_requests) * 4, tokenizer
        )
    if args.dataset == "json-mode-eval":
        assert (
            args.apply_chat_template is False
        ), "JSON mode evaluation does not support applying chat template"
        return JSONModeEvalDataset(tokenizer)
    if args.dataset == "loogle":
        assert (
            args.apply_chat_template is False
        ), "Loogle dataset does not support applying chat template"
        return LoogleDataset(tokenizer, testset_name=args.dataset_path)
    if args.dataset == "react":
        assert (
            args.apply_chat_template is False
        ), "ReAct dataset does not support applying chat template"
        return ReActDataset(args.dataset_path, tokenizer)
    raise ValueError(f"Unrecognized dataset {args.dataset}")
