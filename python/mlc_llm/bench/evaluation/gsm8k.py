"""Eval GSM8K with MLCEngine."""

import argparse
import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import tqdm

from mlc_llm import AsyncMLCEngine

DEVICES = ["cuda", "rocm", "metal", "vulkan"]
ANSWER_TRIGGER = "The answer is"
INVALID_ANS = "[invalid]"


def extract_answer(text: str, regex: re.Pattern, select_index: int) -> str:
    """Extract the answer from the text."""
    match_all = regex.findall(text)
    if len(match_all) == 0:
        return INVALID_ANS
    match = match_all[select_index]
    if isinstance(match, tuple):
        match = [m for m in match if m][0]
    match_str: str = match.strip()
    match_str = match_str.lstrip("$").rstrip(".").replace(",", "")
    return match_str


def extract_ground_truth(text: str) -> str:
    """Extract the ground truth from the text."""
    return extract_answer(text, re.compile(r"#### (\-?[0-9\.\,]+)"), 0)


def strict_extract_answer(text: str) -> str:
    """Strictly extract the answer from the text."""
    return extract_answer(text, re.compile(r"The answer is \$?(\-?[0-9\.\,]+)."), 0)


def flexible_extract_answer(text: str) -> str:
    """Extract the last number from the text."""
    return extract_answer(text, re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)"), -1)


def create_few_shot_prompt(n_shot: int, use_cot: bool, random_order=False) -> str:
    """
    Create a prompt for the few-shot learning task.

    Note
    ----
    The examples are taken from the paper https://arxiv.org/pdf/2201.11903.pdf page 35.
    """
    question, chain, answer = [], [], []

    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    index_list = list(range(len(question)))
    if random_order:
        random.shuffle(index_list)

    prompt = ""
    for i in index_list[:n_shot]:
        if use_cot:
            prompt += f"Q: {question[i]}\nA: {chain[i]} {ANSWER_TRIGGER} {answer[i]}.\n\n"
        else:
            prompt += f"Question: {question[i]}\nAnswer: {ANSWER_TRIGGER} {answer[i]}.\n\n"
    return prompt


def create_prompt(question: str, n_shot: int, use_cot: bool, random_order: bool = False) -> str:
    """Create a prompt for the few-shot learning task."""
    prompt = create_few_shot_prompt(n_shot, use_cot, random_order)
    if use_cot:
        prompt += f"Q: {question}\nA:"
    else:
        prompt += f"Question: {question}\nAnswer:"
    return prompt


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to GSM8K test dataset home."
    )
    parser.add_argument("--device", type=str, choices=["auto"] + DEVICES, default="auto")
    parser.add_argument("--model-lib", type=str, default=None)
    parser.add_argument("--n-shot", type=int, default=8)
    parser.add_argument("--disable_cot", action="store_true", default=False)
    parser.add_argument("-bs", "--batch-size", type=int, default=16)
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args()


async def send_request(
    async_engine: AsyncMLCEngine,
    prompts: List[str],
    semaphore: asyncio.Semaphore,
):
    """Send the calibration requests to the engine."""
    tasks = []

    async def generate_task(prompt):
        async with semaphore:
            return await async_engine.completions.create(
                prompt=prompt,
                stream=False,
                max_tokens=512,
                stop=["Q:", "Question:"],
                temperature=0.0,
            )

    for prompt in prompts:
        task = asyncio.create_task(generate_task(prompt))
        tasks.append(task)

    return await tqdm.asyncio.tqdm.gather(*tasks)


async def evaluate(  # pylint: disable=too-many-arguments, too-many-locals
    model: str,
    device: str,
    dataset: Path,
    model_lib: Optional[str],
    n_shot: int,
    use_cot: bool,
    batch_size: int,
    log_dir: Optional[Path],  # pylint: disable=redefined-outer-name
):
    """Evaluate GSM8K for the model."""
    mode: Literal["local", "interactive", "server"] = (
        "server" if batch_size > 4 else "interactive" if batch_size == 1 else "local"
    )
    async_engine = AsyncMLCEngine(model, device=device, model_lib=model_lib, mode=mode)

    with open(dataset / "test.jsonl", "r", encoding="utf-8") as file:
        tests = [json.loads(line) for line in file]

    prompts = [create_prompt(test["question"], n_shot, use_cot) for test in tests]
    responses = await send_request(async_engine, prompts, asyncio.Semaphore(batch_size))
    assert len(responses) == len(tests)

    num_strict_correct, num_flexible_correct = 0, 0
    num_tests = len(tests)
    logs = []

    for response, test in zip(responses, tests):
        response_text = response.choices[0].text.strip()
        gt_answer = extract_ground_truth(test["answer"])
        assert gt_answer != INVALID_ANS
        strict_answer = strict_extract_answer(response_text)
        flexible_answer = flexible_extract_answer(response_text)

        if gt_answer == strict_extract_answer(response_text):
            # If the answer is exactly the same as the response, then it is correct
            num_strict_correct += 1
            num_flexible_correct += 1

        elif gt_answer == flexible_extract_answer(response_text):
            # Try flexible extract if the strict match fails
            num_flexible_correct += 1

        logs.append(
            {
                "question": test["question"],
                "response": response_text,
                "ground_truth": gt_answer,
                "strict_answer": strict_answer,
                "flexible_answer": flexible_answer,
                "strict_match": gt_answer == strict_answer,
                "flexible_match": gt_answer == flexible_answer,
            }
        )

    results = {
        "config": {
            "model": model,
            "device": device,
            "model_lib": model_lib,
            "n_shot": n_shot,
            "use_cot": use_cot,
        },
        "results": {
            "strict_match": num_strict_correct,
            "flexible_match": num_flexible_correct,
            "total": num_tests,
        },
    }
    print(
        f"Strict Matching Accuracy: {num_strict_correct} / {num_tests} = "
        f"{num_strict_correct /num_tests * 100:.2f}%"
    )
    print(
        f"Flexible Matching Accuracy: {num_flexible_correct} / {num_tests} = "
        f"{num_flexible_correct /num_tests * 100:.2f}%"
    )

    if log_dir:
        with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(log_dir / "logs.json", "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    log_dir: Optional[Path] = None
    if args.log_dir is not None:
        time_dir = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = args.log_dir / time_dir
        log_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        evaluate(
            model=args.model,
            device=args.device,
            dataset=args.dataset,
            model_lib=args.model_lib,
            n_shot=args.n_shot,
            use_cot=not args.disable_cot,
            batch_size=args.batch_size,
            log_dir=log_dir,
        )
    )
    end_time = datetime.now()
    print(f"Time used: {end_time - start_time}")
