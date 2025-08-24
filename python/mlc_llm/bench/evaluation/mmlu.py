"""Eval MMLU with MLCEngine."""

import argparse
import asyncio
import csv
import json
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tqdm

from mlc_llm import AsyncMLCEngine

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
PADDING_LEN = max(len(subject) for subject in SUBJECTS)
DEVICES = ["cuda", "rocm", "metal", "vulkan"]
PROMPT_TEMPLATE = string.Template("$Q\nA. $A\nB. $B\nC. $C\nD. $D\nAnswer:")


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to MMLU test dataset home."
    )
    parser.add_argument("--device", type=str, choices=["auto"] + DEVICES, default="auto")
    parser.add_argument("--model-lib", type=str, default=None)
    parser.add_argument("-s", "--subject", nargs="+", type=str, choices=SUBJECTS, default=SUBJECTS)
    parser.add_argument("-bs", "--batch-size", type=int, default=16)
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args()


async def send_request(
    async_engine: AsyncMLCEngine,
    prompts: List[str],
    semaphore: asyncio.Semaphore,
    subject: str,
):
    """Send the calibration requests to the engine."""
    tasks = []

    async def generate_task(prompt):
        async with semaphore:
            return await async_engine.completions.create(
                prompt=prompt,
                stream=False,
                max_tokens=1,
                temperature=1.0,
                logprobs=True,
                top_logprobs=5,
            )

    for prompt in prompts:
        task = asyncio.create_task(generate_task(prompt))
        tasks.append(task)

    return await tqdm.asyncio.tqdm.gather(
        *tasks,
        desc=f"Running {subject.ljust(PADDING_LEN)}",
        bar_format="{desc} {percentage:3.0f}%|{bar}{r_bar}",
    )


async def evaluate(  # pylint: disable=too-many-arguments, too-many-locals
    model: str,
    device: str,
    dataset: Path,
    model_lib: Optional[str],
    subjects: List[str],
    semaphore: asyncio.Semaphore,
    log_dir: Optional[Path],  # pylint: disable=redefined-outer-name
):
    """Evaluate MMLU for the model."""
    async_engine = AsyncMLCEngine(model, device=device, model_lib=model_lib, mode="server")

    results: Dict[str, Any] = {}
    for subject in subjects:
        with open(dataset / "test" / f"{subject}_test.csv", encoding="utf-8") as csvfile:
            tests = list(csv.reader(csvfile, delimiter=",", quotechar='"'))
            assert all(len(test) == 6 for test in tests)

        logs = []
        num_correct = 0
        prompts = [
            PROMPT_TEMPLATE.substitute(Q=test[0], A=test[1], B=test[2], C=test[3], D=test[4])
            for test in tests
        ]
        responses = await send_request(async_engine, prompts, semaphore, subject)

        assert len(responses) == len(tests)
        for response, test in zip(responses, tests):
            token_logprobs = {}
            logprobs = response.choices[0].logprobs.content[0].top_logprobs
            for logprob in logprobs:
                if logprob.token not in token_logprobs:
                    token_logprobs[logprob.token] = logprob.logprob

            abcd_logprobs = {}
            for choice in ["A", "B", "C", "D"]:
                abcd_logprobs[choice] = token_logprobs[choice] if choice in token_logprobs else -100

            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[int(np.argmax(list(abcd_logprobs.values())))]
            num_correct += pred == test[5]

            logs.append(
                {
                    "Question": {
                        "Q": test[0],
                        "A": test[1],
                        "B": test[2],
                        "C": test[3],
                        "D": test[4],
                    },
                    "Answer": test[5],
                    "Response": {
                        "pred": pred,
                        "logprobs": list(abcd_logprobs.values()),
                    },
                }
            )

        results[subject] = {
            "correct": num_correct,
            "total": len(tests),
            "accuracy": num_correct / len(tests),
        }

        if log_dir:
            with open(log_dir / "subjects" / f"{subject}.json", "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)

    total_correct, total_tests = 0, 0
    for subject, v in results.items():
        num_correct, num_tests, accuracy = v["correct"], v["total"], v["accuracy"]
        print(f"{subject}: {num_correct} / {num_tests} = {accuracy * 100:.2f}%")
        total_correct += num_correct
        total_tests += num_tests

    total_accuracy = total_correct / total_tests
    results["total"] = {
        "correct": total_correct,
        "total": total_tests,
        "accuracy": total_accuracy,
    }
    print(f"Total accuracy: {total_correct} / {total_tests} = {total_accuracy * 100:.2f}%")

    if log_dir:
        results = {
            "config": {
                "model": model,
                "device": device,
                "model_lib": model_lib,
                "subjects": subjects,
            },
            "results": results,
        }
        with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    log_dir: Optional[Path] = None
    if args.log_dir is not None:
        time_dir = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = args.log_dir / time_dir
        (log_dir / "subjects").mkdir(parents=True, exist_ok=True)
    asyncio.run(
        evaluate(
            model=args.model,
            device=args.device,
            dataset=args.dataset,
            model_lib=args.model_lib,
            subjects=args.subject,
            semaphore=asyncio.Semaphore(args.batch_size),
            log_dir=log_dir,
        )
    )
    end_time = datetime.now()
    print(f"Time used: {end_time - start_time}")
