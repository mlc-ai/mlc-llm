"""Python entrypoint for calibration."""

import asyncio
import json
import random
from typing import List, Mapping, Optional, Tuple

import numpy as np
import tqdm.asyncio
import tvm
from tvm.contrib import tvmjs

from mlc_llm.serve.engine import AsyncMLCEngine, EngineConfig
from mlc_llm.tokenizers import Tokenizer


class CalibrationObserver:
    """A singleton class to observe the calibration parameters.""" ""

    instance: "CalibrationObserver" = None

    params: Mapping[str, tvm.nd.NDArray] = {}

    @staticmethod
    def get():
        """Get the singleton instance of the class.""" ""
        if CalibrationObserver.instance is None:
            CalibrationObserver.instance = CalibrationObserver()
        return CalibrationObserver.instance

    @tvm.register_func("mlc_llm.calibration_observer")
    @staticmethod
    def callback(name: str, mode: str, value: "tvm.nd.NDArray", out_value: "tvm.nd.NDArray"):
        """The callback function to update the saved calibration parameters."""
        instance = CalibrationObserver.get()
        if mode == "max":
            reducer = np.maximum
        else:
            raise NotImplementedError(f"Unsupported calibration mode: {mode}")
        if name in instance.params:
            instance.params[name] = reducer(instance.params[name], value.numpy())
        else:
            instance.params[name] = value.numpy()
        out_value.copyfrom(instance.params[name])

    def save_params(self, output: str):
        """Save the calibration parameters to the given output directory."""
        tvmjs.dump_ndarray_cache(
            self.params,
            output,
            encode_format="f32-to-bf16",
            meta_data=None,
            show_progress=False,
            update_if_exists=True,
        )


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: Tokenizer,
) -> List[Tuple[str, int, int]]:
    """Sample the requests from the given dataset."""
    # pylint: disable=too-many-locals
    # Load the dataset.
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset
    ]
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer.encode_batch(prompts)
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer.encode_batch(completions)
    tokenized_dataset: List[Tuple[str, List[int], int]] = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, token_ids, output_len in tokenized_dataset:
        prompt_len = len(token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def send_calibration_requests(
    async_engine: AsyncMLCEngine,
    sampled_requests: List[Tuple[str, int, int]],
    max_concurrent_requests: int,
) -> None:
    """Send the calibration requests to the engine."""
    tasks = []

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def generate_task(request_idx):
        async with semaphore:
            prompt, _, output_len = sampled_requests[request_idx]
            await async_engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_len,
                request_id=str(request_idx),
            )

    for i in range(len(sampled_requests)):
        task = asyncio.create_task(generate_task(i))
        tasks.append(task)
    await tqdm.asyncio.tqdm.gather(*tasks)


def calibrate(
    model: str,
    device: str,
    model_lib: Optional[str],
    dataset: str,
    output: str,
    num_calibration_samples: int,
    *,
    seed: int,
    max_num_sequence: Optional[int] = None,
    max_total_sequence_length: Optional[int] = None,
    prefill_chunk_size: Optional[int] = None,
    max_history_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
) -> None:
    """Calibrate the quantized model using the given dataset."""
    # pylint: disable=too-many-arguments, too-many-locals
    random.seed(seed)
    async_engine = AsyncMLCEngine(
        model=model,
        device=device,
        model_lib=model_lib,
        mode="server",
        engine_config=EngineConfig(
            max_num_sequence=max_history_size,
            max_total_sequence_length=max_total_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
            max_history_size=max_history_size,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
    )
    sampled_requests = sample_requests(dataset, num_calibration_samples, async_engine.tokenizer)
    asyncio.run(
        send_calibration_requests(
            async_engine, sampled_requests, max_concurrent_requests=max_num_sequence or 32
        )
    )
    async_engine.terminate()

    calibrator = CalibrationObserver.get()
    calibrator.save_params(output)
