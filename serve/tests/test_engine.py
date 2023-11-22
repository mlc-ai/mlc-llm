import torch

import argparse
import json
import random
import os

from mlc_llm import utils
from mlc_serve.engine import (
    Request,
    ChatMessage,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
    get_engine_config
)
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule


def test(args: argparse.Namespace):
    # Examples. "--max-output-len" can be used to specify the number of output tokens.
    #
    # Profile the gpu memory usage, and use the maximum number of cache blocks possible:
    # python serve/tests/test_engine_paged_cache_model.py --local-id vicuna-v1-7b-q4f16_ft --max-num-batched-tokens 2560 --max-input-len 256
    #
    # Mistral:
    # python serve/tests/test_engine_paged_cache_model.py  --local-id Mistral-7B-v0.1-q0f16 --long-prompt --max-num-batched-tokens 24000 --max-input-len 8000 --max-output-len 20
    #
    # Disco:
    # python serve/tests/test_engine_paged_cache_model.py --local-id vicuna-v1-7b-q0f16-presharded-gpu2

    engine_config = get_engine_config({
        "use_staging_engine": args.use_staging_engine,
        "max_num_sequences": args.max_num_sequences,
        "max_input_len": args.max_input_len,
        "min_decode_steps": args.min_decode_steps,
        "max_decode_steps": args.max_decode_steps,
        "prompt_allocate_ratio": args.prompt_allocate_ratio
    })

    if args.use_staging_engine:
        engine = StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(args.model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": args.model_artifact_path,
                "engine_config": engine_config,
            },
        )
        engine.start()
    else:
        engine = SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path = args.model_artifact_path,
                engine_config = engine_config,
        ))

    sampling_params_greedy = SamplingParams(
        temperature=0.0,
    )

    if args.use_random_sampling:
        sampling_params_random = SamplingParams(
            temperature=0.9,
            top_p=1.0,
        )
        sampling_params_choices = [sampling_params_random, sampling_params_greedy]
    else:
        sampling_params_choices = [sampling_params_greedy]

    if args.long_prompt:
        with open("serve/tests/data/long_prompts.json", "r") as f:
            prompts = json.load(f)["prompts"]
            prompts = [prompts[0], prompts[2], prompts[3]]
    else:
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

    for i, prompt in enumerate(prompts):
        engine.add(
            [
                Request(
                    request_id=str(i),
                    messages=[ChatMessage(role="user", content=prompt)],
                    sampling_params=random.choice(sampling_params_choices),
                    stopping_criteria=StoppingCriteria(max_tokens=args.max_output_len, stop_sequences=None),
                    debug_options=DebugOptions(prompt=prompt),
                )
            ]
        )

    generated = ["" for _ in range(len(prompts))]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            seq = res.sequences[0]
            if not seq.is_finished:
                generated[int(res.request_id)] += seq.delta

    if args.long_prompt:
        for g in generated:
            print(f"Generated text = '{g}'")
    else:
        for p, g in zip(prompts, generated):
            print(f"Prompt = '{p}', generated text = '{g}'")

    if args.use_staging_engine:
        engine.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-num-sequences", type=int, default=8)
    parser.add_argument("--max-output-len", type=int, default=20)
    parser.add_argument("--prompt-allocate-ratio", type=float, default=2.0)
    parser.add_argument("--long-prompt", action="store_true")
    parser.add_argument("--use-random-sampling", action="store_true")
    parser.add_argument("--use-staging-engine", action="store_true")
    parser.add_argument("--min-decode-steps", type=int, default=12)
    parser.add_argument("--max-decode-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.model_artifact_path = os.path.join(args.artifact_path, args.local_id)
    if not os.path.exists(args.model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test(args)
