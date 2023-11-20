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
    FinishReason,
    get_engine_config
)
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule

def create_engine(
        model_artifact_path, 
        use_staging_engine, 
        max_num_batched_tokens, 
        max_input_len,
        
    ):
    engine_config = get_engine_config({
        "use_staging_engine": use_staging_engine,
        "max_num_batched_tokens": max_num_batched_tokens, 
        "max_input_len": max_input_len,    
        # Use defaults for "min_decode_steps", "max_decode_steps", "prompt_allocate_ratio"
    })

    if use_staging_engine:
        engine = StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": model_artifact_path,
                "engine_config": engine_config,
            },
        )
        engine.start()
    else:
        engine = SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path = model_artifact_path,
                engine_config = engine_config,
        ))
    return engine

def create_request(idx, prompt, temp, max_tokens, stop, ignore_eos):
    return Request(
        request_id = str(idx),
        messages = [ChatMessage(role="user", content=prompt)],
        sampling_params = SamplingParams(
                            temperature=0.0,
        ), 
        stopping_criteria = StoppingCriteria(
            max_tokens=max_tokens, 
            stop_sequences=stop
        ), 
        debug_options = DebugOptions(ignore_eos = ignore_eos)
    )

def test_max_tokens(
        model_artifact_path, 
        use_staging_engine, 
        max_num_batched_tokens=2560, 
        max_input_len=2560,
        num_requests=5,
        ignore_eos=False
    ):
    prompt = "Write a merge sort program in Python."
    engine = create_engine(
        model_artifact_path, 
        use_staging_engine, 
        max_num_batched_tokens, 
        max_input_len,
    )

    requests = [create_request(idx=str(n-1), prompt=prompt, temp=0, max_tokens=n, stop=None, ignore_eos=ignore_eos) for n in range(1, num_requests)]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]
            
            if seq.is_finished:
                assert seq.num_generated_tokens == requests[int(res.request_id)].stopping_criteria.max_tokens
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta

    if use_staging_engine:
        engine.stop()


def test_ignore_eos(
    model_artifact_path, 
    use_staging_engine, 
    max_num_batched_tokens=2560, 
    max_input_len=2560,
    num_requests=5,
):
    prompt = "hi"
    engine = create_engine(
        model_artifact_path, 
        use_staging_engine, 
        max_num_batched_tokens, 
        max_input_len,
    )
    s = 113
    requests = [create_request(idx=str(n-s), prompt=prompt, temp=0, max_tokens=n, stop=None, ignore_eos=True) for n in range(s, s+num_requests)]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert seq.num_generated_tokens == requests[int(res.request_id)].stopping_criteria.max_tokens
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta

    if use_staging_engine:
        engine.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="../../../dist")
    args = parser.parse_args()
    model_artifact_path = os.path.join(args.artifact_path, args.local_id)
    
    test_max_tokens(model_artifact_path, use_staging_engine=True)
    test_max_tokens(model_artifact_path, use_staging_engine=False)
    test_ignore_eos(model_artifact_path, use_staging_engine=True)
    test_ignore_eos(model_artifact_path, use_staging_engine=False)
