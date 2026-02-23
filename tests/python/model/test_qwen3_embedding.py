# pylint: disable=invalid-name,missing-docstring
import json
import os

import numpy as np
import pytest
import torch
import tvm
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer
from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime import ShapeTuple
from tvm.runtime.vm import VirtualMachine

MLC_QWEN3_EMB_HF_DIR = os.environ.get("MLC_QWEN3_EMB_HF_DIR")
MLC_QWEN3_EMB_MODEL_DIR = os.environ.get("MLC_QWEN3_EMB_MODEL_DIR")
MLC_QWEN3_EMB_MODEL_LIB = os.environ.get("MLC_QWEN3_EMB_MODEL_LIB")
MLC_QWEN3_EMB_DEVICE = os.environ.get("MLC_QWEN3_EMB_DEVICE", "cuda")

_skip = not all([MLC_QWEN3_EMB_HF_DIR, MLC_QWEN3_EMB_MODEL_DIR, MLC_QWEN3_EMB_MODEL_LIB])
_skip_reason = (
    "Set MLC_QWEN3_EMB_HF_DIR, MLC_QWEN3_EMB_MODEL_DIR, " "MLC_QWEN3_EMB_MODEL_LIB to run this test"
)

TEST_TEXTS = [
    "What is machine learning?",
    "CMU is Carnegie Mellon University",
    "机器学习是人工智能的一个分支",
    "量子コンピュータの基本原理を説明してください",
    "머신러닝은 인공지능의 한 분야입니다.",
    (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: What is the capital of China?"
    ),
    (
        "The Transformer architecture, introduced in the paper Attention Is All You Need, "
        "revolutionized natural language processing by replacing recurrent layers with "
        "self-attention mechanisms. This allows the model to process all positions in a "
        "sequence simultaneously rather than sequentially, leading to significant improvements "
        "in both training efficiency and the ability to capture long-range dependencies. "
        "The key innovation is the multi-head attention mechanism, which allows the model "
        "to jointly attend to information from different representation subspaces at "
        "different positions."
    ),
    "Hello",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]


def _load_embed_weight(hf_dir):
    safetensor_files = [f for f in os.listdir(hf_dir) if f.endswith(".safetensors")]
    for sf in safetensor_files:
        with safe_open(os.path.join(hf_dir, sf), framework="pt", device="cpu") as f:
            if "embed_tokens.weight" in f.keys():
                return f.get_tensor("embed_tokens.weight")
    raise FileNotFoundError(f"embed_tokens.weight not found in {hf_dir}")


def _hf_logits(text, tokenizer, hf_model, embed_weight):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hidden = hf_model(**inputs).last_hidden_state.float()
        logits = hidden @ embed_weight.float().T
    return logits[0, -1, :].numpy()


def _mlc_logits(text, tokenizer, mlc_module, params, metadata, dev, embed_weight):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].numpy().astype(np.int32)
    seq_len = len(input_ids)

    embed_func = mlc_module["embed"]
    prefill_func = mlc_module["prefill_to_last_hidden_states"]

    if mlc_module.implements_function("create_flashinfer_paged_kv_cache"):
        create_kv = mlc_module["create_flashinfer_paged_kv_cache"]
    elif mlc_module.implements_function("create_tir_paged_kv_cache"):
        create_kv = mlc_module["create_tir_paged_kv_cache"]
    else:
        raise RuntimeError("Cannot find KV cache creation function")

    sliding_window = metadata.get("sliding_window_size", -1)
    context_window = metadata.get("context_window_size", 32768)
    prefill_chunk = metadata.get("prefill_chunk_size", 2048)
    max_seq_len = sliding_window if context_window == -1 else context_window

    kv_cache = create_kv(
        ShapeTuple([1]),
        ShapeTuple([max_seq_len]),
        ShapeTuple([prefill_chunk]),
        ShapeTuple([16]),
        ShapeTuple([int(sliding_window != -1)]),
    )

    nd_view = tvm.get_global_func("vm.builtin.reshape")
    add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    tokens_tvm = tvm.runtime.tensor(input_ids, device=dev)
    embedding = embed_func(tokens_tvm, params)
    embedding = nd_view(embedding, ShapeTuple([1, seq_len, embedding.shape[-1]]))

    add_sequence(kv_cache, 0)
    begin_forward(kv_cache, ShapeTuple([0]), ShapeTuple([seq_len]))
    hidden_states, _ = prefill_func(embedding, kv_cache, params)
    end_forward(kv_cache)

    # Compute logits from hidden states using embed_tokens weight (tie_word_embeddings)
    hidden = hidden_states.numpy().astype(np.float32)
    logits = hidden @ embed_weight.float().numpy().T
    return logits[0, -1, :]


@pytest.mark.skipif(_skip, reason=_skip_reason)
def test_mlc_hf_logit_match():
    tokenizer = AutoTokenizer.from_pretrained(MLC_QWEN3_EMB_HF_DIR, padding_side="left")
    hf_model = AutoModel.from_pretrained(MLC_QWEN3_EMB_HF_DIR)
    embed_weight = _load_embed_weight(MLC_QWEN3_EMB_HF_DIR)

    dev = tvm.runtime.device(MLC_QWEN3_EMB_DEVICE, 0)
    ex = tvm.runtime.load_module(MLC_QWEN3_EMB_MODEL_LIB)
    vm = relax.VirtualMachine(ex, dev)
    mlc_module = vm.module

    metadata = json.loads(VirtualMachine(ex, tvm.runtime.device("cpu"))["_metadata"]())
    params_dict, _ = tvmjs.load_tensor_cache(MLC_QWEN3_EMB_MODEL_DIR, dev)
    param_names = [p["name"] for p in metadata["params"]]
    params = [params_dict[name] for name in param_names]

    for text in TEST_TEXTS:
        hf = _hf_logits(text, tokenizer, hf_model, embed_weight)
        mlc = _mlc_logits(text, tokenizer, mlc_module, params, metadata, dev, embed_weight)

        cos_sim = np.dot(hf, mlc) / (np.linalg.norm(hf) * np.linalg.norm(mlc))
        assert cos_sim > 0.99, f"[{text[:30]}] Cosine similarity {cos_sim:.6f} below 0.99"

        max_diff = np.max(np.abs(hf - mlc))
        assert max_diff < 1.0, f"[{text[:30]}] Max absolute diff {max_diff:.6e} exceeds 1.0"

        hf_top10 = set(np.argsort(hf)[-10:])
        mlc_top10 = set(np.argsort(mlc)[-10:])
        overlap = len(hf_top10 & mlc_top10)
        assert overlap >= 7, f"[{text[:30]}] Top-10 overlap {overlap}/10 below 7"


if __name__ == "__main__":
    test_mlc_hf_logit_match()
