"""This script is an example of running and comparing the outputs of two different TVM Relax VMs.
"""
# pylint: disable=missing-docstring,invalid-name
import json

import numpy as np
import torch
import tvm
from transformers import LlamaTokenizer
from tvm import relax
from tvm.contrib import tvmjs

KVCACHE_FUNCS = [
    "vm.builtin.attention_kv_cache_append",
    "vm.builtin.attention_kv_cache_view",
]
DEVICE = "cuda:0"
PROMPT = "What is the meaning of life?"
TOKENIZER = "./dist/debug-llama/"

COMBO = {
    "CURRENT": {
        "model_lib": "./dist/debug-llama/llama.so",
        "params": "./dist/debug-llama",
        "target_func": "fused_fused_dequantize1_NT_matmul6",
    },
    "LEGACY": {
        "model_lib": "./dist/Llama-2-7b-chat-hf-q4f16_1/Llama-2-7b-chat-hf-q4f16_1-cuda.so",
        "params": "./dist/Llama-2-7b-chat-hf-q4f16_1/params",
        "target_func": "fused_fused_decode2_NT_matmul",
    },
}


class Instrument:  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        target_func: str,
    ):
        self.first_time = True
        self.target_func = target_func
        self.saved_args = []  # type: ignore

    def __call__(
        self,
        func,
        func_symbol: str,
        before_run: bool,
        ret_value,
        *args,
    ):
        if before_run:
            return
        if func_symbol.startswith("vm.builtin."):
            if func_symbol not in KVCACHE_FUNCS:
                return
        if func_symbol == self.target_func and self.first_time:
            self.first_time = False
            for arg in args:
                print(arg.shape, arg.dtype)
                self.saved_args.append(arg.numpy())


class TestState:
    def __init__(self, device, model_lib, target_func):
        self.mod = relax.VirtualMachine(
            tvm.runtime.load_module(model_lib),
            device,
        )
        self.inst = Instrument(target_func=target_func)
        self.mod.set_instrument(self.inst)


def _tokenize(sentence: str):
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.to(torch.int32).numpy()
    print(f"Tokenizing: {sentence}")
    print(f"Tokens: {tokens}")
    return tokens


def _load_params(params, device, metadata):
    param_dict, _ = tvmjs.load_ndarray_cache(params, device)
    param_list = []
    for name in [x["name"] for x in metadata["params"]]:
        param_list.append(param_dict[name])
    return param_list


def _load_params_legacy(params, device):
    param_dict, metadata = tvmjs.load_ndarray_cache(params, device)
    param_list = []
    for i in range(metadata["ParamSize"]):
        param_list.append(param_dict[f"param_{i}"])
    return param_list


def _as_input_tuple(scalar):
    return tvm.runtime.ShapeTuple([scalar])


@tvm.register_func("debug_save")
def _debug_save(x, _):
    return tvm.nd.array(x.numpy(), x.device)


def main() -> None:
    device = tvm.device(DEVICE)
    prompt = _tokenize(PROMPT)

    def _run_legacy(model_lib, params, target_func):
        state = TestState(device, model_lib, target_func)
        kv_cache = state.mod["create_kv_cache"]()
        param_list = _load_params_legacy(params, device)
        state.mod["prefill"](
            tvm.nd.array(prompt, device),
            _as_input_tuple(len(prompt[0])),
            kv_cache,
            param_list,
        )
        return state.inst.saved_args

    def _run_current(model_lib, params, target_func):
        state = TestState(device, model_lib, target_func)
        metadata = json.loads(state.mod["_metadata"]())
        kv_cache = state.mod["_initialize_effect"]()
        param_list = _load_params(params, device, metadata)
        state.mod["prefill"](
            tvm.nd.array(prompt, device),
            _as_input_tuple(len(prompt[0])),
            kv_cache,
            param_list,
        )
        return state.inst.saved_args

    print("============== Running old flow =================")
    new_args = _run_current(**COMBO["CURRENT"])
    print("============== Running new flow =================")
    old_args = _run_legacy(**COMBO["LEGACY"])

    for i, (new_arg, old_arg) in enumerate(zip(new_args, old_args)):
        print(f"Checking arg {i}")
        np.testing.assert_allclose(new_arg, old_arg, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    main()
