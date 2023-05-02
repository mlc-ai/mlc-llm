# Used as reference

import argparse
import os
from typing import Callable

import numpy as np
import torch
import tvm
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
from tvm import relax

from mlc_llm import utils
from mlc_llm.conversation import SeparatorStyle, compute_skip_echo_len, conv_templates


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _parse_args():
    args = argparse.ArgumentParser()
    utils.argparse_add_common(args)
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--max-gen-len", type=int, default=2048)
    args.add_argument("--run-torch-model", action="store_true", default=False)
    parsed = args.parse_args()
    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, parsed.model, parsed.dtype
    )
    utils.argparse_postproc_common(parsed)
    return parsed


class ModelWrapper:
    def __init__(self, model: Callable, tokenizer, args: argparse.Namespace):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
        stop_tokens=None,
        keep_first_token=True,
    ):
        prompt_tokens = self.tokenizer.encode(prompt)
        stop_tokens = (
            [self.tokenizer.eos_token_id] if stop_tokens is None else stop_tokens
        )
        if not keep_first_token:
            prompt_tokens = prompt_tokens[1:]
        total_len = max_gen_len + len(prompt_tokens)
        tokens = torch.full((1, total_len), 0).to(torch.int32)
        tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens)
        start_pos = len(prompt_tokens)
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                logits = self.model(tokens[:, :cur_pos])
            else:
                logits = self.model(tokens[:, cur_pos - 1 : cur_pos])
            logits = logits[:, -1, :].to(torch.float64)
            np_logits = logits.detach().cpu().numpy().astype("float64")
            if self.args.debug_dump:
                print(
                    f"logits: min = {np_logits.min()}, max = {np_logits.max()}, "
                    f"mean = {np_logits.mean()}, std = {np_logits.std()}",
                )
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            # the following code assumes bsz == 1
            if next_token[0] in stop_tokens:
                stopped = True
            else:
                stopped = False

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == max_gen_len - 1 or stopped:
                output = tokens[0, : cur_pos + 1]
                output = self.tokenizer.decode(output, skip_special_tokens=True)
                if stop_str:
                    pos = output.rfind(stop_str, len(prompt))
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                yield output
            if stopped:
                break


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def chat(model_wrapper, args):
    # Chat
    conv = conv_templates[args.conv_template].copy()
    keep_first_token = True
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt_unprocessed()
        print(f"{conv.roles[1]}: ", end="", flush=True)
        pre = 0
        skip_echo_len = compute_skip_echo_len(args.conv_template, conv, prompt)
        stop_tokens = (
            [50278, 50279, 50277, 1, 0] if args.conv_template == "stablelm" else None
        )
        for outputs in model_wrapper.generate(
            prompt,
            args.max_gen_len,
            stop_str=conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
            keep_first_token=keep_first_token,
            stop_tokens=stop_tokens,
        ):
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(
                    Colors.OKBLUE + " ".join(outputs[pre : now - 1]) + Colors.ENDC,
                    end=" ",
                    flush=True,
                )
                pre = now - 1
        print(
            Colors.OKBLUE + " ".join(outputs[pre:]) + Colors.ENDC,
            flush=True,
        )

        conv.messages[-1][-1] = " ".join(outputs)
        if "vicuna" in args.model:
            keep_first_token = False
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


def get_tvm_model(args):
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    ex = tvm.runtime.load_module(
        f"{args.artifact_path}/{args.model}_{args.device_name}_{args.dtype}.so"
    )
    vm = relax.VirtualMachine(ex, device)

    class Model:
        def __init__(self, args) -> None:
            self.tot_seq_len = 0
            self.kv_cache = vm["create_kv_cache"]()
            self.args = args

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            inputs = tvm.nd.array(inputs.numpy(), device=device)
            self.tot_seq_len += inputs.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([self.tot_seq_len])
            if inputs.shape[1] > 1:
                logits, kv_cache = vm["encoding"](
                    inputs, seq_len_shape, self.kv_cache, const_params
                )
            else:
                logits, kv_cache = vm["decoding"](
                    inputs, seq_len_shape, self.kv_cache, const_params
                )
            self.kv_cache = kv_cache
            if self.args.debug_dump:
                from tvm._ffi import get_global_func

                f_view = get_global_func("vm.builtin.attention_kv_cache_view")
                for i, cache in enumerate(self.kv_cache):
                    cache = f_view(cache).numpy().astype("float64")
                    print(
                        f"Cache {i}: shape = {cache.shape}, min = {cache.min()}, "
                        f"max = {cache.max()}, mean = {cache.mean()}, std = {cache.std()}"
                    )
                    np.savez(f"/tmp/kvcache-{self.args.dtype}/{i}.npz", cache=cache)

            return torch.from_numpy(logits.numpy())

    model = Model(args)
    return model.forward


def get_pytorch_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    class Model:
        def __init__(self):
            self.past_key_values = None

        def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            output = model(inputs, use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = output.past_key_values
            return output.logits

    wrapped_model = Model()
    return wrapped_model.forward


def main():
    ARGS = _parse_args()
    if ARGS.debug_dump:
        torch.manual_seed(12)
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if ARGS.model.startswith("dolly-"):
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    if not ARGS.run_torch_model:
        model = ModelWrapper(get_tvm_model(ARGS), tokenizer, ARGS)
    else:
        model = ModelWrapper(get_pytorch_model(ARGS), tokenizer, ARGS)
    chat(model, ARGS)


if __name__ == "__main__":
    main()
