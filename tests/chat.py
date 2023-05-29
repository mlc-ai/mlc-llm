# Used as reference

import argparse
import os
from typing import Callable

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore[import]
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
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--max-gen-len", type=int, default=2048)
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )
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
        temperature: float = 1.1,
        top_p: float = 0.7,
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
        stop_str = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
        if conv.sep_style == SeparatorStyle.REDPAJAMA_CHAT:
            stop_str = "<human>:"
        for outputs in model_wrapper.generate(
            prompt,
            args.max_gen_len,
            stop_str=stop_str,
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
        os.path.join(
            args.artifact_path,
            f"{args.model}-{args.quantization.name}-{args.device_name}.so",
        )
    )
    vm = relax.VirtualMachine(ex, device)

    class Model:
        def __init__(self, args) -> None:
            self.tot_seq_len = 0
            self.kv_cache = vm["create_kv_cache"]()
            self.args = args
            try:
                self.prefill_func = vm["prefill"]
            except AttributeError:
                self.prefill_func = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            inputs = inputs.numpy()
            self.tot_seq_len += inputs.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([self.tot_seq_len])
            if inputs.shape[1] > 1 and self.prefill_func:
                inputs = tvm.nd.array(inputs.numpy(), device=device)
                logits, kv_cache = self.prefill_func(
                    inputs, seq_len_shape, self.kv_cache, const_params
                )
            else:
                for i in range(inputs.shape[1]):
                    input_slice = tvm.nd.array(inputs[:, i : i + 1], device=device)
                    logits, kv_cache = vm["decode"](
                        input_slice, seq_len_shape, self.kv_cache, const_params
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
                    np.savez(
                        f"/tmp/kvcache-{self.quantization.name}/{i}.npz", cache=cache
                    )

            return torch.from_numpy(logits.numpy())

    model = Model(args)
    return model.forward


def main():
    ARGS = _parse_args()
    if ARGS.debug_dump:
        torch.manual_seed(12)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(ARGS.artifact_path, "params"), trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if ARGS.model.startswith("dolly-"):
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    model = ModelWrapper(get_tvm_model(ARGS), tokenizer, ARGS)
    chat(model, ARGS)


if __name__ == "__main__":
    main()
