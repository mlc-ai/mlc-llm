"""A command line tool for benchmarking a chat model."""
import argparse
from pathlib import Path

from mlc_chat import ChatConfig, ChatModule

parser = argparse.ArgumentParser(description="Benchmark an MLC LLM ChatModule.")
parser.add_argument(
    "--model",
    type=str,
    help="""The model folder after compiling with MLC-LLM build process. The parameter can either
    be the model name with its quantization scheme (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a
    full path to the model folder. In the former case, we will use the provided name to search for
    the model folder over possible paths.""",
    required=True,
)
parser.add_argument(
    "--model-lib",
    type=str,
    help="""The compiled model library. In MLC LLM, an LLM is compiled to a shared or static
    library (.so or .a), which contains GPU computation to efficiently run the LLM. MLC Chat,
    as the runtime of MLC LLM, depends on the compiled model library to generate tokens.
    """,
    required=False,
)
parser.add_argument(
    "--tensor-parallel-shards",
    "--num-shards",
    type=int,
    help="Number of GPUs to be used.",
    dest="tensor_parallel_shards",
    required=False,
)
parser.add_argument(
    "--device",
    type=str,
    help="""The description of the device to run on. User should provide a string in the form of
    'device_name:device_id' or 'device_name', where 'device_name' is one of 'cuda', 'metal',
    'vulkan', 'rocm', 'opencl', and 'device_id' is the device id to run on. If no 'device_id' is
    provided, it will be set to 0 by default.
    """,
    required=True,
)
parser.add_argument(
    "--prompt",
    type=str,
    help="The prompt to generate from.",
    required=True,
)
parser.add_argument(
    "--generate-length",
    type=int,
    help="The length (numer of tokens) of the generated text.",
    required=True,
)


def _load_prompt(path_or_prompt: str) -> str:
    """Load the prompt from a file or use the provided prompt."""
    try:
        path = Path(path_or_prompt)
        if path.is_file():
            with path.open("r", encoding="utf-8") as in_file:
                return in_file.read()
    except:  # pylint: disable=bare-except
        pass
    return path_or_prompt


def main():
    """The main function that runs the benchmarking."""
    args = parser.parse_args()
    chat_module = ChatModule(
        model=args.model,
        device=args.device,
        chat_config=ChatConfig(tensor_parallel_shards=args.tensor_parallel_shards),
        model_lib_path=args.model_lib,
    )
    prompt = _load_prompt(args.prompt)
    output = chat_module.benchmark_generate(prompt, generate_length=args.generate_length)
    print(f"Generated text:\n{output}\n")
    print(f"Statistics: {chat_module.stats(verbose=True)}")


if __name__ == "__main__":
    main()
