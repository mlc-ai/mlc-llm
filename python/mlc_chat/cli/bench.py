"""Command line entrypoint of benchmark."""
from mlc_chat.help import HELP
from mlc_chat.interface.bench import bench
from mlc_chat.interface.chat import ChatConfigOverride
from mlc_chat.support.argparse import ArgumentParser


def main(argv):
    """Parse command line arguments and call `mlc_llm.interface.bench`."""
    parser = ArgumentParser("MLC LLM Chat CLI")

    parser.add_argument(
        "model",
        type=str,
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the meaning of life?",
        help=HELP["prompt"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="O2",
        help=HELP["opt"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_deploy"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--overrides",
        type=ChatConfigOverride.from_str,
        default="",
        help=HELP["chatconfig_overrides"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--generate-length",
        type=int,
        default=256,
        help=HELP["generate_length"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--model-lib-path",
        type=str,
        default=None,
        help=HELP["model_lib_path"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)
    bench(
        model=parsed.model,
        prompt=parsed.prompt,
        device=parsed.device,
        opt=parsed.opt,
        overrides=parsed.overrides,
        generate_length=parsed.generate_length,
        model_lib_path=parsed.model_lib_path,
    )
