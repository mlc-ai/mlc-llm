"""Command line entrypoint of router."""

from mlc_llm.interface.help import HELP
from mlc_llm.interface.router import serve
from mlc_llm.support.argparse import ArgumentParser


def main(argv):
    """Parse command line arguments and call `mlc_llm.interface.router`."""

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(",")

    parser = ArgumentParser("MLC LLM Router Serve CLI")
    parser.add_argument(
        "model",
        type=str,
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        default=None,
        help=HELP["model_lib"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--router-mode",
        type=str,
        choices=["disagg", "round-robin"],
        default="disagg",
        help="router mode" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--router-host",
        type=str,
        default="127.0.0.1",
        help="router host" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=8000,
        help="router port" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--endpoint-hosts",
        type=list_of_strings,
        default="127.0.0.1",
        help="Host of each endpoint, seperated by comma." + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--endpoint-ports",
        nargs="*",
        type=int,
        default=[8080],
        help="Port of each endpoint, separated by space." + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--endpoint-num-gpus",
        nargs="*",
        type=int,
        default=[1],
        help="Number of GPUs of each endpoint, separated by space." + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--enable-prefix-cache",
        default=False,
        action="store_true",
        help="whether to enable prefix cache" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--pd-balance-factor",
        type=float,
        default=0.0,
        help=HELP["pd_balance_factor"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)
    serve(
        model=parsed.model,
        model_lib=parsed.model_lib,
        router_host=parsed.router_host,
        router_port=parsed.router_port,
        endpoint_hosts=parsed.endpoint_hosts,
        endpoint_ports=parsed.endpoint_ports,
        endpoint_num_gpus=parsed.endpoint_num_gpus,
        enable_prefix_cache=parsed.enable_prefix_cache,
        router_mode=parsed.router_mode,
        pd_balance_factor=parsed.pd_balance_factor,
    )
