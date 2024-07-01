"""Debug compiled models with TVM instrument"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import tvm
from tvm import rpc, runtime
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument

from mlc_llm.interface.help import HELP
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.testing.debug_chat import DebugChat


def _print_as_table(sorted_list):
    print("=" * 100)
    print(
        "Name".ljust(50)
        + "Time (ms)".ljust(12)
        + "Count".ljust(8)
        + "Total time (ms)".ljust(18)
        + "Percentage (%)"
    )
    total_time = sum(record[1][0] * record[1][1] for record in sorted_list) * 1000
    for record in sorted_list:
        time = record[1][0] * 1000
        weighted_time = time * record[1][1]
        percentage = weighted_time / total_time * 100
        print(
            record[0].ljust(50)
            + f"{time:.4f}".ljust(12)
            + str(record[1][1]).ljust(8)
            + f"{weighted_time:.4f}".ljust(18)
            + f"{percentage:.2f}"
        )
    print(f"Total time: {total_time:.4f} ms")


class LibCompare(LibCompareVMInstrument):
    """The default debug instrument to use if users don't specify
    a customized one.

    This debug instrument will dump the arguments and output of each
    VM Call instruction into a .npz file. It will also alert the user
    if any function outputs are NaN or INF.

    Parameters
    ----------
    mod: runtime.Module
        The module of interest to be validated.

    device: runtime.Device
        The device to run the target module on.

    time_eval: bool
        Whether to time evaluate the functions.

    rtol: float
        rtol used in validation

    atol: float
        atol used in validation
    """

    def __init__(  # pylint: disable=too-many-arguments, unused-argument
        self,
        mod: runtime.Module,
        device: runtime.Device,
        debug_out: Path,
        time_eval: bool = True,
        rtol: float = 1e-2,
        atol: float = 1,
        skip_rounds: int = 0,
    ):
        super().__init__(mod, device, True, rtol, atol)
        self.debug_out = debug_out
        self.time_eval = time_eval
        self.time_eval_results: Dict[str, Tuple[float, int]] = {}
        self.visited: Set[str] = set([])
        self.skip_rounds = skip_rounds
        self.counter = 0
        debug_out.mkdir(exist_ok=True, parents=True)

    def reset(self, debug_out: Path):  # pylint: disable=unused-argument
        """Reset the state of the Instrument class

        Note
        ----
        `debug_out` is not used in this class.

        Parameters
        ----------
        debug_out : Path
            the directory to dump the .npz files
        """
        self.debug_out = debug_out
        _print_as_table(
            sorted(
                self.time_eval_results.items(),
                key=lambda x: -(x[1][0] * x[1][1]),
            )
        )
        self.time_eval_results = {}
        self.visited = set([])
        self.counter = 0
        debug_out.mkdir(exist_ok=True, parents=True)

    def skip_instrument(self, func, name, before_run, ret_val, *args):
        if name.startswith("shape_func"):
            return True
        if self.counter < self.skip_rounds:
            self.counter += 1
            print(f"[{self.counter}] Skip validating {name}..")
            return True
        if name in self.visited:
            if self.time_eval and name in self.time_eval_results:
                record = self.time_eval_results[name]
                self.time_eval_results[name] = (record[0], record[1] + 1)
            return True
        self.visited.add(name)
        return False

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ):
        super().compare(name, ref_args, new_args, ret_indices)

        if self.time_eval and name not in self.time_eval_results:
            res = self.mod.time_evaluator(
                name,
                self.device,
                number=20,
                repeat=3,
                min_repeat_ms=100,
                # cache_flush_bytes=256 * 10**6
            )(*new_args)
            self.time_eval_results[name] = (res.mean, 1)
            print(f"Time-eval result {name} on {self.device}:\n {res}")


def get_instrument(args):
    """Get the debug instrument from the CLI arguments"""
    if args.cmp_device is None:
        assert args.cmp_lib_path is None, "cmp_lib_path must be None if cmp_device is None"
        args.cmp_device = args.device
        args.cmp_lib_path = args.model_lib

    if args.cmp_device == "iphone":
        assert args.cmp_lib_path.endswith(".dylib"), "Require a dylib file for iPhone"
        proxy_host = os.environ.get("TVM_RPC_PROXY_HOST", "127.0.0.1")
        proxy_port = int(os.environ.get("TVM_RPC_PROXY_PORT", "9090"))
        sess = rpc.connect(proxy_host, proxy_port, "iphone")
        sess.upload(args.cmp_lib_path)
        lib = sess.load_module(os.path.basename(args.cmp_lib_path))
        cmp_device = sess.metal()
    elif args.cmp_device == "android":
        assert args.cmp_lib_path.endswith(".so"), "Require a so file for Android"
        tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
        tracker_port = int(os.environ.get("TVM_TRACKER_PORT", "9190"))
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        sess = tracker.request("android")
        sess.upload(args.cmp_lib_path)
        lib = sess.load_module(os.path.basename(args.cmp_lib_path))
        cmp_device = sess.cl(0)
    else:
        lib = tvm.runtime.load_module(args.cmp_lib_path)
        cmp_device = tvm.device(args.cmp_device)

    return LibCompare(
        lib,
        cmp_device,
        time_eval=args.time_eval,
        debug_out=Path(args.debug_dir),
    )


def main():
    """The main function to start a DebugChat CLI"""

    parser = ArgumentParser("MLC LLM Chat Debug Tool")
    parser.add_argument(
        "prompt",
        type=str,
        help="The user input prompt.",
    )
    parser.add_argument(
        "--generate-len", type=int, help="Number of output tokens to generate.", required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        help="An MLC model directory that contains `mlc-chat-config.json`",
        required=True,
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        help="The full path to the model library file to use (e.g. a ``.so`` file).",
        required=True,
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        help="The output folder to store the dumped debug files.",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_compile"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--cmp-device",
        type=str,
        default="none",
    )
    parser.add_argument(
        "--cmp-lib-path",
        type=str,
        default="none",
    )
    parser.add_argument(
        "--time-eval",
        action="store_true",
        help="Whether to time evaluate the functions.",
    )
    parsed = parser.parse_args()
    instrument = get_instrument(parsed)
    debug_chat = DebugChat(
        model=parsed.model,
        model_lib=parsed.model_lib,
        debug_dir=Path(parsed.debug_dir),
        device=parsed.device,
        debug_instrument=instrument,
    )
    debug_chat.generate(parsed.prompt, parsed.generate_len)
    # Only print decode for now
    _print_as_table(
        sorted(
            instrument.time_eval_results.items(),
            key=lambda x: -(x[1][0] * x[1][1]),
        )
    )


if __name__ == "__main__":
    main()
