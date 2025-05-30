"""The Python API for MLC Embeddings."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tvm
from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime import Device, Module
from tvm.runtime.relax_vm import VirtualMachine

from mlc_llm.serve import engine_utils
from mlc_llm.support.auto_device import detect_device
from mlc_llm.tokenizers import Tokenizer


def _extract_metadata(mod: Module):
    return json.loads(VirtualMachine(mod, tvm.runtime.device("cpu"))["_metadata"]())


def _load_params(
    model_weight_path: str, device: Device, model_metadata: Dict[str, Any]
) -> List[tvm.nd.NDArray]:
    params, meta = tvmjs.load_ndarray_cache(model_weight_path, device)
    param_names = [param["name"] for param in model_metadata["params"]]
    assert len(param_names) == meta["ParamSize"]

    plist = []
    for param_name in param_names:
        plist.append(params[param_name])
    return plist


def _get_tvm_module(
    model_weight_path: str, lib_path: str, device: Device, instrument: tvm.ffi.Function = None
):
    ex = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(ex, device)
    if instrument:
        vm.set_instrument(instrument)
    metadata = _extract_metadata(ex)
    params = _load_params(model_weight_path, device, metadata)
    return vm.module, params, metadata


class DefaultDebugInstrument:
    """The default debug instrument to use if users don't specify
    a customized one.

    This debug instrument will dump the arguments and output of each
    VM Call instruction into a .npz file. It will also alert the user
    if any function outputs are NaN or INF.
    """

    def __init__(self, debug_out: Path):
        """Constructor

        Parameters
        ----------
        debug_out : Path
            the directory to dump the .npz files
        """
        self.counter = 0
        self.first_nan_occurred = False
        self.first_inf_occurred = False
        self.debug_out = debug_out
        debug_out.mkdir(exist_ok=True, parents=True)

    def reset(self, debug_out: Path):
        """Reset the state of the Instrument class

        Parameters
        ----------
        debug_out : Path
            the directory to dump the .npz files
        """
        self.counter = 0
        self.first_nan_occurred = False
        self.first_inf_occurred = False
        self.debug_out = debug_out
        debug_out.mkdir(exist_ok=True, parents=True)

    def __call__(self, func, name, before_run, ret_val, *args):
        # Determine what functions to look at
        if before_run:  # Whether before the function is called or after
            return
        if name.startswith("vm.builtin.") and "attention_with_fused_qkv" not in name:
            return

        # Decide what to print or save about the function's arguments (where args[-1] is the
        # buffer we write the result to)
        func_name = f"f{self.counter}_{name}"

        # Save the arguments to npz
        arg_dict = {}
        for i, arg in enumerate(args):
            if isinstance(arg, tvm.nd.NDArray):
                arg_dict[f"arg_{i}"] = arg.numpy()

        np.savez(self.debug_out / f"{func_name}.npz", **arg_dict)

        self.counter += 1


class MLCEmbeddings:  # pylint: disable=too-few-public-methods
    """A class to embed queries using MLC LLM encoder models.

    Parameters
    ----------
    model: str
        The model folder after compiling with MLC-LLM build process. The parameter
        can either be the model name with its quantization scheme
        (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
        folder. In the former case, we will use the provided name to search
        for the model folder over possible paths.

    model_lib_path : str
        The full path to the model library file to use (e.g. a ``.so`` file).

    device : Optional[str]
        The description of the device to run on. User should provide a string in the
        form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
        'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
        local device), and 'device_id' is the device id to run on. If no 'device_id'
        is provided, it will be set to 0 by default.

    debug_dir: Path
        The output folder to store the dumped debug files. If None, will not dump any debug files.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        model_lib_path: str,
        device: Optional[str] = "auto",
        debug_dir: Optional[str] = None,
    ):
        self.device = detect_device(device)
        instrument = DefaultDebugInstrument(Path(debug_dir)) if debug_dir else None
        self.mod, self.params, self.metadata = _get_tvm_module(
            model, model_lib_path, self.device, instrument
        )
        self.model_path = model
        self.tokenizer = Tokenizer(self.model_path)
        self.prefill_func = self.mod["prefill"]

    def embed(self, queries: List[str]) -> tvm.runtime.NDArray:
        """
        Embeds a list of queries in a single batch.

        Parameters
        ----------
        queries : List[str]
            A list of queries to embed.

        Returns
        -------
        List[float]
            A list of embeddings for the queries.
        """
        tokens, attention_mask = self._tokenize_queries(queries)
        tokens_tvm = tvm.nd.array(tokens.astype("int32"), device=self.device)
        attention_mask_tvm = tvm.nd.array(attention_mask.astype("int32"), device=self.device)
        output = self.prefill_func(tokens_tvm, attention_mask_tvm, self.params)
        return output

    def _tokenize_queries(self, queries: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tokens = engine_utils.process_prompts(queries, self.tokenizer.encode)  # type: ignore
        max_query_length = max(len(token_seq) for token_seq in tokens)

        token_inputs: np.ndarray = np.zeros((len(tokens), max_query_length), dtype=np.int32)
        attention_mask: np.ndarray = np.zeros((len(tokens), max_query_length), dtype=np.int32)

        for i, token_seq in enumerate(tokens):
            token_inputs[i, : len(token_seq)] = token_seq
            attention_mask[i, : len(token_seq)] = 1

        return token_inputs, attention_mask
