# pylint: disable=missing-docstring,too-many-instance-attributes
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import tvm
from mlc_chat.compiler import MODELS
from mlc_chat.compiler.model.llama_config import LlamaConfig
from mlc_chat.compiler.model.llama_quantization import huggingface_group_quantize
from mlc_chat.compiler.parameter import HuggingFaceLoader
from mlc_chat.support import tqdm
from tvm.runtime import NDArray

if TYPE_CHECKING:
    from tvm.relax.frontend import nn

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def test_load_torch_llama_group_quantize(base_path: Union[str, Path], target: str = "llvm"):
    @dataclass
    class TestGroupQuantizeConfig:
        name: str = "q4f16_1"
        kind: str = "group_quantize"
        group_size: int = 32
        weight_dtype: str = "float16"
        max_int_value: int = 7
        storage_dtype: str = "uint32"
        num_elem_per_storage: int = 8
        num_storage_per_group: int = 4
        quantize_dtype_bits: int = 4

        def quantize(self, _: "nn.Module") -> "nn.Module":
            raise NotImplementedError

    base_path = Path(base_path)
    path_config = base_path / "config.json"
    path_params = base_path / "pytorch_model.bin.index.json"

    model = MODELS["llama"]
    model_config = LlamaConfig.from_file(path_config)
    quantize_config = TestGroupQuantizeConfig()
    loader = HuggingFaceLoader(
        path=path_params,
        extern_param_map=model.source["huggingface-torch"](model_config, None),
        quantize_param_map=huggingface_group_quantize(
            model_config,
            quantize_config,
            target=tvm.target.Target(target),
        ),
    )
    with tqdm.redirect():
        for _name, _param in loader.load():
            ...


def test_group_quantize_vs_numpy():
    bits = {
        "int4": 4,
        "int8": 8,
        "fp16": 16,
        "fp32": 32,
        "int32": 32,
        "uint32": 32,
    }

    # pylint: disable=unused-variable
    def group_quantize_np(
        w: NDArray,  # pylint: disable=invalid-name
        quantize_dtype: str = "int4",
        storage_dtype: str = "uint32",
        group_size: int = 32,
        # symmetric: bool = True,
        # transpose: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        # pylint: disable=too-many-locals
        def _pad_axis_by_factor(tensor: np.ndarray, axis: int, factor: int) -> np.ndarray:
            dim = int(tensor.shape[axis])
            if dim % factor == 0:
                return tensor
            pad_width = [[0, 0] for i in tensor.shape]
            pad_width[axis][1] = factor - (dim % factor)
            return np.pad(tensor, pad_width, mode="constant", constant_values=0)

        def _clip(
            x: np.ndarray,  # pylint: disable=invalid-name
            x_min: int,
            x_max: int,
            dtype: str,
        ) -> np.ndarray:
            return np.clip(x, a_min=x_min, a_max=x_max).astype(dtype)

        num_elem_per_storage = bits[storage_dtype] // bits[quantize_dtype]
        assert group_size % num_elem_per_storage == 0
        num_storage_units = (group_size + num_elem_per_storage - 1) // num_elem_per_storage

        # using numpy for now
        w = w.numpy()

        # Step 1. Tile `w`: [n, k'] -> [n, k, group_size]
        w = _pad_axis_by_factor(w, axis=1, factor=group_size)
        n, k = [int(v) for v in w.shape]  # pylint: disable=invalid-name
        assert k % group_size == 0, "Padding is not working properly"
        k = k // group_size
        w = w.reshape([n, k, group_size])

        # Step 2. Calculate
        if quantize_dtype.startswith("int"):
            max_int_value = (2 ** (bits[quantize_dtype] - 1)) - 1
            # 1) `scale`: [n, k, group_size] -> [n, k]
            scale = np.maximum(np.amax(w, axis=-1), 1e-4) / max_int_value
            # 2) `w`: w / scale

            w = _clip(
                np.round(w / scale[:, :, np.newaxis]).astype("int") + max_int_value,
                x_min=0,
                x_max=max_int_value * 2,
                dtype=storage_dtype,
            )
        else:
            raise NotImplementedError

        # Step 3. Compress `w` to every `num_elem_per_storage` elements
        res = np.zeros((n, k, num_storage_units), dtype=np.uint32)
        for i in range(n):
            for j in range(k):
                for m in range(num_storage_units):  # pylint: disable=invalid-name
                    for k in range(num_elem_per_storage):
                        res[i, j, m] += w[i, j, m * num_elem_per_storage + k] * 2**k
        return tvm.nd.array(res), tvm.nd.array(scale)
        # pylint: enable=too-many-locals


if __name__ == "__main__":
    test_load_torch_llama_group_quantize(
        base_path="./dist/models/Llama-2-7b-hf",
        target="llvm",
    )
    test_load_torch_llama_group_quantize(
        base_path="./dist/models/Llama-2-7b-hf",
        target="nvidia/nvidia-a100",
    )
    test_load_torch_llama_group_quantize(
        base_path="./dist/models/Llama-2-13b-hf",
        target="llvm",
    )
    test_load_torch_llama_group_quantize(
        base_path="./dist/models/Llama-2-13b-hf",
        target="nvidia/nvidia-a100",
    )
