# pylint: disable=invalid-name,missing-docstring,import-error
from typing import Tuple, Any

import torch

import tvm
from tvm.relax.frontend.nn import spec

from mlc_llm.relax_model.llava import LlavaConfig, LlavaVisionModel


def load_llava_from_hf():
    return None


def load_data():
    pass


def pipe():
    pass


def main():
    # Set the device and target
    dev = tvm.cuda()
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": 65536,
            "arch": "sm_" + tvm.cuda().compute_version.replace(".", ""),
        }
    )

    # load model from transformers
    hf_model = load_llava_from_hf()

    # define the model config
    config = LlavaConfig(**hf_model.config.to_dict())

    # define the model
    model = LlavaVisionModel(config=config)

    # define the mod spec
    mod_spec = {}

    # Usercase1, export it to TVM's IRModule, use `mod.show()` to print the IRModule
    mod, _ = model.export_tvm(spec=mod_spec)

    # Usercase2, JIT compile a model
    for name, param in model.state_dict().items():
        param.data = hf_model.state_dict()[name]

    model = model.jit(spec=mod_spec, target=target, device="cuda", out_format="torch", debug=True)

    # Test on librispeech_asr_dummy
    input_image = load_data()
    generated_tokens = pipe(model, config, input_image)


if __name__ == "__main__":
    main()
