# pylint: disable=invalid-name,missing-docstring
import numpy as np
from tvm.relax.frontend.nn import spec

from mlc_llm.models.llama import LlamaConfig, LlamaForCasualLM


def main():
    config = LlamaConfig(
        hidden_act="silu",
        hidden_size=256,
        intermediate_size=688,
        max_sequence_length=128,
        num_attention_heads=8,
        num_hidden_layers=8,
        rms_norm_eps=1e-05,
        vocab_size=4096,
        position_embedding_base=10000,
    )
    batch_size, total_seq_len, dtype = 1, 32, "float32"

    # Usecase 1. Define a model and export it to TVM's IRModule
    model = LlamaForCasualLM(config)
    model.to(dtype=dtype)
    mod_spec = {
        "prefill": {
            "inputs": spec.Tensor([batch_size, "seq_len"], "int32"),
            "total_seq_len": int,
        },
        "decode": {
            "inputs": spec.Tensor([batch_size, 1], "int32"),
            "total_seq_len": int,
        },
        "softmax_with_temperature": {
            "logits": spec.Tensor([1, 1, config.vocab_size], "float32"),
            "temperature": spec.Tensor([], "float32"),
        },
    }
    mod, _ = model.export_tvm(spec=mod_spec)
    mod.show(black_format=False)

    # Usecase 2. JIT compile a model
    for _, param in model.state_dict().items():
        param.data = np.random.rand(*param.shape).astype(param.dtype)
    model = model.jit(
        spec=mod_spec,
        target="llvm",
        device="cpu",
        out_format="torch",
    )

    # Usecase 3. Run a model with PyTorch
    import torch  # pylint: disable=import-outside-toplevel

    result = model["prefill"](
        torch.from_numpy(
            np.random.randint(
                0,
                config.vocab_size,
                size=(batch_size, total_seq_len),
                dtype="int32",
            )
        ),
        total_seq_len,
    )
    assert isinstance(result, torch.Tensor)


if __name__ == "__main__":
    main()
