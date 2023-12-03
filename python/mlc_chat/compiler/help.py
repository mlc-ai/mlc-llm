"""Help message for CLI arguments."""
from .model import MODEL_PRESETS

HELP = {
    "model": (
        """
1) Path to a HuggingFace model directory that contains a `config.json` or
2) Path to `config.json` in HuggingFace format, or
3) The name of a pre-defined model architecture.

A `config.json` file in HuggingFace format defines the model architecture, including the vocabulary
size, the number of layers, the hidden size, number of attention heads, etc.
Example: https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json.

A HuggingFace directory often contains a `config.json` which defines the model architecture,
the non-quantized model weights in PyTorch or SafeTensor format, tokenizer configurations,
as well as an optional `generation_config.json` provides additional default configuration for
text generation.
Example: https://huggingface.co/codellama/CodeLlama-7b-hf/tree/main.

Pre-defined model architectures include """
        + ", ".join(f'"{preset}"' for preset in MODEL_PRESETS)
        + "."
    ).strip(),
    "quantization": """
Quantization format.
""".strip(),
    "model_type": """
Model architecture such as "llama". If not set, it is inferred from `config.json`.
""".strip(),
    "device_compile": """
The GPU device to compile the model to. If not set, it is inferred from GPUs available locally.
""".strip(),
    "device_quantize": """
The device used to do quantization such as "cuda" or "cuda:0". Will detect from local available GPUs
if not specified.
""".strip(),
    "host": """
The host LLVM triple to compile the model to. If not set, it is inferred from the local CPU and OS.
Examples of the LLVM triple:
1) iPhones: arm64-apple-ios;
2) ARM64 Android phones: aarch64-linux-android;
3) WebAssembly: wasm32-unknown-unknown-wasm;
4) Windows: x86_64-pc-windows-msvc;
5) ARM macOS: arm64-apple-darwin.
""".strip(),
    "opt": """
Optimization flags. MLC LLM maintains a predefined set of optimization flags,
denoted as O0, O1, O2, O3, where O0 means no optimization, O2 means majority of them,
and O3 represents extreme optimization that could potentially break the system.
Meanwhile, optimization flags could be explicitly specified via details knobs, e.g.
--opt="cutlass_attn=1;cutlass_norm=0;cublas_gemm=0;cudagraph=0".
""".strip(),
    "system_lib_prefix": """
Adding a prefix to all symbols exported. Similar to "objcopy --prefix-symbols".
This is useful when compiling multiple models into a single library to avoid symbol
conflicts. Different from objcopy, this takes no effect for shared library.
""".strip(),
    "context_window_size": """
Option to provide the maximum sequence length supported by the model.
This is usually explictly shown as context length or context window in the model card.
If this option is not set explicitly, by default, 
it will be determined by `context_window_size` or `max_position_embeddings` in `config.json`,
and the latter is usually inaccurate for some models.
""".strip(),
    "output_compile": """
The name of the output file. The suffix determines if the output file is a shared library or
objects. Available suffixes:
1) Linux: .so (shared), .tar (objects);
2) macOS: .dylib (shared), .tar (objects);
3) Windows: .dll (shared), .tar (objects);
4) Android, iOS: .tar (objects);
5) Web: .wasm (web assembly).
""".strip(),
    "source": """
The path to original model weight, infer from `config` if missing.
""".strip(),
    "source_format": """
The format of source model weight, infer from `config` if missing.
""".strip(),
    "output_quantize": """
The output directory to save the quantized model weight. Will create `params_shard_*.bin` and
`ndarray-cache.json` in this directory.
""".strip(),
    "conv_template": """
Conversation template. It depends on how the model is tuned. Use "LM" for vanilla base model
""".strip(),
    "output_gen_mlc_chat_config": """
The output directory for generated configurations, including `mlc-chat-config.json` and tokenizer
configuration.
""".strip(),
    "sliding_window": """
(Experimental) The sliding window size in sliding window attention (SWA).
This optional field overrides the `sliding_window` in config.json for
those models that use SWA. Currently only useful when compiling Mistral.
This flag subjects to future refactoring.
""".strip(),
    "prefill_chunk_size": """
(Experimental) The chunk size during prefilling. By default,
the chunk size is the same as sliding window or max sequence length.
This flag subjects to future refactoring.
""".strip(),
    "overrides": """
Model configuration override. Configurations to override `mlc-chat-config.json`.
Supports `context_window_size`, `prefill_chunk_size`, `sliding_window`, `max_batch_size`
and `num_shards`. Meanwhile, model config could be explicitly specified via details
knobs, e.g. --overrides "context_window_size=1024;prefill_chunk_size=128".
""".strip(),
}
