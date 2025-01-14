"""Help message for CLI arguments."""

HELP = {
    "config": (
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
"""
    ).strip(),
    "quantization": """
The quantization mode we use to compile. If unprovided, will infer from `model`.
""".strip(),
    "model": """
A path to ``mlc-chat-config.json``, or an MLC model directory that contains `mlc-chat-config.json`.
It can also be a link to a HF repository pointing to an MLC compiled model.
""".strip(),
    "model_lib": """
The full path to the model library file to use (e.g. a ``.so`` file). If unspecified, we will use
the provided ``model`` to search over possible paths. It the model lib is not found, it will be
compiled in a JIT manner.
""".strip(),
    "model_type": """
Model architecture such as "llama". If not set, it is inferred from `mlc-chat-config.json`.
""".strip(),
    "device_compile": """
The GPU device to compile the model to. If not set, it is inferred from GPUs available locally.
""".strip(),
    "device_quantize": """
The device used to do quantization such as "cuda" or "cuda:0". Will detect from local available GPUs
if not specified.
""".strip(),
    "device_deploy": """
The device used to deploy the model such as "cuda" or "cuda:0". Will detect from local
available GPUs if not specified.
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
--opt="cublas_gemm=1;cudagraph=0".
""".strip(),
    "system_lib_prefix": """
Adding a prefix to all symbols exported. Similar to "objcopy --prefix-symbols".
This is useful when compiling multiple models into a single library to avoid symbol
conflicts. Different from objcopy, this takes no effect for shared library.
""".strip(),
    "context_window_size": """
Option to provide the maximum sequence length supported by the model.
This is usually explicitly shown as context length or context window in the model card.
If this option is not set explicitly, by default,
it will be determined by `context_window_size` or `max_position_embeddings` in `config.json`,
and the latter is usually inaccurate for some models.
""".strip(),
    "output_compile": """
The path to the output file. The suffix determines if the output file is a shared library or
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
    "sliding_window_size": """
(Experimental) The sliding window size in sliding window attention (SWA).
This optional field overrides the `sliding_window_size` in config.json for
those models that use SWA. Currently only useful when compiling Mistral.
This flag subjects to future refactoring.
""".strip(),
    "prefill_chunk_size": """
(Experimental) The chunk size during prefilling. By default,
the chunk size is the same as sliding window or max sequence length.
This flag subjects to future refactoring.
""".strip(),
    "attention_sink_size": """
(Experimental) The number of stored sinks. Only supported on Mistral yet. By default,
the number of sinks is 4. This flag subjects to future refactoring.
""".strip(),
    "max_batch_size": """
The maximum allowed batch size set for the KV cache to concurrently support.
""".strip(),
    """tensor_parallel_shards""": """
Number of shards to split the model into in tensor parallelism multi-gpu inference.
""".strip(),
    """pipeline_parallel_stages""": """
Number of pipeline stages to split the model layers for pipeline parallelism.
""".strip(),
    """disaggregation""": """
Whether enable disaggregation when compiling the model.
""".strip(),
    "overrides": """
Model configuration override. Configurations to override `mlc-chat-config.json`. Supports
`context_window_size`, `prefill_chunk_size`, `sliding_window_size`, `attention_sink_size`,
`max_batch_size` and `tensor_parallel_shards`. Meanwhile, model config could be explicitly
specified via details knobs, e.g. --overrides "context_window_size=1024;prefill_chunk_size=128".
""".strip(),
    "modelconfig_overrides": """
Model configuration override. Supports overriding,
`context_window_size`, `prefill_chunk_size`, `sliding_window_size`, `attention_sink_size`,
`max_num_sequence` and `tensor_parallel_shards`. The overrides could be explicitly
specified via details knobs, e.g. --overrides "context_window_size=1024;prefill_chunk_size=128".
""".strip(),
    "debug_dump": """
Specifies the directory where the compiler will store its IRs for debugging purposes
during various phases of compilation. By default, this is set to `None`, indicating
that debug dumping is disabled.
""".strip(),
    "prompt": """
The prompt of the text generation.
""".strip(),
    "generate_length": """
The target length of the text generation.
""".strip(),
    "max_total_sequence_length_serve": """
The KV cache total token capacity, i.e., the maximum total number of tokens that
the KV cache support. This decides the GPU memory size that the KV cache consumes.
If not specified, system will automatically estimate the maximum capacity based
on the vRAM size on GPU.
""".strip(),
    "prefill_chunk_size_serve": """
The maximum number of tokens the model passes for prefill each time.
It should not exceed the prefill chunk size in model config.
If not specified, this defaults to the prefill chunk size in model config.
""".strip(),
    "max_history_size_serve": """
The maximum history length for rolling back the RNN state.
If unspecified, the default value is 1.
KV cache does not need this.
""".strip(),
    "enable_tracing_serve": """
Enable Chrome Tracing for the server.
After enabling, you can send POST request to the "debug/dump_event_trace" entrypoint
to get the Chrome Trace. For example,
"curl -X POST http://127.0.0.1:8000/debug/dump_event_trace -H "Content-Type: application/json" -d '{"model": "dist/llama"}'"
""".strip(),
    "mode_serve": """
The engine mode in MLC LLM. We provide three preset modes: "local", "interactive" and "server".
The default mode is "local".
The choice of mode decides the values of "max_num_sequence", "max_total_seq_length" and
"prefill_chunk_size" when they are not explicitly specified.
1. Mode "local" refers to the local server deployment which has low request concurrency.
   So the max batch size will be set to 4, and max total sequence length and prefill chunk size
   are set to the context window size (or sliding window size) of the model.
2. Mode "interactive" refers to the interactive use of server, which has at most 1 concurrent
   request. So the max batch size will be set to 1, and max total sequence length and prefill
   chunk size are set to the context window size (or sliding window size) of the model.
3. Mode "server" refers to the large server use case which may handle many concurrent request
   and want to use GPU memory as much as possible. In this mode, we will automatically infer
   the largest possible max batch size and max total sequence length.
You can manually specify arguments "max_num_sequence", "max_total_seq_length" and
"prefill_chunk_size" via "--overrides" to override the automatic inferred values.
For example: --overrides "max_num_sequence=32;max_total_seq_length=4096"
""".strip(),
    "additional_models_serve": """
The model paths and (optional) model library paths of additional models (other than the main model).
When engine is enabled with speculative decoding, additional models are needed.
The way of specifying additional models is:
"--additional-models model_path_1 model_path_2 ..." or
"--additional-models model_path_1,model_lib_1 model_path_2 ...".
When the model lib of a model is not given, JIT model compilation will be activated
to compile the model automatically.
""".strip(),
    "gpu_memory_utilization_serve": """
A number in (0, 1) denoting the fraction of GPU memory used by the server in total.
It is used to infer to maximum possible KV cache capacity.
When it is unspecified, it defaults to 0.85.
Under mode "local" or "interactive", the actual memory usage may be significantly smaller than
this number. Under mode "server", the actual memory usage may be slightly larger than this number.
""".strip(),
    "speculative_mode_serve": """
The speculative decoding mode. Right now four options are supported:
 - "disable", where speculative decoding is not enabled,
 - "small_draft", denoting the normal speculative decoding (small draft) style,
 - "eagle", denoting the eagle-style speculative decoding.
 - "medusa", denoting the medusa-style speculative decoding.
The default mode is "disable".
""".strip(),
    "spec_draft_length_serve": """
The number of draft tokens to generate in speculative proposal.
Being 0 means to enable adaptive speculative mode, where the draft length will be
automatically adjusted based on engine state. The default values is 0.
""".strip(),
    "prefix_cache_mode_serve": """
The prefix cache mode. Right now two options are supported:
 - "disable", where prefix cache is not enabled,
 - "radix", denoting the normal paged radix tree based prefix cache,
The default mode is "radix".
""".strip(),
    "prefix_cache_max_num_recycling_seqs_serve": """
The maximum number of sequences in prefix cache, default as max_batch_size.
And set 0 to disable prefix cache, set -1 to have infinite capacity prefix cache.
""".strip(),
    "prefill_mode": """
The prefill mode. "chunked" means the basic prefill with chunked input enabled. "hybrid" means the
hybrid prefill or split-fuse, so that decode step will be converted into prefill.
""".strip(),
    "overrides_serve": """
Overriding extra configurable fields of EngineConfig and model compilation config.
Supporting fields that can be be overridden: "tensor_parallel_shards", "max_num_sequence",
"max_total_seq_length", "prefill_chunk_size", "max_history_size", "gpu_memory_utilization",
"spec_draft_length", "prefix_cache_max_num_recycling_seqs", "context_window_size",
"sliding_window_size", "attention_sink_size".
Please check out the documentation of EngineConfig in mlc_llm/serve/config.py for detailed docstring
of each field.
Example: --overrides "max_num_sequence=32;max_total_seq_length=4096;tensor_parallel_shards=2"
""".strip(),
    "config_package": """
The path to "mlc-package-config.json" which is used for package build.
See "https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCChat/mlc-package-config.json" as an example.
""".strip(),
    "mlc_llm_source_dir": """
The source code path to MLC LLM.
""".strip(),
    "output_package": """
The path of output directory for the package build outputs.
""".strip(),
    "calibration_dataset": """
The path to the calibration dataset.
    """.strip(),
    "num_calibration_samples": """
The number of samples used for calibration.
    """.strip(),
    "output_calibration": """
The output directory to save the calibration params.
    """.strip(),
    "seed_calibrate": """
The seed to sample the calibration dataset.""",
    "pd_balance_factor": """
How much prefill to move to decode engine. For example,
0.1 means the last 10 percent tokens are prefilled by decode engine.
    """.strip(),
}
