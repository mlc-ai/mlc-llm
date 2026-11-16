# Build the audio-to-text components

These scripts build the three standalone components used by WebLLM's
`examples/omni-audio`: PCM features, audio encoding, and text generation.
The example below uses Qwen2.5-Omni-3B for text generation from text or audio.
Each module embeds its component contract and parameter schema. The WebLLM
example composes these components at load time.

Use an MLC checkout containing these components, its matching TVM build, and an
activated Emscripten environment. The TVM JavaScript runtime must come from that
same TVM build. WebGPU builds of the reference conditioner require the scheduler
fix in [apache/tvm#20985](https://github.com/apache/tvm/pull/20985). Install `safetensors`, `torch`, `transformers`, and
`huggingface_hub` in the compiler environment. The reference fixtures and frontend
were verified with `transformers==5.17.0`; use that version for reproduction. Run these commands from the MLC
repository root:

```sh
hf download Qwen/Qwen2.5-Omni-3B \
  --revision f75b40e3da2003cdd6e1829b1f420ca70797c34e \
  --local-dir dist/omni-reference
python examples/python/omni_audio/build_features.py \
  --source dist/omni-reference --out dist/omni-features
python examples/python/omni_audio/build_encoder.py \
  --source dist/omni-reference --out dist/omni-audio --weights
python -m mlc_llm convert_weight dist/omni-reference \
  --model-type qwen2_5_omni_thinker_audio --quantization q0f16 \
  --device cpu -o dist/omni-thinker-f32accum-q0f16
python examples/python/omni_audio/build_thinker.py \
  --source dist/omni-reference --out dist/omni-thinker-f32accum-q0f16
```

The Thinker converter preserves float32 projection biases required by its
float32 accumulation path. Reconvert weights when changing that parameter
schema; an older float16-bias cache is incompatible. The component uses a
4096-token context and 256-token prefill chunks. Its `q0f16` weights are
unquantized; the audio encoder uses float32 weights.

Copy or symlink the three output directories and `omni-reference/tokenizer.json`
into the WebLLM example's `public/artifacts/` directory, preserving names. The
example README describes starting the server. These are local developer
artifacts; no hosted model URL is required. `--device metal` builds native
modules for comparison. `--tiny` on the encoder or Thinker builds synthetic
fixtures only and must not be used as pretrained-model evidence.

The pinned upstream [configuration](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/config.json),
[processor](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/preprocessor_config.json),
and [chat template](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/chat_template.json)
are the source of truth for this example.

## Thinker states for speech

The opt-in FP32 speech profile exports the normalized states needed to align
Thinker text with Talker. It compiles at 32,768 tokens for Qwen2.5-Omni-3B and
[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/ae9e1690543ffd5c0221dc27f79834d0294cba00).
Use the same checkpoint for the audio encoder, Thinker and Talker.

```sh
python -m mlc_llm convert_weight dist/omni-reference \
  --model-type qwen2_5_omni_thinker_audio --quantization q0f32 \
  --encode-format raw --device cpu -o dist/omni-thinker-speech
python examples/python/omni_audio/build_thinker.py \
  --source dist/omni-reference --out dist/omni-thinker-speech \
  --quantization q0f32 --talker
```

This emits the `qwen2_5_omni.thinker_audio_talker.v1` contract. The WebLLM host
aligns actual prompt and reply states and returns `prefix_embeddings` and
`reply_embeddings` for the Talker inputs, alongside streamed text token IDs.
`finish_reason` is 0 for a text EOS and 1 for a token/context bound; reaching a
bound is not a naturally completed answer. The metadata records the supported
speaker BOS IDs and Talker text EOS/PAD IDs from the pinned source assets. The
speech profile samples over the full Thinker vocabulary, as the HF reference
does. `--export-only` validates and writes schemas without compiling a library.

The largest FP32 Thinker buffer is 1,244,659,712 bytes for 3B and 2,179,989,504
bytes for 7B. Raw encoding lets the WebGPU runtime upload weights directly.
BF16-packed weights require a CPU staging tensor that exceeds the default
2 GiB WebAssembly memory limit for 7B. The manifest estimates parameter memory.
Runtime KV cache and workspace require additional memory. See [Talker](TALKER.md),
[flow predictor](FLOW_PREDICTOR.md), and [waveform decoder](WAVEFORM_DECODER.md)
for the subsequent components and their numerical verification boundaries.

Export the pinned voice assets for the speech example:

```sh
python examples/python/omni_audio/export_speakers.py \
  --source dist/omni-reference/spk_dict.pt --out dist/omni-speakers
```

The exporter checks the pinned source hash before loading with
`torch.load(weights_only=True)`, validates both voices and writes portable
`speakers.json` plus a source/output hash receipt. The selected voice's BOS,
reference mel and speaker embedding must be used together.
