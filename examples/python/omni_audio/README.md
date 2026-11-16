# Build the audio-to-text components

These scripts build the three standalone components used by WebLLM's
`examples/omni-audio`: PCM features, audio encoding, and text generation.
They support Qwen2.5-Omni-3B text/audio positions, not image/video positions or
Talker speech output. Each module embeds its validated component contract and
parameter schema; the WebLLM example composes them at load time.

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

The opt-in speech profile exports the normalized states needed to align Thinker
text with Talker. It uses the verified FP32 profile and compiles at 32768 tokens:

```sh
python -m mlc_llm convert_weight dist/omni-reference \
  --model-type qwen2_5_omni_thinker_audio --quantization q0f32 \
  --device cpu -o dist/omni-thinker-speech
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

The default text/audio contract and its exports are unchanged. FP32 speech
weights require a storage-buffer binding of at least 1,244,659,712 bytes; the
resource estimate in the manifest covers weights only, excluding cache and
workspace. The compiled context is an upper bound, not a claim that the full
context fits on every adapter. See [Talker](TALKER.md),
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
