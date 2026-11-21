# Cached Talker component

This component projects caller-aligned Thinker states and generates codec IDs with
Qwen2.5-Omni's cached Talker. It exports codec embedding, the biased
Thinker-to-Talker projection, prefill, decode, cache creation and cache reset.
It is a standalone component, not a replacement for the top-level multimodal
`generate` API.

The loader follows the pinned
[3B checkpoint](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/tree/f75b40e3da2003cdd6e1829b1f420ca70797c34e),
[7B checkpoint](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/ae9e1690543ffd5c0221dc27f79834d0294cba00)
and [Transformers v5.17.0 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py).
It maps all 293 Talker checkpoint tensors to 173 FP32 MLC parameters for either
size.

## Build

Download the pinned model into a local source directory, then run with an MLC/TVM
build and, for WebGPU, the matching Emscripten toolchain:

```sh
hf download Qwen/Qwen2.5-Omni-3B \
  --revision f75b40e3da2003cdd6e1829b1f420ca70797c34e --local-dir ./omni-source
python examples/python/omni_audio/build_talker.py \
  --source ./omni-source --out ./talker-webgpu --device webgpu --weights
```

The library is compiled for the source's 32,768-token context and 256-token
prefill chunks. The verified browser requests use a 4,096-token runtime context;
compilation at the source maximum does not establish maximum-context allocation.
`--device metal` builds the native reference component. `--export-only` validates
and writes parameter/graph schemas without compiling a library. Generated
manifests alone are not a runnable package; the library and converted tensor
cache must also be present. The manifest's `estimated_device_memory_bytes` currently
counts parameter buffers only, excluding runtime KV cache and workspace; it is
not measured VRAM or an enforced memory budget.

## Caller alignment and generation

The `qwen2_5_omni.talker.v1` interface takes FP32 `prefix_embeddings` and
`reply_embeddings` and returns integer `code` values. The manifest declares a
width of 2,048 for 3B or 3,584 for 7B.
For the verified audio/text path, the prefix consists of prompt hidden states
plus token embeddings (audio-placeholder token embeddings zeroed only when actual
audio features were supplied), the actual
speaker BOS embedding, and the first forwarded reply hidden state plus embedding.
The factory adds codec PAD/BOS embeddings to the last two prefix positions.
Reply conditioning advances through the remaining forwarded reply states, then
text EOS/PAD embeddings; its final row repeats. The top-level HF first-EOS cut
must be applied before supplying these tensors. N sampled IDs have N−1 forwarded
reply states; retain the final sampled ID, including EOS, when determining that cut.

Use Thinker's opt-in `get_talker_spec()` exports to obtain normalized hidden
states while preserving the default chat ABI. The verified trace uses FP32
Thinker states and FP32 additions. FP16 Thinker token agreement did not satisfy
the hidden-state gate and is not interchangeable with that verified profile.
Vision and video positional alignment are outside this component contract.

The Web factory defaults to neutral greedy generation. Optional controlled
categorical sampling applies full-history repetition penalty, BOS suppression,
temperature, top-k, then top-p. The verified sampled fixture uses penalty 1.05,
temperature 0.9, top-k 40 and top-p 0.8 with explicit inverse-CDF uniforms. This
matches the categorical policy; it does not reproduce PyTorch's seeded random
number algorithm. Equal-score ties use token-ID order; floating-point filtering
near boundaries can differ from HF. Codec PAD/EOS terminate generation and are excluded from the
output; other unsupported codec IDs are rejected. Generation is bounded to 256
codes to fit the current 512-frame DiT component.

## Verification scope

For 3B, native and Chrome checks cover the pretrained projection, cached
prefill/decode, reset and logits. A completed ten-token Thinker answer,
including text EOS, produced 125 identical codec IDs and natural Talker EOS with both HF
conditioning and captured FP32 Chrome Thinker conditioning.

The composed reference-conditioning → DiT → waveform experiment produced 60,000
samples, with waveform RMSE 8.01e-6 and maximum absolute error 5.57e-4 against HF.
Nine samples exceeded the predeclared pointwise tolerance, so the composed
waveform gate remains failed. This is separate from the passing Talker codec
comparison; numeric parity alone is not an intelligibility evaluation.
