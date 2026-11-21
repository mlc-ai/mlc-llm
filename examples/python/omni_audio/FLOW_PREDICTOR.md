# Standalone flow predictor

`build_dit.py` builds the Qwen2.5-Omni DiT flow predictor, with a bounded frame
capacity. It exports `forward(noise, time, code, reference, speaker, packed_params)`
and returns a float32 velocity tensor with the same shape as `noise`.

Inputs are explicit: `noise[frames, 80]`, `time[1]`, codec IDs `code[codes]`,
precomputed ECAPA `reference[128]`, and speaker vector `speaker[192]`. All inputs
except the int32 codec IDs are float32. Callers must validate finite inputs,
`frames == repeats * codes`, codec IDs in `0..num_embeds`, time in `[0, 1]` and
`1 <= frames <= max_mel_frames` before invoking the VM. The raw tensor ABI has
independent frame/code dimensions; its kernels assume these host checks.

```sh
python examples/python/omni_audio/build_dit.py \
  --source dist/omni-reference --out dist/omni-flow \
  --weights --device metal --max-frames 32
```

The builder rejects capacities beyond the checkpoint's positional limit and
capacities that are not a multiple of codec repeats. The generated manifest uses
the existing `transform` kind and `qwen2_5_omni.flow_predictor.v1` contract. It
records all 319 learned flow tensors and a generated float32 RoPE frequency buffer; the 40 ECAPA reference encoder parameters are
separate and intentionally excluded. HF 5.17 recomputes its non-persistent RoPE
frequencies; the loader follows that behavior instead of using the legacy
BF16-rounded buffer retained in the checkpoint.

`dit_sampling.sample_flow` implements the pinned RK4 3/8 integrator with explicit
initial noise, time points and guidance scale. Pass the compiled `max_mel_frames`
as `max_frames`, and the compiled `repeats` and `num_embeds`. It validates both
conditional and null predictions independently before guidance arithmetic.
The null reference must be the actual ECAPA output for zero reference mels; it
is not generally a zero vector. Time warping and random noise generation remain
explicit caller choices. The returned `[frames, 80]` mel array must be transposed
to `[80, frames]` for the standalone waveform decoder.

Source of truth: the [pinned model configuration](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/config.json)
and [Transformers 5.17.0 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py).
The implementation preserves first-head-only interleaved RoPE and per-layer
block attention masks. Tests compare compiled forward and solver execution with
HF, including full HF forward calls using its real ECAPA conditioner.

Talker and the ECAPA reference-mel encoder are not implemented by this component.
It does not provide a complete text-to-speech pipeline or streaming audio.
