# Standalone waveform decoder

`build_vocoder.py` builds the complete pretrained BigVGAN stage from
Qwen2.5-Omni-3B. Its named `log_mel` input contains synthesis features with shape
`[80, frames]`; its `pcm` output contains `240 * frames` float32 samples at 24 kHz.
These features are different from the 128-bin Whisper features used for audio
input. The existing `waveform_decoder` component kind and v2 contract describe
this stateless stage without adding a global model-specific API.

Talker codec generation and DiT codec-to-mel sampling are separate stages. This
component does not turn text or codec tokens into speech by itself. It operates
on complete bounded clips; independently decoded adjacent clips are not a valid
streaming/chunking implementation because the convolution boundaries differ.

Using the pinned source directory from the companion README:

```sh
python examples/python/omni_audio/build_vocoder.py \
  --source dist/omni-reference --out dist/omni-vocoder \
  --weights --device metal --max-frames 32
```

The output includes the native library, a raw float32 tensor cache, parameter
metadata and validated model/compiled manifests. `run` maps to `forward`, whose
VM arguments are `log_mel` and the packed parameter list. Callers must reject
nonfinite inputs and frame lengths outside `1..max_mel_frames` before invocation.
The generated metadata records that bound, sample rate, samples per frame and
hashes of the builder/model/loader source files. Model parameters are mapped
exactly from `token2wav.code2wav_bigvgan_model.*`; non-persistent anti-alias FIR
coefficients are generated separately.

The source of truth is the pinned
[model configuration](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/config.json)
and the independent
[Transformers 5.17.0 BigVGAN implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py).
The reference verification records the installed implementation SHA256
`976d3ac0cf130b70f52f7fa5a21ae4913aa88554afc74dd42962087ee6eaa5e2`.
The official
[Qwen inference example](https://github.com/QwenLM/Qwen2.5-Omni#quickstart)
uses a 24 kHz output sample rate. The CPU tests compare real compiled kernels
against that implementation on small models, including replicate boundaries,
upsampling, filter coefficients and mel normalization.
