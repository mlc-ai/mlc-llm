# Build Qwen3-Omni text and audio components

These examples export a Qwen3 MoE Thinker, a CNN/attention audio encoder,
and a PCM frontend for Instruct, Thinking and Captioner. The supported
scope is text and standalone audio input with text output. Vision, video
positions and Qwen3 speech generation require additional components.

Use the toolchain setup in [README.md](README.md) and a matching TVM JavaScript
runtime. Install `safetensors` and `torch` for conversion. WebGPU expert
dispatch requires [mlc-ai/mlc-llm#3602](https://github.com/mlc-ai/mlc-llm/pull/3602).

Obtain a snapshot with `config.json`, `model.safetensors.index.json`,
`preprocessor_config.json`, and the shards containing `thinker.model.*`,
`thinker.lm_head.*` and `thinker.audio_tower.*`. The pinned configuration profiles are:

| Repository | Revision / configuration |
| --- | --- |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | [26291f7](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/26291f793822fb6be9555850f06dfe95f2d7e695/config.json) |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | [2f443cf](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking/blob/2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b/config.json) |
| `Qwen/Qwen3-Omni-30B-A3B-Captioner` | [a2bd106](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner/blob/a2bd106cbf527db5676e79662674da22b0545ec0/config.json) |

With the Instruct snapshot at `$SOURCE`:

```sh
REPO=Qwen/Qwen3-Omni-30B-A3B-Instruct
REVISION=26291f793822fb6be9555850f06dfe95f2d7e695
python examples/python/omni_audio/build_qwen3_thinker.py \
  --source "$SOURCE" --repo "$REPO" --revision "$REVISION" --out thinker
python examples/python/omni_audio/convert_qwen3_thinker.py \
  --source "$SOURCE" --repo "$REPO" --revision "$REVISION" \
  --compiled thinker --out thinker-weights
python examples/python/omni_audio/build_qwen3_encoder.py \
  --source "$SOURCE" --repo "$REPO" --revision "$REVISION" --out audio --weights
python examples/python/omni_audio/build_features.py \
  --source "$SOURCE" --out features --unpadded
```

Copy the converted Thinker cache and shards into `thinker/` before loading.
Conversion checks the compiled parameter schema, records source shard hashes,
and refuses to replace an existing cache. The caller supplies and verifies
the snapshot revision. Use separate output directories and each variant's
own weights, chat template and generation policy.

Both builders accept `--export-only` for CPU export checks and `--device metal`
for native libraries. The Thinker uses q4f16_1 at the source context limit.
Choose runtime context separately. Resource estimates count parameter bytes;
KV caches, temporary buffers and measured device memory are additional.

The PCM contract accepts finite mono 16 kHz audio, at least 201 samples long
and within the exported capacity. It reflects at the actual clip boundary
and returns floor(samples / 160) frames. The encoder uses 100-frame CNN chunks
with dynamic right-padding, then attention windows of at most 104 rows.
The host preserves valid lengths while concatenating chunks. These rules
follow the pinned [Transformers implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py).

The Thinker contract takes token IDs and optional audio embeddings, with audio
marker 151675 and EOS 151645. Captioner expects one audio clip and produces a
text caption. Thinking and Captioner disable audio output. Hosts apply their
checkpoint's template and policy when invoking these standalone contracts.
See the pinned [Captioner model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner/blob/a2bd106cbf527db5676e79662674da22b0545ec0/README.md).

The current checks cover component schemas and short exact-q4 token
comparisons. Full original-HF numerical and quality parity remain outside
that evidence.
