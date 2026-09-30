.. _model-artifact-manifest:

Model Artifact Manifest
=======================

The artifact manifest is an opt-in contract between converted weights, a
compiled model library, and a frontend.  Models without the sidecar keep the
legacy ``mlc-chat-config.json`` behavior.

The converted model directory contains ``mlc-model-manifest.json``.  A
compiled library carries the matching contract in ``_metadata.artifact``.
Both documents use one top-level ``schema_version`` and reject unknown fields.
The ``interface_id`` binds the public task description, while
``parameter_schema_id`` binds post-quantization parameter names, shapes, and
dtypes.

For the experimental Gemma 4 text-and-audio target, the package sidecar has
this shape (hashes are abbreviated here):

.. code:: json

   {
     "schema": "mlc.model-package",
     "schema_version": 1,
     "chat_config": "mlc-chat-config.json",
     "interface_id": "sha256:<64 lowercase hex digits>",
     "weights": {
       "manifest": "tensor-cache.json",
       "parameter_schema_id": "sha256:<64 lowercase hex digits>"
     },
     "tasks": {
       "chat.completions": {
         "executor": "generation",
         "inputs": {
           "text": {"processor": "tokenizer"},
           "audio": {
             "processor": {
               "kind": "audio_decode",
               "format": "pcm_f32",
               "sample_rate_hz": 16000,
               "channels": 1,
               "min_samples": 161,
               "max_samples": 480000
             },
             "adapter": "audio",
             "prompt": {
               "prefix_token_ids": [256000],
               "placeholder_token_id": 258881,
               "suffix_token_ids": [258883]
             }
           }
         },
         "output": "text"
       }
     }
   }

The compiled half names entrypoints by role rather than by model family:

.. code:: json

   {
     "schema": "mlc.compiled-program",
     "schema_version": 1,
     "interface_id": "sha256:<same interface hash>",
     "parameter_schema_id": "sha256:<same parameter hash>",
     "programs": {
       "generation": {
         "kind": "token_generation",
         "exports": {
           "embed_tokens": "embed",
           "prefill_tokens": "prefill_tokens",
           "decode_tokens": "decode_tokens",
           "create_kv_cache": "create_tir_paged_kv_cache"
         },
         "adapters": {"audio": "audio_embed"}
       }
     },
     "resources": {
       "required_features": ["shader-f16"],
       "max_storage_buffer_binding_size": 201326592,
       "estimated_device_memory_bytes": 2797972550
     }
   }

Both sizes are computed from the parameter shapes.  A named dimension such as
``vocab_size`` takes its value from the model config.

``estimated_device_memory_bytes`` is the total size of the model parameters.
It does not include the KV cache or anything the runtime allocates, so treat
it as a lower bound when picking a device.

Each key in ``exports`` is a role and each value is the name of a function in
the compiled library.  A role fixes the arguments the frontend passes, so the
function can have any name.  A ``token_generation`` program declares
``embed_tokens``, ``create_kv_cache`` and at least one of the two pairs below.
All four functions also take the KV cache and the parameters.

.. list-table::
   :header-rows: 1

   * - Role
     - Arguments
   * - ``prefill_tokens``
     - embeddings ``[1, total_len, hidden_size]``, token IDs ``[1, total_len]``,
       modality IDs ``[1, total_len]``
   * - ``decode_tokens``
     - token IDs ``[batch_size, 1]``
   * - ``prefill_embeds``
     - embeddings ``[1, total_len, hidden_size]``
   * - ``decode_embeds``
     - embeddings ``[1, 1, hidden_size]``

Prefill has no batch dimension.  Sequences are laid end to end along
``total_len`` and the KV cache is told where each one starts.
``decode_tokens`` takes one token per sequence and stacks them along
``batch_size``.  ``decode_embeds`` matches the existing ``decode`` export,
which takes one sequence.

A model declares the token pair when it needs the token IDs inside the model,
as Gemma 4 does for its per-layer embeddings.  Other models declare the
embedding pair and point it at their existing ``prefill`` and ``decode``.  A
model may declare both pairs when both give the same result, and a frontend
then calls the one it implements.  Half a pair is rejected.  A
modality ID is 0 for a text token and 1 for a position filled by an adapter.

The frontend decodes the input, for example WAV to mono 16 kHz float32 PCM.
The compiled adapter does the feature extraction and projection.  The number
of embeddings an adapter returns can vary, and the frontend splits them to fit
the compiled prefill limit.

Compatibility and scope
-----------------------

WebLLM is the first manifest consumer.  Other MLC backends continue to read
``mlc-chat-config.json`` and are unchanged; they do not gain audio ingestion
merely by seeing this sidecar.  Gemma 4 needs the token IDs next to the
embeddings at every layer, so it declares the token pair only and cannot be
served by the native engine yet.  Missing sidecars select the legacy path, while
a present but malformed or mismatched contract is an error.

Version 1 implements text and audio input for ``google/gemma-4-E2B-it`` and
text output.  Vision and video towers, remote or compressed audio, native
server audio ingestion, and speech-only/ASR pipelines are outside this
milestone.  Future canonical processors can reuse the task/adapter structure,
but each frontend must implement that canonical representation once.
