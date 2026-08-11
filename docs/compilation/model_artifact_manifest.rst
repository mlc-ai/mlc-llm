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
       "manifest": "ndarray-cache.json",
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
           "prefill_prompt": "prefill_prompt",
           "decode_tokens": "decode_tokens",
           "create_kv_cache": "create_tir_paged_kv_cache"
         },
         "adapters": {"audio": "audio_embed"}
       }
     },
     "resources": {
       "required_features": ["shader-f16"],
       "max_storage_buffer_binding_size": 0,
       "estimated_device_memory_bytes": 0
     }
   }

``prefill_prompt`` consumes a canonical prompt bundle: embeddings with shape
``[1, sequence_length, hidden_size]``, token IDs with shape ``[1,
sequence_length]``, and modality IDs with the same shape.  The frontend owns
portable decoding (for example WAV to mono 16 kHz float32 PCM).  The compiled
adapter owns model-specific feature extraction and projection.  Adapter output
length is dynamic and frontends must chunk it to the compiled prefill limit.

Compatibility and scope
-----------------------

WebLLM is the first manifest consumer.  Other MLC backends continue to read
``mlc-chat-config.json`` and are unchanged; they do not gain audio ingestion
merely by seeing this sidecar.  Missing sidecars select the legacy path, while
a present but malformed or mismatched contract is an error.

Version 1 implements text and audio input for ``google/gemma-4-E2B-it`` and
text output.  Vision and video towers, remote or compressed audio, native
server audio ingestion, and speech-only/ASR pipelines are outside this
milestone.  Future canonical processors can reuse the task/adapter structure,
but each frontend must implement that canonical representation once.
