.. _model-artifact-manifest:

Model Artifact Manifest
=======================

The artifact manifest ties converted weights, a compiled model library and a
frontend together.  It is opt-in.  A model without one loads from
``mlc-chat-config.json`` as before.

The converted model directory contains ``mlc-model-manifest.json``.  A
compiled library carries the matching contract in ``_metadata.artifact``.
Both documents use one top-level ``schema_version`` and reject unknown fields.
``interface_id`` is a hash of the task description and
``parameter_schema_id`` a hash of the quantized parameter names, shapes and
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

Each key in ``exports`` is a role and the value is a function in the compiled
library.  A ``token_generation`` program declares ``embed_tokens``,
``create_kv_cache`` and one or both of the pairs below.  The four functions
in the table also take the KV cache and the parameters.  ``embed_tokens``
takes token IDs and the parameters.  ``create_kv_cache`` takes only its size
arguments.  ``total_len`` is the length of all sequences in the batch laid end
to end.

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

Gemma 4 declares the token pair because it looks up token IDs at every layer.
Other models declare the embedding pair and point it at their existing
``prefill`` and ``decode``.  A model may declare both when they give the same
result.  A modality ID is 0 for a text token and 1 for a position filled by an
adapter.

The frontend decodes the input, for example WAV to mono 16 kHz float32 PCM.
The compiled adapter does the feature extraction and projection.  The frontend
splits the adapter output to fit the prefill chunk size.

Image inputs
------------

An image input uses the ``image_decode`` processor.  LLaVA 1.5 declares it
like this:

.. code:: json

   {
     "processor": {
       "kind": "image_decode",
       "format": "rgb_u8",
       "layout": "nhwc",
       "resize": {"mode": "center_crop", "height": 336, "width": 336},
       "num_embeddings": 576
     },
     "adapter": "image",
     "prompt": {"placeholder_token_id": 32000}
   }

The frontend decodes the image, drops the alpha channel, resizes it and
passes a ``uint8`` tensor of shape ``[1, height, width, 3]`` to the adapter.
The ``resize`` object names the policy and the target size:

- ``stretch`` scales each axis on its own to the target size and ignores the
  aspect ratio.
- ``center_crop`` scales the image uniformly until it covers the target size,
  then keeps the centered ``height`` by ``width`` region.

The adapter converts the pixels to floating point, applies the mean and
standard deviation normalization, runs the vision tower and projects the
result.  It returns ``num_embeddings`` rows of shape
``[num_embeddings, hidden_size]``.  The count is fixed, so a frontend can
reserve the placeholder span and check the context window before it runs the
adapter.  The manifest does not name a resampling filter.  Frontends should
use bilinear or better.

Models that pick a resolution or a crop grid per image, such as
Phi-3.5-vision, cannot be described in this version.  Their embedding count
depends on the input, and the rule that derives it differs per model family.
They keep the ``image_embed`` path without a manifest until a resize mode is
defined for them.  One contiguous placeholder span is still enough for such
models when the adapter emits its row separators as embeddings.

Adding a processor kind does not change ``schema_version``.  The documents
keep the same fields, and a package that declares no image input produces the
same bytes and the same ``interface_id`` as before.  A frontend that does not
know ``image_decode`` rejects the package when it parses the processor.  That
is the intended failure for a contract it cannot honor.

LLaVA declares the embedding pair and points it at its existing ``prefill``
and ``decode``.  It exports no new functions, so the native engine and
released frontends keep working.

Compatibility and scope
-----------------------

WebLLM is the first manifest consumer.  Other MLC backends read
``mlc-chat-config.json`` and ignore the manifest.  The native engine cannot
serve Gemma 4 yet, since it does not pass token IDs to the model.  A model
without a manifest loads as before.  A manifest that is malformed or does not
match the library is an error.

Version 1 covers text and audio input for ``google/gemma-4-E2B-it``, text
and fixed-size image input for LLaVA, and text output.  Dynamic resolution
image input, video, compressed or remote audio, audio through the native
server, and speech-only pipelines are not included.
