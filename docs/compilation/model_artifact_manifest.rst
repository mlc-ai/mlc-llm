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

A model that keeps a recurrent state next to the KV cache, such as Qwen3.5,
also declares ``create_rnn_state``.  Its prefill and decode functions then
take the KV cache and the recurrent state, in that order, before the
parameters.

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
When the compiled library needs the same values in a wider type, its program
lists the adapter under ``adapter_dtypes``.  WebGPU has no 8 bit storage
type, so a WebGPU library takes the pixels as ``uint32``.
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
and fixed-size image input for LLaVA, Gemma 3 and Qwen3.5, and text output.
Dynamic resolution image input, video, compressed or remote audio, audio
through the native server, and speech-only pipelines are not included.

Multi-component contracts (schema version 2)
-------------------------------------------

Version 2 describes a graph of components with named inputs and outputs.  It is
an experimental validation and planning contract: the compiler does not yet
emit executable graphs, and normal WebLLM model loading rejects graphs before loading model
configurations, libraries or weights.  WebLLM also provides an experimental host
scheduler with caller-supplied component backends; this does not supply real
model kernels or make these packages loadable through MLCEngine.  Parsing a manifest does not establish
support for a model or a media format.  Version 1 manifests and their interface
hashes retain their existing interpretation.

A package declares ``components`` and ``tasks`` instead of a single global
weight set and executor.  Each component has a versioned ``contract``, typed
input and output ports, its own weight-manifest and parameter-schema reference
(or explicit ``null`` for a parameterless component), and request-local state.
Component kinds describe behavior: ``encoder``, ``autoregressive``,
``code_predictor``, ``waveform_decoder`` and ``transform``.  They do not select
model-specific runtime implementations.

Task inputs and outputs are named maps.  Text, audio, image and video inputs can
therefore coexist in one task, with both text and audio outputs.  A public input
binds to one or more component input ports; a public output binds to a component
output port.  Explicit edges connect component outputs to inputs.  A port
specifies its versioned semantic contract, dtype, shape and whether delivery is
a value or a stream.  Position grids, timestamps, valid lengths and conditioning
must have their own typed ports when the model requires them.  Declaring a PCM
port alone does not implement feature extraction or audio/video alignment.

Port compatibility requires exact agreement, including symbolic dimension
names.  Every active component input must be bound exactly once.  Graphs reject
missing endpoints, duplicate destinations, dead branches and cycles.  Recurrent
transitions remain inside a component's state rather than becoming graph edges.
A component output may feed both a public output and another component, allowing
text generation and speech conditioning to share the same producer.

Outputs declare whether they are optional.  The planning helper computes the
reverse dependency closure of the selected outputs and returns components in
topological order.  With a text producer feeding an optional speech branch,
selecting only text omits the speech components.  Selecting text and audio keeps
both branches.  Required outputs cannot be silently dropped.  This plan does
not allocate resources, schedule kernels or implement stream cancellation,
backpressure or final flushing.

The compiled contract repeats each component specification and adds a module
path, export-role mapping and resource requirements.  Package and compiled
component specifications must match exactly.  Each component declares a
``run`` export; stateful components also declare ``create_state`` and
``reset_state``.  This validates declarations, not the existence or ABI of an
export in a binary.  Multiple components may name one module.  Per-component
memory estimates do not establish peak simultaneous residency.

Version 2's ``interface_id`` hashes the complete normalized package except the
``interface_id`` field itself, including weight-manifest paths.  Object keys are
sorted recursively, arrays preserve order, and JSON uses compact separators.
The contract restricts identifiers and paths to ASCII and numbers to safe
integers so Python and JavaScript produce the same bytes.  Relocating a declared
weight path changes this identity.  Unknown fields and unsafe relative paths
are rejected rather than ignored.  The separate version 1 tasks-only hash is
unchanged.
