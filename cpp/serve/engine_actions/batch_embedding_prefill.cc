/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/batch_embedding_prefill.cc
 * \brief The batch embedding prefill action for encoder models.
 *
 * This action implements the encoder embedding lane:
 * - FIFO scan of the embedding waiting queue (not unordered_map)
 * - Batch-local dynamic padding
 * - Encoder prefill (no KV cache, no TokenEmbed)
 * - Device-side pooling (CLS / Mean / Last)
 * - Request-level aggregation and callback
 */
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../data.h"
#include "../engine.h"
#include "../engine_state.h"
#include "../model.h"
#include "action.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The action that processes embedding requests from the embedding waiting queue.
 *
 * Design:
 * - Scheduling granularity is per-item (not per-request).
 * - Batching: FIFO scan of embedding_waiting_queue (the source of truth for order).
 * - Admit condition: batch_size <= max_num_sequence AND batch_size * max_len <= prefill_chunk_size.
 * - Encoder execution: directly calls prefill(input_ids, attention_mask, params).
 *   input_ids: [batch, max_len], attention_mask: [batch, max_len] — both int32, 2D.
 * - No KV cache, no TokenEmbed, no speculative/prefix-cache/grammar/disagg.
 * - Pooling on device, only pooled [batch, hidden] copied to host as float32.
 * - Callback sends request-level aggregated EmbeddingResult.
 */
class BatchEmbeddingPrefillActionObj : public EngineActionObj {
 public:
  explicit BatchEmbeddingPrefillActionObj(Model model, EngineConfig engine_config)
      : model_(std::move(model)), engine_config_(std::move(engine_config)) {}

  Array<Request> Step(EngineState estate) final {
    // This action does not interact with the chat request path.
    // It returns an empty array since chat post-processing doesn't apply.
    // The embedding results are delivered through the embedding callback.

    if (estate->embedding_waiting_queue.empty()) {
      return {};
    }

    NVTXScopedRange nvtx_scope("BatchEmbeddingPrefill");

    int max_batch = engine_config_->max_num_sequence;
    int64_t max_total_tokens = engine_config_->prefill_chunk_size;

    // Collect items from the waiting queue using FIFO order.
    // CRITICAL: iterate embedding_waiting_queue (ordered), NOT the unordered_map.
    struct ScheduledItem {
      EmbeddingRequestState* state;
      int item_idx;  // index into state->request->items
    };
    std::vector<ScheduledItem> batch_items;
    int candidate_max_len = 0;

    for (const EmbeddingRequest& req : estate->embedding_waiting_queue) {
      auto it = estate->embedding_request_states.find(req->id);
      if (it == estate->embedding_request_states.end()) continue;
      EmbeddingRequestState& state = it->second;

      // Batch must be homogeneous in pooling_strategy/normalize; stop (not skip) on
      // mismatch to preserve FIFO order.
      if (!batch_items.empty()) {
        const auto& head_req = batch_items[0].state->request;
        if (state.request->pooling_strategy != head_req->pooling_strategy ||
            state.request->normalize != head_req->normalize) {
          goto done_collecting;
        }
      }

      int num_items = static_cast<int>(state.request->items.size());
      for (int i = state.completed_items; i < num_items; ++i) {
        int item_len = static_cast<int>(state.request->items[i].token_ids.size());
        int new_max_len = std::max(candidate_max_len, item_len);
        int new_batch_size = static_cast<int>(batch_items.size()) + 1;

        // Admit condition.
        if (new_batch_size > max_batch) goto done_collecting;
        if (static_cast<int64_t>(new_batch_size) * new_max_len > max_total_tokens)
          goto done_collecting;

        batch_items.push_back({&state, i});
        candidate_max_len = new_max_len;
      }
    }
  done_collecting:

    if (batch_items.empty()) {
      return {};
    }

    int batch_size = static_cast<int>(batch_items.size());
    int max_len = candidate_max_len;

    // Build padded input_ids and attention_mask on host.
    // BERT contract: input_ids [batch_size, max_len] int32, attention_mask [batch_size, max_len] int32.
    Tensor input_ids_host =
        Tensor::Empty({batch_size, max_len}, DataType::Int(32), Device{DLDeviceType::kDLCPU, 0});
    Tensor attention_mask_host =
        Tensor::Empty({batch_size, max_len}, DataType::Int(32), Device{DLDeviceType::kDLCPU, 0});

    int32_t* ids_ptr = static_cast<int32_t*>(input_ids_host->data);
    int32_t* mask_ptr = static_cast<int32_t*>(attention_mask_host->data);

    std::vector<int> real_lengths(batch_size);

    for (int b = 0; b < batch_size; ++b) {
      const auto& item = batch_items[b].state->request->items[batch_items[b].item_idx];
      int seq_len = static_cast<int>(item.token_ids.size());
      real_lengths[b] = seq_len;

      // Fill token IDs (right-padded with 0).
      for (int s = 0; s < seq_len; ++s) {
        ids_ptr[b * max_len + s] = item.token_ids[s];
      }
      for (int s = seq_len; s < max_len; ++s) {
        ids_ptr[b * max_len + s] = 0;
      }

      // Fill attention mask (1 for real tokens, 0 for padding).
      for (int s = 0; s < seq_len; ++s) {
        mask_ptr[b * max_len + s] = 1;
      }
      for (int s = seq_len; s < max_len; ++s) {
        mask_ptr[b * max_len + s] = 0;
      }
    }

    // Run encoder prefill, then pool directly into a CPU float32 buffer.
    int pooling_strategy = static_cast<int>(batch_items[0].state->request->pooling_strategy);
    ObjectRef hidden_states =
        model_->EncoderPrefill(input_ids_host, attention_mask_host, batch_size, max_len);
    Tensor pooled_cpu = model_->PoolEncoderHiddenStates(
        hidden_states, real_lengths, batch_size, max_len, pooling_strategy);
    int hidden_dim = static_cast<int>(pooled_cpu->shape[1]);

    // L2 normalize on the CPU float32 buffer if requested.
    bool normalize = batch_items[0].state->request->normalize;
    if (normalize) {
      float* data = static_cast<float*>(pooled_cpu->data);
      for (int b = 0; b < batch_size; ++b) {
        float norm = 0.0f;
        for (int h = 0; h < hidden_dim; ++h) {
          float v = data[b * hidden_dim + h];
          norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
          for (int h = 0; h < hidden_dim; ++h) {
            data[b * hidden_dim + h] /= norm;
          }
        }
      }
    }

    // Write pooled results back into per-request result buffers and track completion.
    int float_bytes = sizeof(float);

    for (int b = 0; b < batch_size; ++b) {
      auto* state = batch_items[b].state;
      int item_idx = batch_items[b].item_idx;
      int original_item_index = state->request->items[item_idx].item_index;

      // Copy this item's embedding into the request's result buffer at the correct row.
      float* src = static_cast<float*>(pooled_cpu->data) + b * hidden_dim;
      float* dst = static_cast<float*>(state->result_buffer->data) +
                   original_item_index * hidden_dim;
      std::memcpy(dst, src, hidden_dim * float_bytes);

      // Update prompt tokens.
      state->prompt_tokens += real_lengths[b];
      state->completed_items++;
    }

    // Completed requests form a prefix of the waiting queue under current scheduling;
    // break on the first pending request to keep that invariant explicit.
    Array<EmbeddingResult> completed_results;
    size_t completed_prefix_len = 0;

    for (const EmbeddingRequest& req : estate->embedding_waiting_queue) {
      auto it = estate->embedding_request_states.find(req->id);
      if (it == estate->embedding_request_states.end()) break;
      EmbeddingRequestState& state = it->second;
      int total_items = static_cast<int>(state.request->items.size());
      if (state.completed_items < total_items) break;
      completed_results.push_back(
          EmbeddingResult(state.request->id, state.result_buffer, state.prompt_tokens));
      ++completed_prefix_len;
    }

    for (size_t i = 0; i < completed_prefix_len; ++i) {
      estate->embedding_request_states.erase(estate->embedding_waiting_queue[i]->id);
    }
    estate->embedding_waiting_queue.erase(
        estate->embedding_waiting_queue.begin(),
        estate->embedding_waiting_queue.begin() + completed_prefix_len);

    // Fire the embedding callback if we have results.
    if (!completed_results.empty() && estate->embedding_request_callback_ != nullptr) {
      estate->embedding_request_callback_(completed_results);
    }

    // Return empty — embedding lane doesn't feed into chat post-processing.
    return {};
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BatchEmbeddingPrefillActionObj>();
  }

  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("mlc.serve.BatchEmbeddingPrefillAction",
                              BatchEmbeddingPrefillActionObj, EngineActionObj);

 private:
  /*! \brief The encoder model. */
  Model model_;
  /*! \brief Engine config for batching constraints. */
  EngineConfig engine_config_;
};

TVM_FFI_STATIC_INIT_BLOCK() { BatchEmbeddingPrefillActionObj::RegisterReflection(); }

EngineAction EngineAction::BatchEmbeddingPrefill(Model model, EngineConfig engine_config) {
  return EngineAction(
      tvm::ffi::make_object<BatchEmbeddingPrefillActionObj>(std::move(model),
                                                            std::move(engine_config)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
