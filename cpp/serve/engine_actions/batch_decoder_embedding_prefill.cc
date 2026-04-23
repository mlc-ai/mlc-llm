/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/batch_decoder_embedding_prefill.cc
 * \brief The batch embedding prefill action for decoder-only models.
 *
 * This action implements the decoder embedding lane:
 * - FIFO scan of the embedding waiting queue
 * - Batch-local dynamic left-padding
 * - Decoder embedding prefill (no KV cache, no TokenEmbed)
 * - Device-side last-token pool at buffer row [max_len - 1]
 * - Request-level aggregation and callback
 *
 * Structurally parallel to BatchEmbeddingPrefillAction (encoder lane). The
 * only semantic differences are:
 *  - Inputs are left-padded rather than right-padded. Real tokens occupy
 *    ``[max_len - seq_len, max_len)``; padding sits on the left.
 *  - Pooling is always kLast. Under left padding the last real token sits
 *    at buffer row ``max_len - 1`` for every item in the batch, so we pass
 *    ``lengths[i] == max_len`` to PoolEmbeddingHiddenStates and let it pick
 *    that final row uniformly.
 *  - Calls Model::DecoderEmbeddingPrefill instead of Model::EncoderPrefill.
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

namespace {

struct ScheduledItem {
  EmbeddingRequestState* state;
  int item_idx;  // index into state->request->items
};

struct CollectedBatch {
  std::vector<ScheduledItem> items;
  int max_len = 0;
};

/*!
 * \brief Scan the embedding waiting queue in FIFO order and collect items to
 * form the next decoder-embedding batch. Mirrors CollectBatchItems in the
 * encoder lane: stops on batch-size / token-budget / pooling mismatch, and
 * always admits at least one item so an oversized request cannot deadlock
 * the lane.
 */
CollectedBatch CollectBatchItems(EngineState estate, int max_batch,
                                 int64_t max_total_tokens) {
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
        return {std::move(batch_items), candidate_max_len};
      }
    }

    int num_items = static_cast<int>(state.request->items.size());
    for (int i = state.completed_items; i < num_items; ++i) {
      int item_len = static_cast<int>(state.request->items[i].token_ids.size());
      int new_max_len = std::max(candidate_max_len, item_len);
      int new_batch_size = static_cast<int>(batch_items.size()) + 1;

      // Admit condition. Always admit at least one item, even if it exceeds
      // the soft token budget, to prevent a single oversized item from
      // deadlocking the embedding lane.
      if (new_batch_size > max_batch) {
        return {std::move(batch_items), candidate_max_len};
      }
      if (!batch_items.empty() &&
          static_cast<int64_t>(new_batch_size) * new_max_len > max_total_tokens) {
        return {std::move(batch_items), candidate_max_len};
      }

      batch_items.push_back({&state, i});
      candidate_max_len = new_max_len;
    }
  }

  return {std::move(batch_items), candidate_max_len};
}

}  // namespace

/*!
 * \brief The action that processes embedding requests from the embedding waiting queue
 * for decoder-only embedding models (e.g. Qwen3-Embedding).
 *
 * Design:
 *  - Scheduling granularity is per-item, matching the encoder lane.
 *  - Batching: FIFO scan of embedding_waiting_queue (the source of truth for order).
 *  - Admit condition: batch_size <= max_num_sequence AND
 *                     batch_size * max_len <= prefill_chunk_size.
 *  - Decoder execution: calls Model::DecoderEmbeddingPrefill, which wraps the compiled
 *    ``prefill_embedding(input_ids, attention_mask, params)`` function. No KV cache,
 *    no TokenEmbed, no speculative / prefix-cache / grammar / disagg.
 *  - Pooling runs inside Model::PoolEmbeddingHiddenStates (kLast strategy); only the
 *    pooled ``[batch, hidden]`` buffer is materialized on host as float32.
 *  - Callback sends request-level aggregated EmbeddingResult, identical to encoder.
 */
class BatchDecoderEmbeddingPrefillActionObj : public EngineActionObj {
 public:
  explicit BatchDecoderEmbeddingPrefillActionObj(Model model, EngineConfig engine_config)
      : model_(std::move(model)), engine_config_(std::move(engine_config)) {}

  Array<Request> Step(EngineState estate) final {
    if (estate->embedding_waiting_queue.empty()) {
      return {};
    }

    NVTXScopedRange nvtx_scope("BatchDecoderEmbeddingPrefill");

    int max_batch = engine_config_->max_num_sequence;
    int64_t max_total_tokens = engine_config_->prefill_chunk_size;

    // Collect items from the waiting queue using FIFO order.
    // CRITICAL: iterate embedding_waiting_queue (ordered), NOT the unordered_map.
    CollectedBatch batch = CollectBatchItems(estate, max_batch, max_total_tokens);
    if (batch.items.empty()) {
      return {};
    }
    auto& batch_items = batch.items;

    int batch_size = static_cast<int>(batch_items.size());
    int max_len = batch.max_len;

    // Decoder-embedding lane is kLast-only by design: left-padding makes the
    // last real token land at buffer row max_len - 1 for every item, so no
    // per-sample gather is needed. Any other strategy would have to resolve
    // to a per-item index and defeats the purpose of left-padding here.
    PoolingStrategy head_strategy = batch_items[0].state->request->pooling_strategy;
    TVM_FFI_ICHECK_EQ(static_cast<int>(head_strategy), static_cast<int>(PoolingStrategy::kLast))
        << "Decoder embedding lane requires pooling_strategy=last; got "
        << static_cast<int>(head_strategy);

    // Build LEFT-padded input_ids and attention_mask on host.
    // Real tokens occupy [max_len - seq_len, max_len); padding sits on the left.
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
      int pad = max_len - seq_len;

      // Left-pad: zeros for [0, pad), tokens for [pad, max_len).
      for (int s = 0; s < pad; ++s) {
        ids_ptr[b * max_len + s] = 0;
        mask_ptr[b * max_len + s] = 0;
      }
      for (int s = 0; s < seq_len; ++s) {
        ids_ptr[b * max_len + pad + s] = item.token_ids[s];
        mask_ptr[b * max_len + pad + s] = 1;
      }
    }

    // Run decoder embedding prefill, then pool directly into a CPU float32 buffer.
    // Under left padding the last real token is at buffer row max_len - 1 for
    // every item, so we pass lengths=[max_len]*batch and let kLast pick that
    // final row uniformly — no per-sample gather needed.
    ObjectRef hidden_states =
        model_->DecoderEmbeddingPrefill(input_ids_host, attention_mask_host, batch_size, max_len);
    std::vector<int> pool_lengths(batch_size, max_len);
    Tensor pooled_cpu = model_->PoolEmbeddingHiddenStates(
        hidden_states, pool_lengths, batch_size, max_len,
        static_cast<int>(PoolingStrategy::kLast));
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
    refl::ObjectDef<BatchDecoderEmbeddingPrefillActionObj>();
  }

  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("mlc.serve.BatchDecoderEmbeddingPrefillAction",
                              BatchDecoderEmbeddingPrefillActionObj, EngineActionObj);

 private:
  /*! \brief The decoder-only embedding model. */
  Model model_;
  /*! \brief Engine config for batching constraints. */
  EngineConfig engine_config_;
};

TVM_FFI_STATIC_INIT_BLOCK() { BatchDecoderEmbeddingPrefillActionObj::RegisterReflection(); }

EngineAction EngineAction::BatchDecoderEmbeddingPrefill(Model model, EngineConfig engine_config) {
  return EngineAction(
      tvm::ffi::make_object<BatchDecoderEmbeddingPrefillActionObj>(std::move(model),
                                                                   std::move(engine_config)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
