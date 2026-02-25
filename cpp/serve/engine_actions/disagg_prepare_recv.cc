/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/new_request_prefill.cc
 */

#include <optional>

#include "../../support/utils.h"
#include "../sampler/sampler.h"
#include "batch_prefill_base.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs prefill preparation in disaggregation system.
 * It picks a new request, reserve its KV data locations, and returns the
 * KV data locations and the matched prefix length in prefix cache.
 */
class DisaggPrepareReceiveActionObj : public BatchPrefillBaseActionObj {
 public:
  explicit DisaggPrepareReceiveActionObj(Array<Model> models, EngineConfig engine_config,
                                         std::vector<picojson::object> model_configs,
                                         Optional<EventTraceRecorder> trace_recorder,
                                         FRequestStreamCallback request_stream_callback)
      : BatchPrefillBaseActionObj(std::move(models), std::move(engine_config),
                                  std::move(model_configs), std::move(trace_recorder)),
        request_stream_callback_(std::move(request_stream_callback)) {
    CHECK(kv_state_kind_ == KVStateKind::kKVCache)
        << "Only PagedKVCache supports prefill preparation and KV migration";
  }

  Array<Request> Step(EngineState estate) final {
    std::vector<Request> processed_requests;

    // - Find the requests in `waiting_queue` that can prefill in this step.
    std::optional<PrefillInput> prefill_input_opt;
    while (true) {
      prefill_input_opt = GetRequestStateEntriesToPrefill(estate);
      if (!prefill_input_opt.has_value()) {
        break;
      }
      PrefillInput prefill_input = prefill_input_opt.value();
      int prefix_matched_length = 0;
      Request request = prefill_input.rsentry->request;
      processed_requests.push_back(request);
      int total_input_length = 0;
      for (const Data& data : request->inputs) {
        total_input_length += data->GetLength();
      }

      {
        NVTXScopedRange nvtx_scope("DisaggPrepareReceive matching prefix");
        prefix_matched_length = MatchPrefixCache(estate, &prefill_input);
      }

      auto tstart = std::chrono::high_resolution_clock::now();

      // - Update status of request states from pending to alive.
      Array<String> request_ids;
      std::vector<RequestState> rstates_of_entries;
      std::vector<RequestStateStatus> status_before_prefill;
      UpdateRequestToAlive({prefill_input}, estate, &request_ids, &rstates_of_entries,
                           &status_before_prefill);
      // "UpdateRequestToAlive" may add the request to the engine's running request queue.
      // We erase it since it's pending for the prefill instance to send the KV data over.
      if (!estate->running_queue.empty() && estate->running_queue.back().same_as(request)) {
        estate->running_queue.pop_back();
      }

      // - Add the sequence to each model.
      int prefill_length = -1;
      Tensor logits_for_sample{nullptr};
      std::vector<IntTuple> kv_append_metadata;
      kv_append_metadata.reserve(models_.size());
      for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
        const RequestStateEntry& rsentry = prefill_input.rsentry;
        RequestModelState mstate = rsentry->mstates[model_id];
        Array<Data> input_data = mstate->inputs;
        mstate->inputs.clear();
        int input_length = prefill_input.max_prefill_length;
        if (prefill_length == -1) {
          prefill_length = input_length;
        } else {
          TVM_FFI_ICHECK_EQ(prefill_length, input_length);
        }
        mstate->num_prefilled_tokens += input_length;

        TVM_FFI_ICHECK(mstate->draft_output_tokens.empty());
        TVM_FFI_ICHECK(mstate->draft_token_slots.empty());
        if (status_before_prefill[0] == RequestStateStatus::kPending &&
            !estate->prefix_cache->HasSequence(mstate->internal_id)) {
          // Add the sequence to the model, or fork the sequence from its parent.
          // If the sequence is already in prefix cache, it has also been added/forked in the
          // KVCache.
          if (rsentry->parent_idx == -1) {
            models_[model_id]->AddNewSequence(mstate->internal_id);
          } else {
            models_[model_id]->ForkSequence(
                rstates_of_entries[0]->entries[rsentry->parent_idx]->mstates[model_id]->internal_id,
                mstate->internal_id);
          }
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            models_[model_id]->EnableSlidingWindowForSeq(mstate->internal_id);
          }
        }

        // Record the of the prefilled inputs for prefix cache update.
        for (int j = 0; j < static_cast<int>(input_data.size()); ++j) {
          if (!model_id && !prefill_input.is_decode) {
            mstate->prefilled_inputs.push_back(input_data[j]);
          }
        }

        int64_t request_internal_id = mstate->internal_id;
        RECORD_EVENT(trace_recorder_, request_ids, "start prefill");
        IntTuple compressed_kv_append_metadata = {0};
        if (prefill_length > 0) {
          compressed_kv_append_metadata =
              models_[model_id]->DisaggPrepareKVRecv(request_internal_id, prefill_length);
        }
        kv_append_metadata.push_back(compressed_kv_append_metadata);
        RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");
      }

      // - Commit the prefix cache changes from previous round of action.
      // Note: we commit prefix cache changes here to overlap this commit with the GPU execution.
      estate->prefix_cache->CommitSequenceExtention();

      auto tend = std::chrono::high_resolution_clock::now();

      // - Remove the request from the waiting queue.
      auto it_request =
          std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), request);
      TVM_FFI_ICHECK(it_request != estate->waiting_queue.end());
      estate->waiting_queue.erase(it_request);

      {
        NVTXScopedRange nvtx_scope("Call request stream callback");
        picojson::object response_body;
        response_body["prompt_length"] = picojson::value(static_cast<int64_t>(total_input_length));
        response_body["prefix_matched_length"] =
            picojson::value(static_cast<int64_t>(prefix_matched_length));
        // We further flatten the metadata array of all models into a single array.
        picojson::array kv_append_metadata_arr;
        for (const IntTuple& compressed_kv_append_metadata : kv_append_metadata) {
          for (int64_t value : compressed_kv_append_metadata) {
            kv_append_metadata_arr.push_back(picojson::value(value));
          }
          TVM_FFI_ICHECK(!compressed_kv_append_metadata.empty());
          int num_segments = compressed_kv_append_metadata[0];
          TVM_FFI_ICHECK_EQ(compressed_kv_append_metadata.size(), num_segments * 2 + 1);
          int transmission_length = 0;
          for (int i = 0; i < num_segments; ++i) {
            transmission_length += compressed_kv_append_metadata[i * 2 + 2];
          }
          CHECK_EQ(transmission_length, prefill_length);
        }

        response_body["kv_append_metadata"] =
            picojson::value(Base64Encode(picojson::value(kv_append_metadata_arr).serialize()));

        picojson::object usage;
        usage["prompt_tokens"] = picojson::value(static_cast<int64_t>(0));
        usage["completion_tokens"] = picojson::value(static_cast<int64_t>(0));
        usage["total_tokens"] = picojson::value(static_cast<int64_t>(0));
        usage["extra"] = picojson::value(response_body);
        RequestStreamOutput stream_output =
            RequestStreamOutput::Usage(request->id, picojson::value(usage).serialize());
        // - Invoke the stream callback function once for all collected requests.
        request_stream_callback_(Array<RequestStreamOutput>{stream_output});
      }
    }

    for (const Request& request : processed_requests) {
      CHECK(std::find(estate->running_queue.begin(), estate->running_queue.end(), request) ==
            estate->running_queue.end());
    }
    return {processed_requests};
  }

 private:
  // Mimicked from BatchPrefillBaseActionObj::GetRequestStateEntriesToPrefill
  std::optional<PrefillInput> GetRequestStateEntriesToPrefill(EngineState estate) {
    const std::vector<RequestStateEntry>* running_rsentries;
    {
      NVTXScopedRange nvtx_scope("BatchDecode getting requests");
      running_rsentries = &estate->GetRunningRequestStateEntries();
      if (!(running_rsentries->size() <= models_[0]->GetNumAvailablePages())) {
        // Even the decode cannot be performed.
        // As a result, directly return without doing prefill.
        return {};
      }
    }
    int num_running_rsentries = static_cast<int>(running_rsentries->size());

    Request request{nullptr};
    for (const Request& request_candidate : estate->waiting_queue) {
      if (request_candidate->generation_cfg->debug_config.disagg_config.kind ==
          DisaggRequestKind::kPrepareReceive) {
        request = request_candidate;
        break;
      }
    }
    if (!request.defined()) {
      // No request to prepare for prefill.
      return {};
    }
    CHECK_EQ(request->generation_cfg->debug_config.disagg_config.kv_window_begin.value_or(0), 0);

    std::vector<PrefillInput> prefill_input_for_all_models;
    prefill_input_for_all_models.reserve(models_.size());

    // We first collect the inputs that can be prefilled for each model.
    // The inputs for each model are expected to be exactly the same.
    for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
      NVTXScopedRange nvtx_scope("Process request " + request->id);

      PrefillInput prefill_input;
      // - Try to prefill pending requests.
      int num_available_pages = models_[i]->GetNumAvailablePages();
      int current_total_seq_len = models_[i]->GetCurrentTotalSequenceLength();

      RequestState rstate = estate->GetRequestState(request);
      bool prefill_stops = false;
      for (int j = 1; j < static_cast<int>(rstate->entries.size()); ++j) {
        CHECK(rstate->entries[j]->mstates[i]->inputs.empty())
            << "Re-prefill of preempted requests is not supported by prefill preparation.";
      }
      const RequestStateEntry& rsentry = rstate->entries[0];
      CHECK(!rsentry->mstates[i]->inputs.empty()) << "The request entry must have pending inputs.";

      // Todo: handle the case that input length is 1.

      int input_length = rsentry->mstates[i]->GetInputLength();
      // Update the input length with the requested KV window, where "[begin:end]"
      // means the KV range to prefill on a prefill instance.
      int kv_window_begin =
          request->generation_cfg->debug_config.disagg_config.kv_window_begin.value_or(0);
      int kv_window_end =
          request->generation_cfg->debug_config.disagg_config.kv_window_end.value_or(input_length);
      CHECK_EQ(kv_window_begin, 0);
      if (kv_window_end < 0) {
        kv_window_end = input_length + kv_window_end;
      }
      CHECK_GE(kv_window_end, 0);
      CHECK_LT(kv_window_end, input_length)
          << "Prefill the full input on the remote machine is not supported.";
      int orig_input_length = input_length;
      input_length = kv_window_end;

      int num_require_pages = (input_length + engine_config_->kv_cache_page_size - 1) /
                              engine_config_->kv_cache_page_size;
      bool sliding_window_enabled = sliding_window_sizes_[i] != -1;
      int num_required_pages_under_sliding_window = std::numeric_limits<int>::max();
      if (sliding_window_enabled) {
        // Sliding window for model i is enabled.
        int max_single_request_page_requirement =
            1 + (sliding_window_sizes_[i] + engine_config_->kv_cache_page_size - 1) /
                    engine_config_->kv_cache_page_size;
        int num_total_prefilled_tokens = rsentry->mstates[i]->num_prefilled_tokens;
        int num_pages_in_use = (std::min(num_total_prefilled_tokens, sliding_window_sizes_[i]) +
                                engine_config_->kv_cache_page_size - 1) /
                               engine_config_->kv_cache_page_size;
        num_required_pages_under_sliding_window =
            max_single_request_page_requirement - num_pages_in_use;
        num_require_pages = std::min(num_require_pages, num_required_pages_under_sliding_window);
        TVM_FFI_ICHECK_GE(num_require_pages, 0);
      }

      // Check if the entire request state entry can fit for prefill.
      bool can_prefill = false;
      {
        NVTXScopedRange nvtx_scope("Attempt");
        for (int num_child_to_activate = rsentry->child_indices.size(); num_child_to_activate >= 0;
             --num_child_to_activate) {
          while (!HasPrefillSpace(num_require_pages, sliding_window_enabled, num_running_rsentries,
                                  num_available_pages, current_total_seq_len, input_length,
                                  engine_config_->max_total_sequence_length)) {
            if (!estate->prefix_cache->TryFreeMemory()) break;
            // Update number of available pages after memory free.
            num_available_pages = models_[i]->GetNumAvailablePages();
            current_total_seq_len = models_[i]->GetCurrentTotalSequenceLength();
          }
          if (CanPrefill(estate, 1 + num_child_to_activate, input_length, num_require_pages,
                         num_available_pages, current_total_seq_len, num_running_rsentries,
                         kv_state_kind_, sliding_window_enabled)) {
            prefill_input = {rsentry, input_length, num_child_to_activate, /*is_decode=*/false};
            can_prefill = true;
            break;
          }
        }
      }
      if (!can_prefill) {
        return std::nullopt;
      }
      rsentry->mstates[i]->inputs =
          SplitData(rsentry->mstates[i]->inputs, orig_input_length, kv_window_end).first;
      prefill_input_for_all_models.push_back(prefill_input);
    }

    // Prefill inputs of all models should be the same.
    TVM_FFI_ICHECK(!prefill_input_for_all_models.empty());
    PrefillInput prefill_input = prefill_input_for_all_models[0];
    {
      NVTXScopedRange nvtx_scope("reduction");
      for (int i = 1; i < static_cast<int>(prefill_input_for_all_models.size()); ++i) {
        TVM_FFI_ICHECK(prefill_input_for_all_models[i].rsentry.same_as(prefill_input.rsentry));
        TVM_FFI_ICHECK_EQ(prefill_input_for_all_models[i].max_prefill_length,
                          prefill_input.max_prefill_length);
        TVM_FFI_ICHECK_EQ(prefill_input_for_all_models[i].num_child_to_activate,
                          prefill_input.num_child_to_activate);
      }
    }

    return prefill_input;
  }

  // Mimicked from BatchPrefillBaseActionObj::CanPrefill
  bool CanPrefill(EngineState estate, int num_prefill_rsentries, int total_input_length,
                  int num_required_pages, int num_available_pages, int current_total_seq_len,
                  int num_running_rsentries, KVStateKind kv_state_kind,
                  bool sliding_window_enabled) {
    // No exceeding of the maximum allowed requests that can
    // run simultaneously.
    int spec_factor = engine_config_->speculative_mode != SpeculativeMode::kDisable
                          ? (estate->spec_draft_length + 1)
                          : 1;
    if ((num_running_rsentries + num_prefill_rsentries) * spec_factor >
        std::min(static_cast<int64_t>(engine_config_->max_num_sequence),
                 engine_config_->prefill_chunk_size)) {
      return false;
    }

    // NOTE: The conditions are heuristic and can be revised.
    // Cond 1: at least one decode can be performed after prefill.
    // Cond 2: number of total tokens after "x" times of decode does not
    // exceed the limit, where "x" is a watermark number can
    // be configured and adjusted in the future.
    if (num_required_pages + 400 > num_available_pages) {
      return false;
    }
    return HasPrefillSpace(num_required_pages, sliding_window_enabled,
                           (num_running_rsentries + num_prefill_rsentries), num_available_pages,
                           current_total_seq_len, total_input_length,
                           engine_config_->max_total_sequence_length);
  }

  // Mimicked from NewRequestPrefillActionObj::MatchPrefixCache
  int MatchPrefixCache(EngineState estate, PrefillInput* input) final {
    RequestStateEntry rsentry = input->rsentry;
    if (estate->prefix_cache->Mode() == PrefixCacheMode::kDisable) {
      return 0;
    }
    if (rsentry->parent_idx == -1 && rsentry->status == RequestStateStatus::kPending &&
        !estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
      std::vector<int32_t> tokens = GetConcatPrefillInputData(rsentry->mstates[0]);
      if (tokens.empty()) {
        // If the RequestStateEntry is of empty input data, or not fully tokenized, do nothing
        // and return.
        return 0;
      }
      PrefixCacheMatchedResult result = estate->prefix_cache->InsertSequence(
          rsentry->mstates[0]->internal_id, tokens, models_[0]->GetSlidingWindowSize(),
          models_[0]->GetAttentionSinkSize());

      if (result.prefilled_offset == 0) {
        // Add new sequence
        CHECK_EQ(result.forked_seq_id, -1);
        CHECK_EQ(result.reused_seq_id, -1);
        CHECK_EQ(result.reused_seq_pop_last_tokens, 0);
        for (Model model : models_) {
          model->AddNewSequence(rsentry->mstates[0]->internal_id);
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            model->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
          }
        }
      } else {
        if (result.forked_seq_id != -1) {
          CHECK_EQ(result.reused_seq_id, -1);
          CHECK_EQ(result.reused_seq_pop_last_tokens, 0);
          // Fork from active sequence
          for (Model model : models_) {
            model->ForkSequence(result.forked_seq_id, rsentry->mstates[0]->internal_id,
                                result.prefilled_offset);
            // Enable sliding window for the sequence if it is not a parent.
            if (rsentry->child_indices.empty()) {
              model->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
            }
          }
        } else {
          // Reuse recycling sequence
          CHECK_EQ(result.forked_seq_id, -1);
          estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
          for (int i = 0; i < rsentry->mstates.size(); ++i) {
            rsentry->mstates[i]->internal_id = result.reused_seq_id;
          }
          if (result.reused_seq_pop_last_tokens > 0) {
            for (Model model : models_) {
              model->PopNFromKVCache(rsentry->mstates[0]->internal_id,
                                     result.reused_seq_pop_last_tokens);
            }
          }
        }
      }
      // Pop matched prefix
      if (result.prefilled_offset) {
        for (int i = 0; i < rsentry->mstates.size(); ++i) {
          PopPrefillInputData(rsentry->mstates[i], result.prefilled_offset);
        }
      }
      // Update max prefill length
      input->max_prefill_length =
          std::min(input->max_prefill_length, rsentry->mstates[0]->GetInputLength());
      return result.prefilled_offset;
    }
    return 0;
  }

  /*!
   * \brief The stream callback function to passes back the KV cache metadata
   * and prefix matched length in prefix cache.
   */
  FRequestStreamCallback request_stream_callback_;
};

EngineAction EngineAction::DisaggPrepareReceive(Array<Model> models, EngineConfig engine_config,
                                                std::vector<picojson::object> model_configs,
                                                Optional<EventTraceRecorder> trace_recorder,
                                                FRequestStreamCallback request_stream_callback) {
  return EngineAction(tvm::ffi::make_object<DisaggPrepareReceiveActionObj>(
      std::move(models), std::move(engine_config), std::move(model_configs),
      std::move(trace_recorder), std::move(request_stream_callback)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
