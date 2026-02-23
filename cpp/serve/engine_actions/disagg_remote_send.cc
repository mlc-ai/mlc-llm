/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/new_request_prefill.cc
 */

#include "../sampler/sampler.h"
#include "batch_prefill_base.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that prefills requests in the `waiting_queue` of
 * the engine state.
 * Aside from that, this action sends the computed KV data to remote
 * instances after computing the KV data.
 */
class DisaggRemoteSendActionObj : public BatchPrefillBaseActionObj {
 public:
  explicit DisaggRemoteSendActionObj(Array<Model> models,
                                     std::vector<ModelWorkspace> model_workspaces,
                                     EngineConfig engine_config,
                                     std::vector<picojson::object> model_configs,
                                     Optional<EventTraceRecorder> trace_recorder,
                                     FRequestStreamCallback request_stream_callback, Device device)
      : BatchPrefillBaseActionObj(std::move(models), std::move(engine_config),
                                  std::move(model_configs), std::move(trace_recorder)),
        model_workspaces_(std::move(model_workspaces)),
        request_stream_callback_(std::move(request_stream_callback)),
        device_(device) {
    if (device.device_type == DLDeviceType::kDLCUDA ||
        device.device_type == DLDeviceType::kDLROCM) {
      // The compute stream is the default stream.
      compute_stream_ = DeviceAPI::Get(device)->GetCurrentStream(device);
    }
  }

  // Mimicked from NewRequestPrefillActionObj::Step
  Array<Request> Step(EngineState estate) final {
    // - Find the requests in `waiting_queue` that can prefill in this step.
    std::vector<PrefillInput> prefill_inputs;
    {
      NVTXScopedRange nvtx_scope("DisaggRemoteSend getting requests");
      prefill_inputs = GetRequestStateEntriesToPrefill(estate);
      if (prefill_inputs.empty()) {
        return {};
      }
    }

    int num_rsentries = prefill_inputs.size();
    {
      NVTXScopedRange nvtx_scope("DisaggRemoteSend matching prefix");
      for (int i = 0; i < num_rsentries; ++i) {
        MatchPrefixCache(estate, &prefill_inputs[i]);
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // - Update status of request states from pending to alive.
    Array<String> request_ids;
    std::vector<RequestState> rstates_of_entries;
    std::vector<RequestStateStatus> status_before_prefill;
    UpdateRequestToAlive(prefill_inputs, estate, &request_ids, &rstates_of_entries,
                         &status_before_prefill);

    // - Get embedding and run prefill for each model.
    // NOTE: we don't keep the logits as we don't run sampling in this action by design.
    std::vector<int> prefill_lengths;
    prefill_lengths.resize(/*size=*/num_rsentries, /*value=*/-1);
    for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
      std::vector<int64_t> request_internal_ids;
      request_internal_ids.reserve(num_rsentries);
      ObjectRef embeddings = model_workspaces_[model_id].embeddings;
      int cum_prefill_length = 0;
      bool single_input =
          num_rsentries == 1 && prefill_inputs[0].rsentry->mstates[model_id]->inputs.size() == 1;
      std::vector<int64_t> cached_token_data;
      for (int i = 0; i < num_rsentries; ++i) {
        const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
        RequestModelState mstate = rsentry->mstates[model_id];
        auto [input_data, input_length] =
            ChunkPrefillInputData(mstate, prefill_inputs[i].max_prefill_length);
        if (prefill_lengths[i] == -1) {
          prefill_lengths[i] = input_length;
        } else {
          TVM_FFI_ICHECK_EQ(prefill_lengths[i], input_length);
        }
        mstate->num_prefilled_tokens += input_length;

        TVM_FFI_ICHECK(mstate->draft_output_tokens.empty());
        TVM_FFI_ICHECK(mstate->draft_token_slots.empty());
        if (status_before_prefill[i] == RequestStateStatus::kPending &&
            !estate->prefix_cache->HasSequence(mstate->internal_id)) {
          // Add the sequence to the model.
          // If the sequence is already in prefix cache, it has also been added/forked in the
          // KVCache.
          CHECK_EQ(rsentry->parent_idx, -1);
          models_[model_id]->AddNewSequence(mstate->internal_id);
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            models_[model_id]->EnableSlidingWindowForSeq(mstate->internal_id);
          }
          DisaggConfig disagg_config = mstate->request->generation_cfg->debug_config.disagg_config;
          CHECK(disagg_config.dst_group_offset.has_value());
          models_[model_id]->DisaggMarkKVSend(
              mstate->internal_id, disagg_config.kv_window_begin.value_or(0),
              disagg_config.kv_append_metadata[model_id], disagg_config.dst_group_offset.value());
        }
        request_internal_ids.push_back(mstate->internal_id);
        RECORD_EVENT(trace_recorder_, rsentry->request->id, "start embedding");
        for (int j = 0; j < static_cast<int>(input_data.size()); ++j) {
          if (!model_id && !prefill_inputs[i].is_decode) {
            mstate->prefilled_inputs.push_back(input_data[j]);
          }
          if (const auto* token_data = input_data[j].as<TokenDataNode>()) {
            cached_token_data.insert(cached_token_data.end(), token_data->token_ids.begin(),
                                     token_data->token_ids.end());
          } else {
            if (!cached_token_data.empty()) {
              embeddings = TokenData(cached_token_data)
                               ->GetEmbedding(models_[model_id],
                                              /*dst=*/!single_input ? &embeddings : nullptr,
                                              /*offset=*/cum_prefill_length);
              cum_prefill_length += cached_token_data.size();
              cached_token_data.clear();
            }
            embeddings = input_data[j]->GetEmbedding(models_[model_id],
                                                     /*dst=*/!single_input ? &embeddings : nullptr,
                                                     /*offset=*/cum_prefill_length);
            cum_prefill_length += input_data[j]->GetLength();
          }
        }
        RECORD_EVENT(trace_recorder_, rsentry->request->id, "finish embedding");
      }
      if (!cached_token_data.empty()) {
        embeddings = TokenData(cached_token_data)
                         ->GetEmbedding(models_[model_id],
                                        /*dst=*/!single_input ? &embeddings : nullptr,
                                        /*offset=*/cum_prefill_length);
        cum_prefill_length += cached_token_data.size();
        cached_token_data.clear();
      }

      RECORD_EVENT(trace_recorder_, request_ids, "start prefill");
      Tensor logits =
          models_[model_id]->BatchPrefill(embeddings, request_internal_ids, prefill_lengths);
      RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");
      TVM_FFI_ICHECK_EQ(logits->ndim, 3);
      TVM_FFI_ICHECK_EQ(logits->shape[0], 1);
      TVM_FFI_ICHECK_EQ(logits->shape[1], num_rsentries);
    }

    // - Commit the prefix cache changes from previous round of action.
    // Note: we commit prefix cache changes here to overlap this commit with the GPU execution.
    estate->prefix_cache->CommitSequenceExtention();

    // - We run synchronize to make sure that the prefill is finished.
    // We need explicit synchronization because we don't do sampling in this action.
    DeviceAPI::Get(device_)->StreamSync(device_, compute_stream_);

    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_prefill_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    std::vector<Request> processed_requests =
        RemoveProcessedRequests(prefill_inputs, estate, rstates_of_entries);
    estate->running_rsentries_changed = true;
    return processed_requests;
  }

 private:
  // Mimicked from BatchPrefillBaseActionObj::GetRequestStateEntriesToPrefill
  std::vector<PrefillInput> GetRequestStateEntriesToPrefill(EngineState estate) {
    // Preempt request state entries when decode cannot apply.
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

    // Explicitly filter the waiting queue to only keep the requests
    // with disaggregation request kind "kRemoteSend".
    std::vector<Request> waiting_queue;
    waiting_queue.reserve(estate->waiting_queue.size());
    for (Request request : estate->waiting_queue) {
      if (request->generation_cfg->debug_config.disagg_config.kind ==
          DisaggRequestKind::kRemoteSend) {
        waiting_queue.push_back(request);
      }
    }
    if (waiting_queue.empty()) {
      // No request to prefill.
      return {};
    }

    std::vector<std::vector<PrefillInput>> prefill_inputs_for_all_models;
    prefill_inputs_for_all_models.reserve(models_.size());

    int num_running_rsentries = static_cast<int>(running_rsentries->size());
    // We first collect the inputs that can be prefilled for each model.
    // Then we make a reduction to return the maximum common inputs.
    for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
      std::vector<PrefillInput> prefill_inputs;
      // - Try to prefill pending requests.
      int total_input_length = 0;
      int total_required_pages = 0;
      int num_available_pages;
      int current_total_seq_len;
      {
        NVTXScopedRange nvtx_scope("KV cache GetNumAvailablePages");
        num_available_pages = models_[i]->GetNumAvailablePages();
      }
      {
        NVTXScopedRange nvtx_scope("KV cache GetCurrentTotalSequenceLength");
        current_total_seq_len = models_[i]->GetCurrentTotalSequenceLength();
      }

      int num_prefill_rsentries = 0;
      for (const Request& request : waiting_queue) {
        NVTXScopedRange nvtx_scope("Process request " + request->id);
        RequestState rstate = estate->GetRequestState(request);
        CHECK_EQ(rstate->entries.size(), 1) << "n > 1 is not supported.";
        const RequestStateEntry& rsentry = rstate->entries[0];
        CHECK(!rsentry->mstates[i]->inputs.empty())
            << "The request entry must have pending inputs.";

        int input_length = rsentry->mstates[i]->GetInputLength();
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
          int parent_ptr = rsentry->parent_idx;
          while (parent_ptr != -1) {
            num_total_prefilled_tokens +=
                rstate->entries[parent_ptr]->mstates[i]->num_prefilled_tokens;
            parent_ptr = rstate->entries[parent_ptr]->parent_idx;
          }

          int num_pages_in_use = (std::min(num_total_prefilled_tokens, sliding_window_sizes_[i]) +
                                  engine_config_->kv_cache_page_size - 1) /
                                 engine_config_->kv_cache_page_size;
          num_required_pages_under_sliding_window =
              max_single_request_page_requirement - num_pages_in_use;
          num_require_pages = std::min(num_require_pages, num_required_pages_under_sliding_window);
          TVM_FFI_ICHECK_GE(num_require_pages, 0);
        }

        total_input_length += input_length;
        total_required_pages += num_require_pages;
        // - Attempt 1. Check if the entire request state entry can fit for prefill.
        bool can_prefill = false;
        {
          NVTXScopedRange nvtx_scope("Attempt 1");
          for (int num_child_to_activate = rsentry->child_indices.size();
               num_child_to_activate >= 0; --num_child_to_activate) {
            while (!HasPrefillSpace(total_required_pages, sliding_window_enabled,
                                    (num_running_rsentries + num_prefill_rsentries),
                                    num_available_pages, current_total_seq_len, total_input_length,
                                    engine_config_->max_total_sequence_length)) {
              if (!estate->prefix_cache->TryFreeMemory()) break;
              // Update number of available pages after memory free.
              num_available_pages = models_[i]->GetNumAvailablePages();
              current_total_seq_len = models_[i]->GetCurrentTotalSequenceLength();
            }
            if (CanPrefill(estate, num_prefill_rsentries + 1 + num_child_to_activate,
                           total_input_length, total_required_pages, num_available_pages,
                           current_total_seq_len, num_running_rsentries, kv_state_kind_,
                           sliding_window_enabled)) {
              prefill_inputs.push_back(
                  {rsentry, input_length, num_child_to_activate, /*is_decode=*/false});
              num_prefill_rsentries += 1 + num_child_to_activate;
              can_prefill = true;
              break;
            }
          }
        }
        if (can_prefill) {
          continue;
        }
        total_input_length -= input_length;
        total_required_pages -= num_require_pages;

        // - Attempt 2. Check if the request state entry can partially fit by input chunking.
        TVM_FFI_ICHECK_LE(total_input_length, engine_config_->prefill_chunk_size);
        if (engine_config_->prefill_chunk_size - total_input_length >= input_length ||
            engine_config_->prefill_chunk_size == total_input_length) {
          // 1. If the input length can fit the remaining prefill chunk size,
          // it means the failure of attempt 1 is not because of the input
          // length being too long, and thus chunking does not help.
          // 2. If the total input length already reaches the prefill chunk size,
          // the current request state entry will not be able to be processed.
          // So we can safely return in either case.
          break;
        }
        input_length = engine_config_->prefill_chunk_size - total_input_length;
        num_require_pages = (input_length + engine_config_->kv_cache_page_size - 1) /
                            engine_config_->kv_cache_page_size;
        if (sliding_window_enabled) {
          // Sliding window for model i is enabled.
          num_require_pages = std::min(num_require_pages, num_required_pages_under_sliding_window);
          TVM_FFI_ICHECK_GE(num_require_pages, 0);
        }

        {
          NVTXScopedRange nvtx_scope("Attempt 2");
          total_input_length += input_length;
          total_required_pages += num_require_pages;
          if (CanPrefill(estate, num_prefill_rsentries + 1, total_input_length,
                         total_required_pages, num_available_pages, current_total_seq_len,
                         num_running_rsentries, kv_state_kind_, sliding_window_enabled)) {
            prefill_inputs.push_back({rsentry, input_length, 0, /*is_decode=*/false});
          }
        }

        // - Prefill stops here.
        break;
      }
      prefill_inputs_for_all_models.push_back(prefill_inputs);
    }

    // Reduce over the prefill inputs of all models.
    TVM_FFI_ICHECK(!prefill_inputs_for_all_models.empty());
    int num_prefill_inputs = prefill_inputs_for_all_models[0].size();
    for (int i = 1; i < static_cast<int>(prefill_inputs_for_all_models.size()); ++i) {
      num_prefill_inputs =
          std::min(num_prefill_inputs, static_cast<int>(prefill_inputs_for_all_models[i].size()));
    }

    if (num_prefill_inputs == 0) {
      return {};
    }

    // Add the decode requests to the prefill inputs if prefill mode is hybrid.
    std::vector<PrefillInput> prefill_inputs(prefill_inputs_for_all_models[0].begin(),
                                             prefill_inputs_for_all_models[0].end());
    {
      NVTXScopedRange nvtx_scope("reduction");
      for (int i = 1; i < static_cast<int>(prefill_inputs_for_all_models.size()); ++i) {
        // Prefill input lengths except the last one are supposed to be the same for all models.
        for (int j = 0; j < num_prefill_inputs - 1; ++j) {
          TVM_FFI_ICHECK(
              prefill_inputs_for_all_models[i][j].rsentry.same_as(prefill_inputs[j].rsentry));
          TVM_FFI_ICHECK_EQ(prefill_inputs_for_all_models[i][j].max_prefill_length,
                            prefill_inputs[j].max_prefill_length);
          prefill_inputs[j].num_child_to_activate =
              std::min(prefill_inputs[j].num_child_to_activate,
                       prefill_inputs_for_all_models[i][j].num_child_to_activate);
        }
        // The input length of the last input is the minimum among all models.
        TVM_FFI_ICHECK(prefill_inputs_for_all_models[i][num_prefill_inputs - 1].rsentry.same_as(
            prefill_inputs[num_prefill_inputs - 1].rsentry));
        prefill_inputs[num_prefill_inputs - 1].max_prefill_length =
            std::min(prefill_inputs[num_prefill_inputs - 1].max_prefill_length,
                     prefill_inputs_for_all_models[i][num_prefill_inputs - 1].max_prefill_length);
        prefill_inputs[num_prefill_inputs - 1].num_child_to_activate = std::min(
            prefill_inputs[num_prefill_inputs - 1].num_child_to_activate,
            prefill_inputs_for_all_models[i][num_prefill_inputs - 1].num_child_to_activate);
      }
    }

    return prefill_inputs;
  }

  // Copied from NewRequestPrefillActionObj::MatchPrefixCache
  /*!
   * \brief Match the request state entry with prefix cache, to skip prefilling common prefix
   * tokens. If the request state entry is not added to KVCache yet, this method will add/fork the
   * request in the KVCache, depending on the matching result from prefix cache.
   * \param estate The engine state.
   * \param[in, out] input The prefill input to be matched and updated.
   * \return The matched length in prefix cache.
   */
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
        for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
          Model model = models_[model_id];
          RequestModelState mstate = rsentry->mstates[model_id];
          model->AddNewSequence(rsentry->mstates[0]->internal_id);
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            model->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
          }
          DisaggConfig disagg_config = mstate->request->generation_cfg->debug_config.disagg_config;
          models_[model_id]->DisaggMarkKVSend(
              mstate->internal_id, disagg_config.kv_window_begin.value_or(0),
              disagg_config.kv_append_metadata[model_id], disagg_config.dst_group_offset.value());
        }
      } else {
        if (result.forked_seq_id != -1) {
          CHECK_EQ(result.reused_seq_id, -1);
          CHECK_EQ(result.reused_seq_pop_last_tokens, 0);
          // Fork from active sequence
          for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
            Model model = models_[model_id];
            RequestModelState mstate = rsentry->mstates[model_id];
            model->ForkSequence(result.forked_seq_id, rsentry->mstates[0]->internal_id,
                                result.prefilled_offset);
            // Enable sliding window for the sequence if it is not a parent.
            if (rsentry->child_indices.empty()) {
              model->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
            }
            DisaggConfig disagg_config =
                mstate->request->generation_cfg->debug_config.disagg_config;
            models_[model_id]->DisaggMarkKVSend(
                mstate->internal_id, disagg_config.kv_window_begin.value_or(0),
                disagg_config.kv_append_metadata[model_id], disagg_config.dst_group_offset.value());
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
          for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
            RequestModelState mstate = rsentry->mstates[model_id];
            DisaggConfig disagg_config =
                mstate->request->generation_cfg->debug_config.disagg_config;
            models_[model_id]->DisaggMarkKVSend(
                mstate->internal_id, disagg_config.kv_window_begin.value_or(0),
                disagg_config.kv_append_metadata[model_id], disagg_config.dst_group_offset.value());
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

  /*! \brief Workspace of each model. */
  std::vector<ModelWorkspace> model_workspaces_;
  /*! \brief The stream callback function to passes back the sampled results after prefill. */
  FRequestStreamCallback request_stream_callback_;
  /*! \brief The device which we run synchronization for after prefill. */
  Device device_;
  /*! \brief The compute stream to run synchronization for. */
  TVMStreamHandle compute_stream_ = nullptr;
};

EngineAction EngineAction::DisaggRemoteSend(
    Array<Model> models, std::vector<ModelWorkspace> model_workspaces, EngineConfig engine_config,
    std::vector<picojson::object> model_configs, Optional<EventTraceRecorder> trace_recorder,
    FRequestStreamCallback request_stream_callback, Device device) {
  return EngineAction(tvm::ffi::make_object<DisaggRemoteSendActionObj>(
      std::move(models), std::move(model_workspaces), std::move(engine_config),
      std::move(model_configs), std::move(trace_recorder), std::move(request_stream_callback),
      device));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
