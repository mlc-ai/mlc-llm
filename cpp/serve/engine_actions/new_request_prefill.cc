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
 */
class NewRequestPrefillActionObj : public BatchPrefillBaseActionObj {
 public:
  explicit NewRequestPrefillActionObj(Array<Model> models, LogitProcessor logit_processor,
                                      Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                      EngineConfig engine_config,
                                      std::vector<picojson::object> model_configs,
                                      Optional<EventTraceRecorder> trace_recorder)
      : BatchPrefillBaseActionObj(std::move(models), std::move(engine_config),
                                  std::move(model_configs), std::move(trace_recorder)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)) {}

  Array<Request> Step(EngineState estate) final {
    // - Find the requests in `waiting_queue` that can prefill in this step.
    std::vector<PrefillInput> prefill_inputs;
    {
      NVTXScopedRange nvtx_scope("NewRequestPrefill getting requests");
      prefill_inputs = GetRequestStateEntriesToPrefill(estate);
      if (prefill_inputs.empty()) {
        return {};
      }
    }

    int num_rsentries = prefill_inputs.size();
    {
      NVTXScopedRange nvtx_scope("NewRequestPrefill matching prefix");
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
    std::vector<int> prefill_lengths;
    prefill_lengths.resize(/*size=*/num_rsentries, /*value=*/-1);
    Tensor logits_for_sample{nullptr};
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
          // Add the sequence to the model, or fork the sequence from its parent.
          // If the sequence is already in prefix cache, it has also been added/forked in the
          // KVCache.
          if (rsentry->parent_idx == -1) {
            models_[model_id]->AddNewSequence(mstate->internal_id);
          } else {
            models_[model_id]->ForkSequence(
                rstates_of_entries[i]->entries[rsentry->parent_idx]->mstates[model_id]->internal_id,
                mstate->internal_id);
          }
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            models_[model_id]->EnableSlidingWindowForSeq(mstate->internal_id);
          }
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

      if (model_id == 0) {
        // We only need to sample for model 0 in prefill.
        logits_for_sample = logits;
      }
    }

    // - Update logits.
    TVM_FFI_ICHECK(logits_for_sample.defined());
    Array<GenerationConfig> generation_cfg;
    Array<RequestModelState> mstates_for_logitproc;
    generation_cfg.reserve(num_rsentries);
    mstates_for_logitproc.reserve(num_rsentries);
    for (int i = 0; i < num_rsentries; ++i) {
      generation_cfg.push_back(prefill_inputs[i].rsentry->request->generation_cfg);
      mstates_for_logitproc.push_back(prefill_inputs[i].rsentry->mstates[0]);
    }
    logits_for_sample = logits_for_sample.CreateView({num_rsentries, logits_for_sample->shape[2]},
                                                     logits_for_sample->dtype);
    logit_processor_->InplaceUpdateLogits(logits_for_sample, generation_cfg, mstates_for_logitproc,
                                          request_ids);

    // - Compute probability distributions.
    Tensor probs_on_device =
        logit_processor_->ComputeProbsFromLogits(logits_for_sample, generation_cfg, request_ids);

    // - Commit the prefix cache changes from previous round of action.
    // Note: we commit prefix cache changes here to overlap this commit with the GPU execution.
    estate->prefix_cache->CommitSequenceExtention();

    // - Sample tokens.
    //   For rsentries which have children, sample
    //   one token for each rstate that is depending.
    //   Otherwise, sample a token for the current rstate.
    std::vector<int> sample_indices;
    std::vector<RequestStateEntry> rsentries_for_sample;
    std::vector<RandomGenerator*> rngs;
    std::vector<bool> rsentry_activated;
    sample_indices.reserve(num_rsentries);
    rsentries_for_sample.reserve(num_rsentries);
    rngs.reserve(num_rsentries);
    rsentry_activated.reserve(num_rsentries);
    request_ids.clear();
    generation_cfg.clear();
    for (int i = 0; i < num_rsentries; ++i) {
      const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
      // No sample for rsentries with remaining inputs.
      if (!rsentry->mstates[0]->inputs.empty()) {
        continue;
      }

      int remaining_num_child_to_activate = prefill_inputs[i].num_child_to_activate;
      for (int child_idx : rsentry->child_indices) {
        // If rstates_of_entries[i]->entries[child_idx] has no committed token,
        // the prefill of the current rsentry will unblock
        // rstates_of_entries[i]->entries[child_idx],
        // and thus we want to sample a token for rstates_of_entries[i]->entries[child_idx].
        if (rstates_of_entries[i]->entries[child_idx]->status != RequestStateStatus::kPending ||
            !rstates_of_entries[i]->entries[child_idx]->mstates[0]->committed_tokens.empty()) {
          continue;
        }
        sample_indices.push_back(i);
        rsentries_for_sample.push_back(rstates_of_entries[i]->entries[child_idx]);
        request_ids.push_back(rsentry->request->id);
        generation_cfg.push_back(rsentry->request->generation_cfg);
        rngs.push_back(&rstates_of_entries[i]->entries[child_idx]->rng);

        TVM_FFI_ICHECK(rstates_of_entries[i]->entries[child_idx]->status ==
                       RequestStateStatus::kPending);
        // We only fork the first `num_child_to_activate` children.
        // The children not being forked will be forked via later prefills.
        // Usually `num_child_to_activate` is the same as the number of children.
        // But it can be fewer subject to the KV cache max num sequence limit.
        if (remaining_num_child_to_activate == 0) {
          rsentry_activated.push_back(false);
          continue;
        }
        rsentry_activated.push_back(true);
        --remaining_num_child_to_activate;
        rstates_of_entries[i]->entries[child_idx]->status = RequestStateStatus::kAlive;
        for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
          int64_t child_internal_id =
              rstates_of_entries[i]->entries[child_idx]->mstates[model_id]->internal_id;
          models_[model_id]->ForkSequence(rsentry->mstates[model_id]->internal_id,
                                          child_internal_id);
          // Enable sliding window for the child sequence if the child is not a parent.
          if (rstates_of_entries[i]->entries[child_idx]->child_indices.empty()) {
            models_[model_id]->EnableSlidingWindowForSeq(child_internal_id);
          }
        }
      }
      if (rsentry->child_indices.empty()) {
        // If rsentry has no child, we sample a token for itself.
        sample_indices.push_back(i);
        rsentries_for_sample.push_back(rsentry);
        request_ids.push_back(rsentry->request->id);
        generation_cfg.push_back(rsentry->request->generation_cfg);
        rngs.push_back(&rsentry->rng);
        rsentry_activated.push_back(true);
      }
    }
    Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
        probs_on_device, sample_indices, request_ids, generation_cfg);
    std::vector<SampleResult> sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
        renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
    TVM_FFI_ICHECK_EQ(sample_results.size(), rsentries_for_sample.size());

    // - Update the committed tokens of states.
    // - If a request is first-time prefilled, set the prefill finish time.
    UpdateRequestStateEntriesWithSampleResults(rsentries_for_sample, rsentry_activated,
                                               sample_results);

    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_prefill_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    std::vector<Request> processed_requests =
        RemoveProcessedRequests(prefill_inputs, estate, rstates_of_entries);
    estate->running_rsentries_changed = true;
    return processed_requests;
  }

 private:
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Workspace of each model. */
  std::vector<ModelWorkspace> model_workspaces_;

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
};  // namespace serve

EngineAction EngineAction::NewRequestPrefill(Array<Model> models, LogitProcessor logit_processor,
                                             Sampler sampler,
                                             std::vector<ModelWorkspace> model_workspaces,
                                             EngineConfig engine_config,
                                             std::vector<picojson::object> model_configs,
                                             Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<NewRequestPrefillActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(engine_config), std::move(model_configs),
      std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
