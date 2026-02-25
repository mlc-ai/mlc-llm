/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/eagle_new_request_prefill.cc
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
class EagleNewRequestPrefillActionObj : public BatchPrefillBaseActionObj {
 public:
  explicit EagleNewRequestPrefillActionObj(Array<Model> models, LogitProcessor logit_processor,
                                           Sampler sampler,
                                           std::vector<ModelWorkspace> model_workspaces,
                                           DraftTokenWorkspaceManager draft_token_workspace_manager,
                                           EngineConfig engine_config,
                                           std::vector<picojson::object> model_configs,
                                           Optional<EventTraceRecorder> trace_recorder)
      : BatchPrefillBaseActionObj(std::move(models), std::move(engine_config),
                                  std::move(model_configs), std::move(trace_recorder)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)) {}

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
    ObjectRef hidden_states_for_input{nullptr};
    ObjectRef hidden_states_for_sample{nullptr};
    Tensor logits_for_sample{nullptr};
    // A map used to record the entry and child_idx pair needed to fork sequence.
    // The base model (id 0) should record all the pairs and all the small models
    // fork sequences according to this map.
    std::unordered_map<int, std::unordered_set<int>> fork_rsentry_child_map;
    std::vector<bool> extra_prefill_tokens;
    prefill_lengths.resize(/*size=*/num_rsentries, /*value=*/false);
    for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
      std::vector<int64_t> request_internal_ids;
      request_internal_ids.reserve(num_rsentries);
      ObjectRef embeddings = model_workspaces_[model_id].embeddings;
      int cum_prefill_length = 0;
      bool single_input =
          num_rsentries == 1 && prefill_inputs[0].rsentry->mstates[model_id]->inputs.size() == 1;
      for (int i = 0; i < num_rsentries; ++i) {
        const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
        RequestModelState mstate = rsentry->mstates[model_id];
        TVM_FFI_ICHECK(mstate->draft_output_tokens.empty());
        TVM_FFI_ICHECK(mstate->draft_token_slots.empty());
        if (status_before_prefill[i] == RequestStateStatus::kPending) {
          if (!estate->prefix_cache->HasSequence(mstate->internal_id)) {
            // Add the sequence to the model, or fork the sequence from its parent.
            // If the sequence is already in prefix cache, it has also been added/forked in the
            // KVCache.
            if (rsentry->parent_idx == -1) {
              models_[model_id]->AddNewSequence(mstate->internal_id);
            } else {
              models_[model_id]->ForkSequence(rstates_of_entries[i]
                                                  ->entries[rsentry->parent_idx]
                                                  ->mstates[model_id]
                                                  ->internal_id,
                                              mstate->internal_id);
            }
          }
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            models_[model_id]->EnableSlidingWindowForSeq(mstate->internal_id);
          }
          // Shift the input tokens by 1 for eagle models.
          if (model_id == 0) {
            for (int j = 1; j < static_cast<int>(models_.size()); ++j) {
              TVM_FFI_ICHECK(rsentry->mstates[j]->inputs.size());
              TokenData token_data = Downcast<TokenData>(rsentry->mstates[j]->inputs[0]);
              rsentry->mstates[j]->inputs.Set(
                  0, TokenData(
                         IntTuple(token_data->token_ids.begin() + 1, token_data->token_ids.end())));
            }
          }
        }
        request_internal_ids.push_back(mstate->internal_id);

        if (engine_config_->speculative_mode == SpeculativeMode::kMedusa && model_id > 0) {
          // Embedding is only needed for the base model in Medusa.
          continue;
        }
        auto [input_data, input_length] =
            ChunkPrefillInputData(mstate, prefill_inputs[i].max_prefill_length);
        if (prefill_lengths[i] == -1) {
          prefill_lengths[i] = input_length;
        } else {
          TVM_FFI_ICHECK_EQ(prefill_lengths[i], input_length);
        }
        mstate->num_prefilled_tokens += input_length;

        RECORD_EVENT(trace_recorder_, prefill_inputs[i].rsentry->request->id, "start embedding");
        // Speculative models shift left the input tokens by 1 when base model has committed tokens.
        // Note: for n > 1 cases Eagle doesn't work because parent entry doesn't shift input tokens.
        for (int j = 0; j < static_cast<int>(input_data.size()); ++j) {
          if (model_id == 0) {
            mstate->prefilled_inputs.push_back(input_data[j]);
          }
          embeddings = input_data[j]->GetEmbedding(
              models_[model_id],
              /*dst=*/!single_input ? &model_workspaces_[model_id].embeddings : nullptr,
              /*offset=*/cum_prefill_length);
          cum_prefill_length += input_data[j]->GetLength();
        }
        RECORD_EVENT(trace_recorder_, rsentry->request->id, "finish embedding");
      }

      RECORD_EVENT(trace_recorder_, request_ids, "start prefill");

      Array<Tensor> multi_step_logits{nullptr};

      if (model_id == 0 || engine_config_->speculative_mode == SpeculativeMode::kEagle) {
        ObjectRef embedding_or_hidden_states{nullptr};
        if (model_id == 0) {
          embedding_or_hidden_states = embeddings;
        } else {
          embedding_or_hidden_states =
              models_[model_id]->FuseEmbedHidden(embeddings, hidden_states_for_input,
                                                 /*batch_size*/ 1, /*seq_len*/ cum_prefill_length);
        }
        // hidden_states: (b * s, h)
        ObjectRef hidden_states = models_[model_id]->BatchPrefillToLastHidden(
            embedding_or_hidden_states, request_internal_ids, prefill_lengths);
        RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");

        if (model_id == 0) {
          // We only need to sample for model 0 in prefill.
          hidden_states_for_input = hidden_states;

          // - Commit the prefix cache changes from previous round of action.
          // Note: we commit prefix cache changes here to overlap this commit with the GPU
          // execution.
          estate->prefix_cache->CommitSequenceExtention();
        }

        // Whether to use base model to get logits.
        int sample_model_id = !models_[model_id]->CanGetLogits() ? 0 : model_id;

        std::vector<int> logit_positions;
        {
          // Prepare the logit positions
          logit_positions.reserve(prefill_lengths.size());
          int total_len = 0;
          for (int i = 0; i < prefill_lengths.size(); ++i) {
            total_len += prefill_lengths[i];
            logit_positions.push_back(total_len - 1);
          }
        }
        // hidden_states_for_sample: (b * s, h)
        hidden_states_for_sample = models_[sample_model_id]->GatherHiddenStates(
            hidden_states, logit_positions, &model_workspaces_[model_id].hidden_states);
        // logits_for_sample: (b * s, v)
        logits_for_sample = models_[sample_model_id]->GetLogits(hidden_states_for_sample);
      } else if (engine_config_->speculative_mode == SpeculativeMode::kMedusa) {
        // Note: spec_draft_length in engine config has to be match the model config in Medusa.
        multi_step_logits = models_[model_id]->GetMultiStepLogits(hidden_states_for_sample);
      } else {
        LOG(FATAL) << "unreachable";
      }

      Array<String> child_request_ids;
      // - Prepare the configurations for the sampler.
      //   For prefill_inputs which have children, sample
      //   one token for each rstate that is depending.
      //   Otherwise, sample a token for the current rstate.
      std::vector<int> child_sample_indices;
      std::vector<RequestStateEntry> rsentries_for_sample;
      std::vector<RandomGenerator*> rngs;
      std::vector<bool> rsentry_activated;
      Array<GenerationConfig> child_generation_cfg;
      child_sample_indices.reserve(num_rsentries);
      child_generation_cfg.reserve(num_rsentries);
      child_request_ids.reserve(num_rsentries);
      rsentries_for_sample.reserve(num_rsentries);
      rngs.reserve(num_rsentries);
      rsentry_activated.reserve(num_rsentries);
      for (int i = 0; i < num_rsentries; ++i) {
        const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
        // No sample for rsentries with remaining inputs.
        if (!rsentry->mstates[0]->inputs.empty()) {
          continue;
        }

        int remaining_num_child_to_activate = prefill_inputs[i].num_child_to_activate;
        for (int child_idx : rsentry->child_indices) {
          // Only use base model to judge if we need to add child entries.
          if ((rstates_of_entries[i]->entries[child_idx]->status == RequestStateStatus::kPending &&
                   rstates_of_entries[i]
                       ->entries[child_idx]
                       ->mstates[0]
                       ->committed_tokens.empty() ||
               fork_rsentry_child_map[i].count(child_idx))) {
            // If rstates_of_entries[i]->entries[child_idx] has no committed token,
            // the prefill of the current rsentry will unblock
            // rstates_of_entries[i]->entries[child_idx],
            // and thus we want to sample a token for rstates_of_entries[i]->entries[child_idx].
            fork_rsentry_child_map[i].insert(child_idx);
            child_sample_indices.push_back(i);
            rsentries_for_sample.push_back(rstates_of_entries[i]->entries[child_idx]);
            child_request_ids.push_back(rsentry->request->id);
            child_generation_cfg.push_back(rsentry->request->generation_cfg);
            rngs.push_back(&rstates_of_entries[i]->entries[child_idx]->rng);

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
            if (model_id == 0) {
              TVM_FFI_ICHECK(rstates_of_entries[i]->entries[child_idx]->status ==
                             RequestStateStatus::kPending);
              rstates_of_entries[i]->entries[child_idx]->status = RequestStateStatus::kAlive;
            }
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
          child_sample_indices.push_back(i);
          rsentries_for_sample.push_back(rsentry);
          child_request_ids.push_back(rsentry->request->id);
          child_generation_cfg.push_back(rsentry->request->generation_cfg);
          rngs.push_back(&rsentry->rng);
          rsentry_activated.push_back(true);
        }
      }

      // - Prepare input for logit processor.
      TVM_FFI_ICHECK(logits_for_sample.defined());
      Array<GenerationConfig> generation_cfg;
      Array<RequestModelState> mstates_for_logitproc;
      std::vector<int> sample_indices(num_rsentries);
      generation_cfg.reserve(num_rsentries);
      mstates_for_logitproc.reserve(num_rsentries);
      std::iota(sample_indices.begin(), sample_indices.end(), 0);
      for (int i = 0; i < num_rsentries; ++i) {
        generation_cfg.push_back(prefill_inputs[i].rsentry->request->generation_cfg);
        mstates_for_logitproc.push_back(prefill_inputs[i].rsentry->mstates[model_id]);
      }
      if (model_id == 0 || engine_config_->speculative_mode == SpeculativeMode::kEagle) {
        const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
            logit_processor_, sampler_, logits_for_sample, generation_cfg, request_ids,
            mstates_for_logitproc, rngs, sample_indices, child_generation_cfg, child_request_ids,
            child_sample_indices);
        if (model_id == 0) {
          UpdateRequestStateEntriesWithSampleResults(rsentries_for_sample, rsentry_activated,
                                                     sample_results);
          // Add the sampled token as an input of the eagle models.
          if (engine_config_->speculative_mode == SpeculativeMode::kEagle) {
            for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
              for (int mid = 1; mid < static_cast<int>(models_.size()); ++mid) {
                TokenData token_data =
                    Downcast<TokenData>(rsentries_for_sample[i]->mstates[mid]->inputs.back());
                std::vector<int32_t> token_ids = {token_data->token_ids.begin(),
                                                  token_data->token_ids.end()};
                token_ids.push_back(sample_results[i].GetTokenId());
                int ninputs =
                    static_cast<int>(rsentries_for_sample[i]->mstates[mid]->inputs.size());
                rsentries_for_sample[i]->mstates[mid]->inputs.Set(
                    ninputs - 1, TokenData(IntTuple(token_ids.begin(), token_ids.end())));
              }
            }
          }
        } else {
          // - Slice and save hidden_states_for_sample
          UpdateRequestStatesWithDraftProposals(rsentries_for_sample, sample_results, model_id,
                                                renormalized_probs, hidden_states_for_sample,
                                                estate, child_sample_indices);
        }
      } else if (engine_config_->speculative_mode == SpeculativeMode::kMedusa) {
        TVM_FFI_ICHECK_NE(estate->spec_draft_length, 0);
        for (int draft_id = 0; draft_id < estate->spec_draft_length; ++draft_id) {
          const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
              logit_processor_, sampler_, multi_step_logits[draft_id], generation_cfg, request_ids,
              mstates_for_logitproc, rngs, sample_indices, child_generation_cfg, child_request_ids,
              child_sample_indices);

          UpdateRequestStatesWithDraftProposals(
              rsentries_for_sample, sample_results, model_id, renormalized_probs,
              /*hidden_states=*/ObjectRef{nullptr}, estate, child_sample_indices);
        }
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_prefill_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    std::vector<Request> processed_requests =
        RemoveProcessedRequests(prefill_inputs, estate, rstates_of_entries);
    estate->running_rsentries_changed = true;
    return processed_requests;
  }

  void UpdateRequestStatesWithDraftProposals(
      const std::vector<RequestStateEntry>& rsentries_for_sample,
      const std::vector<SampleResult>& sample_results, int model_id,
      const Tensor& renormalized_probs, const ObjectRef& hidden_states_for_sample,
      EngineState estate, const std::vector<int>& sample_indices) {
    std::vector<int> reuse_count(renormalized_probs->shape[0], 0);
    for (int i = 0; i < static_cast<int>(sample_indices.size()); ++i) {
      // The same probability may be sampled multiple times.
      reuse_count[sample_indices[i]]++;
    }
    draft_token_workspace_manager_->AllocSlots(renormalized_probs->shape[0], reuse_count,
                                               &draft_token_slots_);

    models_[0]->ScatterDraftProbs(renormalized_probs, draft_token_slots_,
                                  &model_workspaces_[0].draft_probs_storage);
    if (engine_config_->speculative_mode == SpeculativeMode::kEagle &&
        estate->spec_draft_length > 1) {
      models_[0]->ScatterHiddenStates(hidden_states_for_sample, draft_token_slots_,
                                      &model_workspaces_[0].draft_hidden_states_storage);
    }
    for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
      int parent_idx =
          rsentries_for_sample[i]->mstates[model_id]->draft_output_tokens.empty()
              ? -1
              : rsentries_for_sample[i]->mstates[model_id]->draft_output_tokens.size() - 1;
      rsentries_for_sample[i]->mstates[model_id]->AddDraftToken(
          sample_results[i], draft_token_slots_[sample_indices[i]], parent_idx);
    }
  }

 private:
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Workspace of each model. */
  std::vector<ModelWorkspace> model_workspaces_;
  /*! \brief The draft token workspace manager. */
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
  /*! \brief Temporary buffer to store the slots of the current draft tokens */
  std::vector<int> draft_token_slots_;

  /*!
   * \brief Match the request state entry with prefix cache, to skip prefilling common prefix
   * tokens. If the request state entry is not added to KVCache yet, this method will add/fork the
   * request in the KVCache, depending on the matching result from prefix cache.
   * \param estate The engine state.
   * \param[in, out] input The prefill input to be matched and updated.
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
        // Add new sequence.
        // Note: Almost same as without eagle speculative decoding. But in prefill step, the
        // prefill embedding input in draft model will be shifted one token, compared to the base
        // model. Just the new sequence without prefix cache. Here we merely add the new sequence
        // in advance of prefill step.
        CHECK_EQ(result.forked_seq_id, -1);
        CHECK_EQ(result.reused_seq_id, -1);
        CHECK_EQ(result.reused_seq_pop_last_tokens, 0);
        for (int i = 0; i < models_.size(); ++i) {
          models_[i]->AddNewSequence(rsentry->mstates[0]->internal_id);
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            models_[i]->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
          }
        }
      } else {
        if (result.forked_seq_id != -1) {
          // Fork from active sequence
          // Note: Due to the shifted KVCache between base model and draft model, we do a trick
          // over forking sequence:
          // For example. we have a sequence of [0, 1, 2] in base model KVCache, and the
          // corresponding sequence of [1, 2, 3] in draft model KVCache, where token [3] was
          // sampled from base model, but not appended in base model KVCache. Then we get a new
          // sequence [0, 1, 4] to prefill. Although the new sequence matches first two tokens
          // with the sequence [0, 1, 2], we have to fork from the first token 0, not the second
          // token 1. Because if we fork from the second token, we will prefill like: Base model:
          // [0, 1] + prefill([4]) => [5] Draft model: [1] + prefill([4, 5]) The lengths to
          // prefill is different between base model and draft model, which is illegal. So we roll
          // back one token in prefix cache to fork from the first token. Then the prefill will be
          // like: Base model: [0] + prefill([1, 4]) => [5] Draft model: [1] + prefill([4, 5]) And
          // we shift the input prefill data as other new sequence, to avoid double prefilling
          // token 1, and make the prefill length aligned between base model and draft model.
          CHECK_EQ(result.reused_seq_id, -1);
          CHECK_EQ(result.reused_seq_pop_last_tokens, 0);
          estate->prefix_cache->RollBackSequence(rsentry->mstates[0]->internal_id, 1);
          for (int i = 0; i < models_.size(); ++i) {
            models_[i]->ForkSequence(result.forked_seq_id, rsentry->mstates[0]->internal_id,
                                     result.prefilled_offset - 1);
            // Enable sliding window for the sequence if it is not a parent.
            if (rsentry->child_indices.empty()) {
              models_[i]->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
            }
          }
        } else {
          // Reuse recycling sequence
          // Note: The processing for reusing recycling sequence is like forking sequence. And we
          // also roll back one token due to the reason mentioned above.
          CHECK_EQ(result.forked_seq_id, -1);
          estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
          for (int i = 0; i < rsentry->mstates.size(); ++i) {
            rsentry->mstates[i]->internal_id = result.reused_seq_id;
          }
          estate->prefix_cache->RollBackSequence(rsentry->mstates[0]->internal_id, 1);
          for (int i = 0; i < models_.size(); ++i) {
            models_[i]->PopNFromKVCache(rsentry->mstates[0]->internal_id,
                                        result.reused_seq_pop_last_tokens + 1);
          }
          result.prefilled_offset -= 1;
        }
      }
      // Pop matched prefix
      if (result.prefilled_offset > 0) {
        for (int i = 0; i < rsentry->mstates.size(); ++i) {
          PopPrefillInputData(rsentry->mstates[i], result.prefilled_offset);
        }
      }
      // Update max prefill length
      input->max_prefill_length =
          std::min(input->max_prefill_length, rsentry->mstates[0]->GetInputLength());
      return result.prefilled_offset - 1;
    }
    return 0;
  }
};

EngineAction EngineAction::EagleNewRequestPrefill(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces,
    DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
    std::vector<picojson::object> model_configs, Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<EagleNewRequestPrefillActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(model_configs), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
