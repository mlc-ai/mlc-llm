/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/eagle_batch_verify.cc
 */

#include <tvm/runtime/threading_backend.h>

#include <cmath>
#include <exception>
#include <numeric>

#include "../../support/random.h"
#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs verification for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 */
class EagleBatchVerifyActionObj : public EngineActionObj {
 public:
  explicit EagleBatchVerifyActionObj(Array<Model> models, LogitProcessor logit_processor,
                                     Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                     DraftTokenWorkspaceManager draft_token_workspace_manager,
                                     EngineConfig engine_config,
                                     Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)),
        rng_(RandomGenerator::GetInstance()) {}

  Array<Request> Step(EngineState estate) final {
    // - Only run spec decode when there are two models (llm+ssm) and >=1 running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    const auto& [rsentries, draft_lengths, total_draft_length] = GetDraftsToVerify(estate);
    TVM_FFI_ICHECK_EQ(rsentries.size(), draft_lengths.size());
    if (rsentries.empty()) {
      return {};
    }

    auto tstart = std::chrono::high_resolution_clock::now();
    int num_rsentries = rsentries.size();
    Array<String> request_ids =
        rsentries.Map([](const RequestStateEntry& rstate) { return rstate->request->id; });

    // - Get embedding and run verify.
    std::vector<int64_t> request_internal_ids;
    std::vector<int32_t> all_tokens_to_verify;
    Array<RequestModelState> verify_request_mstates;
    Array<RequestModelState> draft_request_mstates;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    std::vector<std::vector<SampleResult>> draft_output_tokens;
    std::vector<std::vector<int>> draft_token_indices;
    std::vector<int64_t> token_tree_parent_ptr;
    request_internal_ids.reserve(num_rsentries);
    all_tokens_to_verify.reserve(total_draft_length);
    token_tree_parent_ptr.reserve(total_draft_length);
    verify_request_mstates.reserve(num_rsentries);
    draft_request_mstates.reserve(num_rsentries);
    rngs.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    draft_output_tokens.reserve(num_rsentries);
    draft_token_indices.reserve(num_rsentries);
    draft_token_slots_.clear();

    for (int i = 0; i < num_rsentries; ++i) {
      RequestModelState verify_mstate = rsentries[i]->mstates[verify_model_id_];
      RequestModelState draft_mstate = rsentries[i]->mstates[draft_model_id_];
      request_internal_ids.push_back(verify_mstate->internal_id);
      TVM_FFI_ICHECK(!draft_lengths.empty());
      TVM_FFI_ICHECK_EQ(draft_lengths[i], draft_mstate->draft_output_tokens.size());
      TVM_FFI_ICHECK_EQ(draft_lengths[i], draft_mstate->draft_token_slots.size());
      // the last committed token + all the draft tokens but the last one.
      all_tokens_to_verify.push_back(draft_mstate->committed_tokens.back().GetTokenId());
      draft_token_slots_.push_back(0);  // placeholder for the last committed token
      token_tree_parent_ptr.push_back(-1);

      for (int j = 0; j < static_cast<int>(draft_mstate->draft_output_tokens.size()); ++j) {
        all_tokens_to_verify.push_back(draft_mstate->draft_output_tokens[j].GetTokenId());
        draft_token_slots_.push_back(draft_mstate->draft_token_slots[j]);
        token_tree_parent_ptr.push_back(draft_mstate->draft_token_parent_idx[j] + 1);
      }
      std::vector<int> cur_draft_token_indices(draft_mstate->draft_output_tokens.size() + 1);
      std::iota(cur_draft_token_indices.begin(), cur_draft_token_indices.end(), -1);
      draft_token_indices.emplace_back(std::move(cur_draft_token_indices));
      verify_request_mstates.push_back(verify_mstate);
      draft_request_mstates.push_back(draft_mstate);
      generation_cfg.push_back(rsentries[i]->request->generation_cfg);
      rngs.push_back(&rsentries[i]->rng);
      draft_output_tokens.push_back(draft_mstate->draft_output_tokens);
    }

    Tensor draft_probs_on_device = models_[draft_model_id_]->GatherDraftProbs(
        model_workspaces_[verify_model_id_].draft_probs_storage, draft_token_slots_,
        &model_workspaces_[verify_model_id_].draft_probs);

    std::vector<int> cum_verify_lengths = {0};
    cum_verify_lengths.reserve(num_rsentries + 1);
    std::vector<int> verify_lengths;
    for (int i = 0; i < num_rsentries; ++i) {
      // Add one committed token.
      verify_lengths.push_back(draft_lengths[i] + 1);
      cum_verify_lengths.push_back(cum_verify_lengths.back() + verify_lengths.back());
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start verify embedding");
    ObjectRef embeddings = models_[verify_model_id_]->TokenEmbed(
        {IntTuple{all_tokens_to_verify.begin(), all_tokens_to_verify.end()}});
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify embedding");

    RECORD_EVENT(trace_recorder_, request_ids, "start verify");
    ObjectRef hidden_states = models_[verify_model_id_]->BatchVerifyToLastHidden(
        embeddings, request_internal_ids, verify_lengths, token_tree_parent_ptr);
    Tensor logits = models_[verify_model_id_]->GetLogits(hidden_states);
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify");
    TVM_FFI_ICHECK_EQ(logits->ndim, 2);
    TVM_FFI_ICHECK_EQ(logits->shape[0], cum_verify_lengths.back());

    // - Update logits.
    logit_processor_->InplaceUpdateLogits(logits, generation_cfg, verify_request_mstates,
                                          request_ids, &cum_verify_lengths, &draft_request_mstates,
                                          &draft_token_indices);

    // - Compute probability distributions.
    Tensor probs_on_device = logit_processor_->ComputeProbsFromLogits(
        logits, generation_cfg, request_ids, &cum_verify_lengths);

    // - Commit the prefix cache changes from previous round of action.
    // Note: we commit prefix cache changes here to overlap this commit with the GPU execution.
    estate->prefix_cache->CommitSequenceExtention();

    std::vector<int> sample_indices(num_rsentries);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
        probs_on_device, sample_indices, request_ids, generation_cfg);
    auto [sample_results_arr, _] = sampler_->BatchVerifyDraftTokensWithProbAfterTopP(
        renormalized_probs, request_ids, cum_verify_lengths, generation_cfg, rngs,
        draft_output_tokens, token_tree_parent_ptr, draft_probs_on_device);
    TVM_FFI_ICHECK_EQ(sample_results_arr.size(), num_rsentries);

    // We collect the requests whose drafts are fully accepted.
    // When a request's draft is fully accepted, there is an extra token proposed
    // by the draft model but not added into the draft model's KV cache.
    // In this case, an additional batch decode step is needed for these requests.
    std::vector<int64_t> fully_accepted_rsentries;
    std::vector<int64_t> verify_model_seq_internal_ids;
    std::vector<int64_t> accepted_token_tree_leaf_nodes;
    fully_accepted_rsentries.reserve(num_rsentries);
    verify_model_seq_internal_ids.reserve(num_rsentries);
    accepted_token_tree_leaf_nodes.reserve(num_rsentries);

    std::vector<int> last_accepted_hidden_positions;
    last_accepted_hidden_positions.reserve(num_rsentries);
    for (int i = 0; i < num_rsentries; ++i) {
      const std::vector<SampleResult>& sample_results = sample_results_arr[i];
      int accept_length = sample_results.size();
      TVM_FFI_ICHECK_GE(accept_length, 1);
      for (SampleResult sample_result : sample_results) {
        rsentries[i]->mstates[verify_model_id_]->CommitToken(sample_result);
        rsentries[i]->mstates[draft_model_id_]->CommitToken(sample_result);
      }
      // Metrics update
      // live update the output metrics
      rsentries[i]->rstate->metrics.completion_tokens += accept_length;
      rsentries[i]->rstate->metrics.decode_tokens += accept_length;
      estate->metrics.spec_decode.Update(cum_verify_lengths[i + 1] - cum_verify_lengths[i],
                                         accept_length);
      // - Minus one because the last draft token has no kv cache entry
      // - Take max with 0 in case of all accepted.
      int rollback_length =
          std::max(cum_verify_lengths[i + 1] - cum_verify_lengths[i] - accept_length, 0);

      // Commit accepted tokens to the "verify_model", rollback kv cache
      // in the "draft_model".
      // NOTE: when number of small models is more than 1 (in the future),
      // it is possible to re-compute prefill for the small models.
      verify_model_seq_internal_ids.push_back(rsentries[i]->mstates[verify_model_id_]->internal_id);
      accepted_token_tree_leaf_nodes.push_back(accept_length - 1);
      if (rollback_length > 0) {
        // Draft model rollback minus one because verify uses one more token.
        models_[draft_model_id_]->PopNFromKVCache(
            rsentries[i]->mstates[draft_model_id_]->internal_id, rollback_length - 1);
      } else {
        fully_accepted_rsentries.push_back(i);
      }
      // clear the draft model state entries
      rsentries[i]->mstates[draft_model_id_]->RemoveAllDraftTokens(&draft_token_slots_);
      draft_token_workspace_manager_->FreeSlots(draft_token_slots_);
      // - Slice and save hidden_states_for_sample
      last_accepted_hidden_positions.push_back(cum_verify_lengths[i] + accept_length - 1);
    }
    models_[verify_model_id_]->CommitAcceptedTokenTreeNodesToKVCache(
        verify_model_seq_internal_ids, accepted_token_tree_leaf_nodes);
    if (!fully_accepted_rsentries.empty() &&
        engine_config_->speculative_mode == SpeculativeMode::kEagle) {
      // - Run a step of batch decode for requests whose drafts are fully accepted.
      // When a request's draft is fully accepted, there is an extra token proposed
      // by the draft model but not added into the draft model's KV cache.
      // In this case, an additional batch decode step is needed for these requests.
      std::vector<int> input_tokens;
      std::vector<int64_t> fully_accepted_request_internal_ids;
      input_tokens.reserve(fully_accepted_rsentries.size());
      fully_accepted_request_internal_ids.reserve(fully_accepted_rsentries.size());

      std::vector<int> hidden_states_positions_for_fully_accepted;
      hidden_states_positions_for_fully_accepted.reserve(fully_accepted_rsentries.size());

      for (int rsentry_id : fully_accepted_rsentries) {
        int num_committed_tokens =
            rsentries[rsentry_id]->mstates[verify_model_id_]->committed_tokens.size();
        // When a request's draft is fully accepted, an additional new token is sampled.
        // So the token needed to fill in the draft model is the committed_token[-2].
        TVM_FFI_ICHECK_GE(num_committed_tokens, 2);
        input_tokens.push_back(rsentries[rsentry_id]
                                   ->mstates[verify_model_id_]
                                   ->committed_tokens[num_committed_tokens - 2]
                                   .GetTokenId());

        // Taking the hidden states of the token before the last token
        hidden_states_positions_for_fully_accepted.push_back(
            last_accepted_hidden_positions[rsentry_id] - 1);
        fully_accepted_request_internal_ids.push_back(
            rsentries[rsentry_id]->mstates[draft_model_id_]->internal_id);
      }

      // - Compute embeddings.
      ObjectRef embeddings = models_[draft_model_id_]->TokenEmbed(
          {IntTuple{input_tokens.begin(), input_tokens.end()}});
      // - Gather hidden states
      ObjectRef hidden_states_for_fully_accepted = models_[draft_model_id_]->GatherHiddenStates(
          hidden_states, hidden_states_positions_for_fully_accepted,
          &model_workspaces_[draft_model_id_].hidden_states);
      // - Invoke model decode.
      ObjectRef fused_embedding_hidden_states = models_[draft_model_id_]->FuseEmbedHidden(
          embeddings, hidden_states_for_fully_accepted,
          /*batch_size*/ fully_accepted_rsentries.size(), /*seq_len*/ 1);
      hidden_states_for_fully_accepted = models_[draft_model_id_]->BatchDecodeToLastHidden(
          fused_embedding_hidden_states, fully_accepted_request_internal_ids);
      // - We explicitly synchronize to avoid the input tokens getting overriden in the
      // next runs of BatchDecode.
      // This is because we do not do sample for this round of batch decode.
      if (hidden_states_for_fully_accepted->IsInstance<DRefObj>()) {
        Downcast<Session>(Downcast<DRef>(hidden_states_for_fully_accepted)->session)->SyncWorker(0);
      } else {
        Tensor hidden_states_for_fully_accepted_nd =
            Downcast<Tensor>(hidden_states_for_fully_accepted);
        DeviceAPI::Get(hidden_states_for_fully_accepted_nd->device)
            ->StreamSync(hidden_states_for_fully_accepted_nd->device, nullptr);
      }
    }
    {
      // One step draft for the following steps

      // Gather hidden states for the last accepted tokens.
      // Use the function and the workspace of the verify model because the information about the
      // hidden states is not available in the draft model for medusa.
      hidden_states = models_[0]->GatherHiddenStates(hidden_states, last_accepted_hidden_positions,
                                                     &model_workspaces_[0].hidden_states);

      std::vector<int> input_tokens;
      Array<RequestModelState> mstates;
      input_tokens.reserve(num_rsentries);
      mstates.reserve(num_rsentries);
      for (const RequestStateEntry& rsentry : rsentries) {
        mstates.push_back(rsentry->mstates[draft_model_id_]);
      }
      for (int i = 0; i < num_rsentries; ++i) {
        TVM_FFI_ICHECK(!mstates[i]->committed_tokens.empty());
        input_tokens.push_back(mstates[i]->committed_tokens.back().GetTokenId());
      }

      Array<Tensor> multi_step_logits{nullptr};  // for medusa output
      if (engine_config_->speculative_mode == SpeculativeMode::kEagle) {
        // - Compute embeddings.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal embedding");
        embeddings = models_[draft_model_id_]->TokenEmbed(
            {IntTuple{input_tokens.begin(), input_tokens.end()}});
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal embedding");

        // - Invoke model decode.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal decode");
        ObjectRef fused_embedding_hidden_states = models_[draft_model_id_]->FuseEmbedHidden(
            embeddings, hidden_states, /*batch_size*/ num_rsentries, /*seq_len*/ 1);
        hidden_states = models_[draft_model_id_]->BatchDecodeToLastHidden(
            fused_embedding_hidden_states, request_internal_ids);

        int lm_head_model_id = models_[draft_model_id_]->CanGetLogits() ? draft_model_id_ : 0;
        logits = models_[lm_head_model_id]->GetLogits(hidden_states);
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal decode");
        TVM_FFI_ICHECK_EQ(logits->ndim, 2);
        TVM_FFI_ICHECK_EQ(logits->shape[0], num_rsentries);
      } else if (engine_config_->speculative_mode == SpeculativeMode::kMedusa) {
        multi_step_logits = models_[draft_model_id_]->GetMultiStepLogits(hidden_states);
      }

      // Fill range [0, num_rsentries) into `sample_indices`.
      std::vector<int> sample_indices(num_rsentries);
      std::iota(sample_indices.begin(), sample_indices.end(), 0);

      if (engine_config_->speculative_mode == SpeculativeMode::kEagle) {
        const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
            logit_processor_, sampler_, logits, generation_cfg, request_ids, mstates, rngs,
            sample_indices, generation_cfg, request_ids, sample_indices);
        UpdateRequestStatesWithDraftProposals(mstates, sample_results, draft_model_id_,
                                              renormalized_probs, hidden_states, estate);
      } else if (engine_config_->speculative_mode == SpeculativeMode::kMedusa) {
        TVM_FFI_ICHECK_NE(estate->spec_draft_length, 0);
        for (int draft_id = 0; draft_id < estate->spec_draft_length; draft_id++) {
          const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
              logit_processor_, sampler_, multi_step_logits[draft_id], generation_cfg, request_ids,
              mstates, rngs, sample_indices, generation_cfg, request_ids, sample_indices);
          UpdateRequestStatesWithDraftProposals(mstates, sample_results, draft_model_id_,
                                                renormalized_probs, hidden_states, estate);
        }
      }
    }
    // reset num_tokens_for_next_decode
    for (const RequestStateEntry& rsentry : rsentries) {
      rsentry->mstates[verify_model_id_]->num_tokens_for_next_decode = 0;
      rsentry->mstates[draft_model_id_]->num_tokens_for_next_decode = 0;
    }
    auto tend = std::chrono::high_resolution_clock::now();
    double elapsed_time = static_cast<double>((tend - tstart).count()) / 1e9;
    estate->metrics.engine_decode_time_sum += elapsed_time;
    estate->metrics.UpdateVerifyTimeByBatchSize(cum_verify_lengths.back(), elapsed_time);

    return estate->running_queue;
  }

 private:
  struct DraftRequestStateEntries {
    /*! \brief The request state entries to verify. */
    Array<RequestStateEntry> draft_rsentries;
    /*! \brief The draft length of each request state. */
    std::vector<int> draft_lengths;
    /*! \brief The total draft length. */
    int total_draft_length;
  };

  /*!
   * \brief Decide whether to run verify for the draft of each request.
   * \param estate The engine state.
   * \return The drafts to verify, together with their respective
   * state and input length.
   */
  DraftRequestStateEntries GetDraftsToVerify(EngineState estate) {
    std::vector<int> draft_lengths;
    int total_draft_length = 0;
    int total_required_pages = 0;
    int num_available_pages = models_[verify_model_id_]->GetNumAvailablePages();

    // Preempt the request state entries that cannot fit the large model for verification.
    std::vector<RequestStateEntry> running_rsentries = estate->GetRunningRequestStateEntries();
    std::vector<int> num_page_requirement;
    num_page_requirement.reserve(running_rsentries.size());
    for (const RequestStateEntry& rsentry : running_rsentries) {
      int draft_length = rsentry->mstates[draft_model_id_]->draft_output_tokens.size();
      int num_require_pages = (draft_length + engine_config_->kv_cache_page_size - 1) /
                              engine_config_->kv_cache_page_size;
      draft_lengths.push_back(draft_length);
      num_page_requirement.push_back(num_require_pages);
      total_draft_length += draft_length;
      total_required_pages += num_require_pages;
    }
    while (!CanVerify(total_required_pages)) {
      if (estate->prefix_cache->TryFreeMemory()) continue;
      RequestStateEntry preempted = PreemptLastRunningRequestStateEntry(
          estate, models_, draft_token_workspace_manager_, trace_recorder_);
      if (preempted.same_as(running_rsentries.back())) {
        total_draft_length -= draft_lengths.back();
        total_required_pages -= num_page_requirement.back();
        draft_lengths.pop_back();
        num_page_requirement.pop_back();
        running_rsentries.pop_back();
      }
    }

    return {running_rsentries, draft_lengths, total_draft_length};
  }

  bool CanVerify(int num_required_pages) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_required_pages <= num_available_pages;
  }

  void UpdateRequestStatesWithDraftProposals(const Array<RequestModelState>& mstates,
                                             const std::vector<SampleResult>& sample_results,
                                             int model_id, const Tensor& renormalized_probs,
                                             const ObjectRef& hidden_states_for_sample,
                                             EngineState estate) {
    draft_token_workspace_manager_->AllocSlots(mstates.size(), &draft_token_slots_);
    models_[0]->ScatterDraftProbs(renormalized_probs, draft_token_slots_,
                                  &model_workspaces_[0].draft_probs_storage);
    if (engine_config_->speculative_mode == SpeculativeMode::kEagle &&
        estate->spec_draft_length > 1) {
      models_[0]->ScatterHiddenStates(hidden_states_for_sample, draft_token_slots_,
                                      &model_workspaces_[0].draft_hidden_states_storage);
    }
    for (int i = 0; i < static_cast<int>(mstates.size()); ++i) {
      int64_t parent_idx = static_cast<int64_t>(mstates[i]->draft_output_tokens.size()) - 1;
      mstates[i]->AddDraftToken(sample_results[i], draft_token_slots_[i], parent_idx);
    }
  }
  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Workspace of each model. */
  std::vector<ModelWorkspace> model_workspaces_;
  /*! \brief The draft token workspace manager. */
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Random number generator. */
  RandomGenerator& rng_;
  /*! \brief The ids of verify/draft models. */
  const int verify_model_id_ = 0;
  const int draft_model_id_ = 1;
  const float eps_ = 1e-5;
  /*! \brief Temporary buffer to store the slots of the current draft tokens */
  std::vector<int> draft_token_slots_;
};

EngineAction EngineAction::EagleBatchVerify(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces,
    DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
    Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<EagleBatchVerifyActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
