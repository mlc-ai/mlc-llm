/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_verify.cc
 */

#include <tvm/runtime/threading_backend.h>

#include <cmath>
#include <exception>

#include "../../random.h"
#include "../config.h"
#include "../model.h"
#include "../sampler.h"
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
class BatchVerifyActionObj : public EngineActionObj {
 public:
  explicit BatchVerifyActionObj(Array<Model> models, LogitProcessor logit_processor,
                                Sampler sampler, KVCacheConfig kv_cache_config,
                                Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        kv_cache_config_(std::move(kv_cache_config)),
        trace_recorder_(std::move(trace_recorder)),
        rng_(RandomGenerator::GetInstance()) {}

  Array<Request> Step(EngineState estate) final {
    // - Only run spec decode when there are two models (llm+ssm) and >=1 running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    const auto& [rsentries, draft_lengths, total_draft_length] = GetDraftsToVerify(estate);
    ICHECK_EQ(rsentries.size(), draft_lengths.size());
    if (rsentries.empty()) {
      return {};
    }

    int num_rsentries = rsentries.size();
    Array<String> request_ids =
        rsentries.Map([](const RequestStateEntry& rstate) { return rstate->request->id; });
    auto tstart = std::chrono::high_resolution_clock::now();

    // - Get embedding and run verify.
    std::vector<int64_t> request_internal_ids;
    std::vector<int32_t> all_tokens_to_verify;
    Array<RequestModelState> verify_request_mstates;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    std::vector<std::vector<SampleResult>> draft_output_tokens;
    std::vector<std::vector<NDArray>> draft_output_prob_dist;
    request_internal_ids.reserve(num_rsentries);
    all_tokens_to_verify.reserve(total_draft_length);
    verify_request_mstates.reserve(num_rsentries);
    rngs.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    draft_output_tokens.reserve(num_rsentries);
    draft_output_prob_dist.reserve(num_rsentries);

    for (int i = 0; i < num_rsentries; ++i) {
      RequestModelState verify_mstate = rsentries[i]->mstates[verify_model_id_];
      RequestModelState draft_mstate = rsentries[i]->mstates[draft_model_id_];
      request_internal_ids.push_back(verify_mstate->internal_id);
      ICHECK(!draft_lengths.empty());
      ICHECK_EQ(draft_lengths[i], draft_mstate->draft_output_tokens.size());
      ICHECK_EQ(draft_lengths[i], draft_mstate->draft_output_prob_dist.size());
      // the last committed token + all the draft tokens but the last one.
      all_tokens_to_verify.push_back(draft_mstate->committed_tokens.back().sampled_token_id.first);
      for (int j = 0; j < static_cast<int>(draft_mstate->draft_output_tokens.size()) - 1; ++j) {
        all_tokens_to_verify.push_back(draft_mstate->draft_output_tokens[j].sampled_token_id.first);
      }
      verify_request_mstates.push_back(verify_mstate);
      generation_cfg.push_back(rsentries[i]->request->generation_cfg);
      rngs.push_back(&rsentries[i]->rng);
      draft_output_tokens.push_back(draft_mstate->draft_output_tokens);
      draft_output_prob_dist.push_back(draft_mstate->draft_output_prob_dist);
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start verify embedding");
    ObjectRef embeddings = models_[verify_model_id_]->TokenEmbed(
        {IntTuple{all_tokens_to_verify.begin(), all_tokens_to_verify.end()}});
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify embedding");

    RECORD_EVENT(trace_recorder_, request_ids, "start verify");
    NDArray logits =
        models_[verify_model_id_]->BatchVerify(embeddings, request_internal_ids, draft_lengths);
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify");
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], total_draft_length);

    // - Update logits.
    std::vector<int> cum_verify_lengths = {0};
    cum_verify_lengths.reserve(num_rsentries + 1);
    for (int i = 0; i < num_rsentries; ++i) {
      cum_verify_lengths.push_back(cum_verify_lengths.back() + draft_lengths[i]);
    }
    logits = logits.CreateView({total_draft_length, logits->shape[2]}, logits->dtype);
    logit_processor_->InplaceUpdateLogits(logits, generation_cfg, verify_request_mstates,
                                          request_ids, &cum_verify_lengths, &draft_output_tokens);

    // - Compute probability distributions.
    NDArray probs_on_device = logit_processor_->ComputeProbsFromLogits(
        logits, generation_cfg, request_ids, &cum_verify_lengths);

    std::vector<std::vector<SampleResult>> sample_results_arr = sampler_->BatchVerifyDraftTokens(
        probs_on_device, request_ids, cum_verify_lengths, generation_cfg, rngs, draft_output_tokens,
        draft_output_prob_dist);
    ICHECK_EQ(sample_results_arr.size(), num_rsentries);

    for (int i = 0; i < num_rsentries; ++i) {
      const std::vector<SampleResult>& sample_results = sample_results_arr[i];
      int accept_length = sample_results.size();
      for (SampleResult sample_result : sample_results) {
        rsentries[i]->mstates[verify_model_id_]->CommitToken(sample_result);
        rsentries[i]->mstates[draft_model_id_]->CommitToken(sample_result);
      }
      estate->stats.current_total_seq_len += accept_length;
      estate->stats.total_accepted_length += accept_length;
      // - Minus one because the last draft token has no kv cache entry
      // - Take max with 0 in case of all accepted.
      int rollback_length =
          std::max(cum_verify_lengths[i + 1] - cum_verify_lengths[i] - accept_length - 1, 0);
      // rollback kv cache
      // NOTE: when number of small models is more than 1 (in the future),
      // it is possible to re-compute prefill for the small models.
      if (rollback_length > 0) {
        models_[verify_model_id_]->PopNFromKVCache(
            rsentries[i]->mstates[verify_model_id_]->internal_id, rollback_length);
        models_[draft_model_id_]->PopNFromKVCache(
            rsentries[i]->mstates[draft_model_id_]->internal_id, rollback_length);
      }
    }

    // clear the draft model state entries
    for (int i = 0; i < num_rsentries; ++i) {
      rsentries[i]->mstates[draft_model_id_]->RemoveAllDraftTokens();
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

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
    std::vector<RequestStateEntry> running_rsentries = GetRunningRequestStateEntries(estate);
    std::vector<int> num_page_requirement;
    num_page_requirement.reserve(running_rsentries.size());
    for (const RequestStateEntry& rsentry : running_rsentries) {
      int draft_length = rsentry->mstates[draft_model_id_]->draft_output_tokens.size();
      int num_require_pages =
          (draft_length + kv_cache_config_->page_size - 1) / kv_cache_config_->page_size;
      draft_lengths.push_back(draft_length);
      num_page_requirement.push_back(num_require_pages);
      total_draft_length += draft_length;
      total_required_pages += num_require_pages;
    }
    while (!CanVerify(total_required_pages)) {
      RequestStateEntry preempted =
          PreemptLastRunningRequestStateEntry(estate, models_, trace_recorder_);
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

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The kv cache config. */
  KVCacheConfig kv_cache_config_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Random number generator. */
  RandomGenerator& rng_;
  /*! \brief The ids of verify/draft models. */
  const int verify_model_id_ = 0;
  const int draft_model_id_ = 1;
  const float eps_ = 1e-5;
};

EngineAction EngineAction::BatchVerify(Array<Model> models, LogitProcessor logit_processor,
                                       Sampler sampler, KVCacheConfig kv_cache_config,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<BatchVerifyActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler), std::move(kv_cache_config),
      std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
