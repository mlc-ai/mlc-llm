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
  explicit BatchVerifyActionObj(Array<Model> models, Sampler sampler, KVCacheConfig kv_cache_config,
                                int max_single_sequence_length,
                                Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        sampler_(std::move(sampler)),
        kv_cache_config_(std::move(kv_cache_config)),
        max_single_sequence_length_(max_single_sequence_length),
        trace_recorder_(std::move(trace_recorder)),
        rng_(RandomGenerator::GetInstance()) {}

  Array<Request> Step(EngineState estate) final {
    // - Only run spec decode when there are two models (llm+ssm) and >=1 running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    const auto& [requests, rstates, draft_lengths, total_draft_length] = GetDraftsToVerify(estate);
    ICHECK_EQ(requests.size(), rstates.size());
    ICHECK_EQ(requests.size(), draft_lengths.size());
    if (requests.empty()) {
      return {};
    }

    int num_requests = requests.size();
    Array<String> request_ids = requests.Map([](const Request& request) { return request->id; });
    auto tstart = std::chrono::high_resolution_clock::now();

    // - Get embedding and run verify.
    std::vector<int64_t> request_internal_ids;
    std::vector<int32_t> all_tokens_to_verify;
    Array<RequestModelState> verify_request_mstates;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    std::vector<std::vector<int>> draft_output_tokens;
    std::vector<std::vector<float>> draft_output_token_prob;
    std::vector<std::vector<NDArray>> draft_output_prob_dist;
    request_internal_ids.reserve(num_requests);
    all_tokens_to_verify.reserve(total_draft_length);
    verify_request_mstates.reserve(num_requests);
    rngs.reserve(num_requests);
    generation_cfg.reserve(num_requests);
    draft_output_tokens.reserve(num_requests);
    draft_output_token_prob.reserve(num_requests);
    draft_output_prob_dist.reserve(num_requests);

    for (int i = 0; i < num_requests; ++i) {
      RequestModelState verify_mstate = rstates[i]->mstates[verify_model_id_];
      RequestModelState draft_mstate = rstates[i]->mstates[draft_model_id_];
      request_internal_ids.push_back(verify_mstate->internal_id);
      ICHECK(!draft_lengths.empty());
      ICHECK_EQ(draft_lengths[i], draft_mstate->draft_output_tokens.size());
      ICHECK_EQ(draft_lengths[i], draft_mstate->draft_output_token_prob.size());
      ICHECK_EQ(draft_lengths[i], draft_mstate->draft_output_prob_dist.size());
      // the last committed token + all the draft tokens but the last one.
      all_tokens_to_verify.push_back(draft_mstate->committed_tokens.back());
      all_tokens_to_verify.insert(all_tokens_to_verify.end(),
                                  draft_mstate->draft_output_tokens.begin(),
                                  draft_mstate->draft_output_tokens.end() - 1);
      verify_request_mstates.push_back(verify_mstate);
      generation_cfg.push_back(requests[i]->generation_cfg);
      rngs.push_back(&rstates[i]->rng);
      draft_output_tokens.push_back(draft_mstate->draft_output_tokens);
      draft_output_token_prob.push_back(draft_mstate->draft_output_token_prob);
      draft_output_prob_dist.push_back(draft_mstate->draft_output_prob_dist);
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start verify embedding");
    NDArray embeddings = models_[verify_model_id_]->TokenEmbed(
        {IntTuple{all_tokens_to_verify.begin(), all_tokens_to_verify.end()}});
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify embedding");

    RECORD_EVENT(trace_recorder_, request_ids, "start verify");
    NDArray logits =
        models_[verify_model_id_]->BatchVerify(embeddings, request_internal_ids, draft_lengths);
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify");
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], total_draft_length);

    std::vector<int> cum_verify_lengths = {0};
    for (int i = 0; i < num_requests; ++i) {
      cum_verify_lengths.push_back(cum_verify_lengths.back() + draft_lengths[i]);
    }
    std::vector<std::vector<int32_t>> accepted_tokens_arr = sampler_->BatchVerifyDraftTokens(
        logits, cum_verify_lengths, models_[verify_model_id_], verify_request_mstates,
        generation_cfg, rngs, draft_output_tokens, draft_output_token_prob, draft_output_prob_dist);
    ICHECK_EQ(accepted_tokens_arr.size(), num_requests);

    for (int i = 0; i < num_requests; ++i) {
      const std::vector<int32_t>& accepted_tokens = accepted_tokens_arr[i];
      int accept_length = accepted_tokens.size();
      for (int32_t token_id : accepted_tokens) {
        rstates[i]->mstates[draft_model_id_]->CommitToken(token_id);
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
            rstates[i]->mstates[verify_model_id_]->internal_id, rollback_length);
        models_[draft_model_id_]->PopNFromKVCache(rstates[i]->mstates[draft_model_id_]->internal_id,
                                                  rollback_length);
      }
    }

    // clear the draft model states
    for (int i = 0; i < num_requests; ++i) {
      rstates[i]->mstates[draft_model_id_]->RemoveAllDraftTokens();
      rstates[i]->mstates[draft_model_id_]->draft_output_token_prob.clear();
      rstates[i]->mstates[draft_model_id_]->draft_output_prob_dist.clear();
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return requests;
  }

 private:
  /*! \brief Check if the drafts can be verified under conditions. */
  bool CanVerify(EngineState estate, int num_verify_req, int total_draft_length,
                 int num_required_pages, int num_available_pages) {
    int num_running_requests = estate->running_queue.size();
    ICHECK_LE(num_running_requests, kv_cache_config_->max_num_sequence);

    // No exceeding of the maximum allowed requests that can
    // run simultaneously.
    if (num_running_requests + num_verify_req > kv_cache_config_->max_num_sequence) {
      return false;
    }

    // NOTE: The conditions are heuristic and can be revised.
    // Cond 1: total input length <= max allowed single sequence length.
    // Cond 2: at least one verify can be performed.
    // Cond 3: number of total tokens does not exceed the limit
    int new_batch_size = num_running_requests + num_verify_req;
    return total_draft_length <= max_single_sequence_length_ &&
           num_required_pages <= num_available_pages &&
           estate->stats.current_total_seq_len + total_draft_length <=
               kv_cache_config_->max_total_sequence_length;
  }

  /*!
   * \brief Decide whether to run verify for the draft of each request.
   * \param estate The engine state.
   * \return The drafts to verify, together with their respective
   * state and input length.
   */
  std::tuple<Array<Request>, Array<RequestState>, std::vector<int>, int> GetDraftsToVerify(
      EngineState estate) {
    // - Try to verify pending requests.
    std::vector<Request> verify_requests;
    std::vector<RequestState> rstates;
    std::vector<int> draft_lengths;
    int total_draft_length = 0;
    int total_required_pages = 0;
    int num_available_pages = models_[verify_model_id_]->GetNumAvailablePages();

    int req_id = 1;
    for (; req_id <= static_cast<int>(estate->running_queue.size()); ++req_id) {
      Request request = estate->running_queue[req_id - 1];
      RequestState rstate = estate->GetRequestState(request);
      int draft_length = rstate->mstates[draft_model_id_]->draft_output_tokens.size();
      int num_require_pages =
          (draft_length + kv_cache_config_->page_size - 1) / kv_cache_config_->page_size;
      total_draft_length += draft_length;
      total_required_pages += num_require_pages;
      if (CanVerify(estate, req_id, total_draft_length, total_required_pages,
                    num_available_pages)) {
        verify_requests.push_back(request);
        rstates.push_back(rstate);
        draft_lengths.push_back(draft_length);
      } else {
        total_draft_length -= draft_length;
        total_required_pages -= num_require_pages;
        break;
      }
    }
    // preempt all the remaining requests
    while (req_id <= static_cast<int>(estate->running_queue.size())) {
      PreemptLastRunningRequest(estate, models_, trace_recorder_);
      req_id += 1;
    }

    return {verify_requests, rstates, draft_lengths, total_draft_length};
  }

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The kv cache config. */
  KVCacheConfig kv_cache_config_;
  /*! \brief The maximum allowed length of a single sequence. */
  int max_single_sequence_length_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Random number generator. */
  RandomGenerator& rng_;
  /*! \brief The ids of verify/draft models. */
  const int verify_model_id_ = 0;
  const int draft_model_id_ = 1;
  const float eps_ = 1e-9;
};

EngineAction EngineAction::BatchVerify(Array<Model> models, Sampler sampler,
                                       KVCacheConfig kv_cache_config,
                                       int max_single_sequence_length,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<BatchVerifyActionObj>(
      std::move(models), std::move(sampler), std::move(kv_cache_config),
      std::move(max_single_sequence_length), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
