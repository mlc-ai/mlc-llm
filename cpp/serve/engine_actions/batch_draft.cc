/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_draft.cc
 */

#include <numeric>

#include "../config.h"
#include "../model.h"
#include "../sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs draft proposal for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 */
class BatchDraftActionObj : public EngineActionObj {
 public:
  explicit BatchDraftActionObj(Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
                               Optional<EventTraceRecorder> trace_recorder, int draft_length)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        trace_recorder_(std::move(trace_recorder)),
        draft_length_(draft_length) {
    ICHECK_GT(draft_length_, 0);
  }

  Array<Request> Step(EngineState estate) final {
    // - Only run spec decode when there are two models (llm+ssm) and >=1 running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    // Preempt request state entries when decode cannot apply.
    std::vector<RequestStateEntry> running_rsentries = GetRunningRequestStateEntries(estate);
    while (!CanDecode(running_rsentries.size())) {
      RequestStateEntry preempted =
          PreemptLastRunningRequestStateEntry(estate, models_, trace_recorder_);
      if (preempted.same_as(running_rsentries.back())) {
        running_rsentries.pop_back();
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    int num_rsentries = running_rsentries.size();
    Array<String> request_ids;
    std::vector<int64_t> request_internal_ids;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    request_ids.reserve(num_rsentries);
    request_internal_ids.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    for (const RequestStateEntry& rsentry : running_rsentries) {
      request_ids.push_back(rsentry->request->id);
      request_internal_ids.push_back(rsentry->mstates[0]->internal_id);
      generation_cfg.push_back(rsentry->request->generation_cfg);
      rngs.push_back(&rsentry->rng);
    }

    // The first model doesn't get involved in draft proposal.
    for (int model_id = 1; model_id < static_cast<int>(models_.size()); ++model_id) {
      // Collect
      // - the last committed token,
      // - the request model state
      // of each request.
      std::vector<int> input_tokens;
      Array<RequestModelState> mstates;
      input_tokens.reserve(num_rsentries);
      mstates.reserve(num_rsentries);
      for (const RequestStateEntry& rsentry : running_rsentries) {
        mstates.push_back(rsentry->mstates[model_id]);
      }
      // draft_length_ rounds of draft proposal.
      for (int draft_id = 0; draft_id < draft_length_; ++draft_id) {
        // prepare new input tokens
        input_tokens.clear();
        for (int i = 0; i < num_rsentries; ++i) {
          // The first draft proposal uses the last committed token.
          input_tokens.push_back(
              draft_id == 0 ? mstates[i]->committed_tokens.back().sampled_token_id.first
                            : mstates[i]->draft_output_tokens.back().sampled_token_id.first);
        }

        // - Compute embeddings.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal embedding");
        ObjectRef embeddings =
            models_[model_id]->TokenEmbed({IntTuple{input_tokens.begin(), input_tokens.end()}});
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal embedding");

        // - Invoke model decode.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal decode");
        NDArray logits = models_[model_id]->BatchDecode(embeddings, request_internal_ids);
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal decode");
        ICHECK_EQ(logits->ndim, 3);
        ICHECK_EQ(logits->shape[0], num_rsentries);
        ICHECK_EQ(logits->shape[1], 1);

        // - Update logits.
        logits = logits.CreateView({num_rsentries, logits->shape[2]}, logits->dtype);
        logit_processor_->InplaceUpdateLogits(logits, generation_cfg, mstates, request_ids);

        // - Compute probability distributions.
        NDArray probs_on_device =
            logit_processor_->ComputeProbsFromLogits(logits, generation_cfg, request_ids);

        // - Sample tokens.
        // Fill range [0, num_rsentries) into `sample_indices`.
        std::vector<int> sample_indices(num_rsentries);
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::vector<NDArray> prob_dist;
        std::vector<SampleResult> sample_results = sampler_->BatchSampleTokens(
            probs_on_device, sample_indices, request_ids, generation_cfg, rngs, &prob_dist);
        ICHECK_EQ(sample_results.size(), num_rsentries);

        // - Add draft token to the state.
        for (int i = 0; i < num_rsentries; ++i) {
          mstates[i]->AddDraftToken(sample_results[i], prob_dist[i]);
          estate->stats.total_draft_length += 1;
        }
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return {};
  }

 private:
  /*! \brief Check if the input requests can be decoded under conditions. */
  bool CanDecode(int num_rsentries) {
    // The first model is not involved in draft proposal.
    for (int model_id = 1; model_id < static_cast<int>(models_.size()); ++model_id) {
      // Check if the model has enough available pages.
      int num_available_pages = models_[model_id]->GetNumAvailablePages();
      if (num_rsentries > num_available_pages) {
        return false;
      }
    }
    return true;
  }

  /*! \brief The model to run draft generation in speculative decoding. */
  Array<Model> models_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Draft proposal length */
  int draft_length_;
};

EngineAction EngineAction::BatchDraft(Array<Model> models, LogitProcessor logit_processor,
                                      Sampler sampler, Optional<EventTraceRecorder> trace_recorder,
                                      int draft_length) {
  return EngineAction(make_object<BatchDraftActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler), std::move(trace_recorder),
      draft_length));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
