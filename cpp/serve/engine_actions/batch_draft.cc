/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_draft.cc
 */

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
  explicit BatchDraftActionObj(Array<Model> models, Sampler sampler,
                               Optional<EventTraceRecorder> trace_recorder, int draft_length)
      : models_(std::move(models)),
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

    // Preempt requests when decode cannot apply.
    while (!CanDecode(estate->running_queue.size())) {
      PreemptLastRunningRequest(estate, models_, trace_recorder_);
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running requests at a time.
    int num_requests = estate->running_queue.size();
    Array<String> request_ids;
    std::vector<int64_t> request_internal_ids;
    Array<GenerationConfig> generation_cfg;
    Array<RequestState> rstates;
    request_ids.reserve(num_requests);
    request_internal_ids.reserve(num_requests);
    generation_cfg.reserve(num_requests);
    rstates.reserve(num_requests);
    for (const Request& request : estate->running_queue) {
      RequestState rstate = estate->GetRequestState(request);
      request_ids.push_back(request->id);
      rstates.push_back(rstate);
      request_internal_ids.push_back(rstate->mstates[0]->internal_id);
      generation_cfg.push_back(request->generation_cfg);
    }

    // The first model doesn't get involved in draft proposal.
    for (int model_id = 1; model_id < static_cast<int>(models_.size()); ++model_id) {
      // Collect
      // - the last committed token,
      // - the request states,
      // - the sampling parameters,
      // of each request.
      std::vector<int> input_tokens;
      Array<RequestModelState> mstates =
          rstates.Map([model_id](const RequestState& rstate) { return rstate->mstates[model_id]; });
      input_tokens.reserve(num_requests);
      // draft_length_ rounds of draft proposal.
      for (int draft_id = 0; draft_id < draft_length_; ++draft_id) {
        // prepare new input tokens
        input_tokens.clear();
        for (int i = 0; i < num_requests; ++i) {
          // The first draft proposal uses the last committed token.
          input_tokens.push_back(draft_id == 0 ? mstates[i]->committed_tokens.back()
                                               : mstates[i]->draft_output_tokens.back());
        }

        // - Compute embeddings.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal embedding");
        NDArray embeddings =
            models_[model_id]->TokenEmbed({IntTuple{input_tokens.begin(), input_tokens.end()}});
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal embedding");
        ICHECK_EQ(embeddings->ndim, 3);
        ICHECK_EQ(embeddings->shape[0], 1);
        ICHECK_EQ(embeddings->shape[1], num_requests);
        embeddings =
            embeddings.CreateView({num_requests, 1, embeddings->shape[2]}, embeddings->dtype);

        // - Invoke model decode.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal decode");
        NDArray logits = models_[model_id]->BatchDecode(embeddings, request_internal_ids);
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal decode");
        ICHECK_EQ(logits->ndim, 3);
        ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
        ICHECK_EQ(logits->shape[1], 1);

        // - Sample tokens.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal sampling");
        std::vector<NDArray> prob_dist;
        std::vector<float> token_probs;
        std::vector<int32_t> next_tokens = sampler_->BatchSampleTokens(
            logits, models_[model_id], mstates, generation_cfg, &prob_dist, &token_probs);
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal sampling");
        ICHECK_EQ(next_tokens.size(), num_requests);

        // - Update the draft tokens, prob dist, token probs of states.
        for (int i = 0; i < num_requests; ++i) {
          mstates[i]->AddDraftToken(next_tokens[i]);
          mstates[i]->draft_output_prob_dist.push_back(prob_dist[i]);
          mstates[i]->draft_output_token_prob.push_back(token_probs[i]);
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
  bool CanDecode(int num_requests) {
    // The first model is not involved in draft proposal.
    for (int model_id = 1; model_id < static_cast<int>(models_.size()); ++model_id) {
      // Check if the model has enough available pages.
      int num_available_pages = models_[model_id]->GetNumAvailablePages();
      if (num_requests > num_available_pages) {
        return false;
      }
    }
    return true;
  }

  /*! \brief The model to run draft generation in speculative decoding. */
  Array<Model> models_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Draft proposal length */
  int draft_length_;
};

EngineAction EngineAction::BatchDraft(Array<Model> models, Sampler sampler,
                                      Optional<EventTraceRecorder> trace_recorder,
                                      int draft_length) {
  return EngineAction(make_object<BatchDraftActionObj>(std::move(models), std::move(sampler),
                                                       std::move(trace_recorder), draft_length));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
