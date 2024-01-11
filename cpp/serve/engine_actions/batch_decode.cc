/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_decode.cc
 */

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
 * \brief The action that runs one-step decode for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 * \note The BatchDecode action **does not** take effect for speculative
 * decoding scenarios where there are multiple models. For speculative
 * decoding in the future, we will use other specific actions.
 */
class BatchDecodeActionObj : public EngineActionObj {
 public:
  explicit BatchDecodeActionObj(Array<Model> models, Sampler sampler,
                                Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        sampler_(std::move(sampler)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Do not run decode when there are multiple models or no running requests.
    if (models_.size() > 1 || estate->running_queue.empty()) {
      return {};
    }

    // Preempt requests when decode cannot apply.
    int num_available_pages = models_[0]->GetNumAvailablePages();
    while (!CanDecode(estate->running_queue.size())) {
      PreemptLastRunningRequest(estate, models_, trace_recorder_);
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running requests at a time.
    int num_requests = estate->running_queue.size();
    estate->stats.current_total_seq_len += num_requests;
    // Collect
    // - the last committed token,
    // - the request states,
    // - the sampling parameters,
    // of each request.
    std::vector<int> input_tokens;
    Array<String> request_ids;
    std::vector<int64_t> request_internal_ids;
    Array<RequestModelState> mstates;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    input_tokens.reserve(num_requests);
    request_ids.reserve(num_requests);
    request_internal_ids.reserve(num_requests);
    mstates.reserve(num_requests);
    generation_cfg.reserve(num_requests);
    rngs.reserve(num_requests);
    for (Request request : estate->running_queue) {
      RequestState rstate = estate->GetRequestState(request);
      input_tokens.push_back(rstate->mstates[0]->committed_tokens.back());
      request_ids.push_back(request->id);
      request_internal_ids.push_back(rstate->mstates[0]->internal_id);
      mstates.push_back(rstate->mstates[0]);
      generation_cfg.push_back(request->generation_cfg);
      rngs.push_back(&rstate->rng);
    }

    // - Compute embeddings.
    RECORD_EVENT(trace_recorder_, request_ids, "start embedding");
    NDArray embeddings =
        models_[0]->TokenEmbed({IntTuple{input_tokens.begin(), input_tokens.end()}});
    RECORD_EVENT(trace_recorder_, request_ids, "finish embedding");
    ICHECK_EQ(embeddings->ndim, 3);
    ICHECK_EQ(embeddings->shape[0], 1);
    ICHECK_EQ(embeddings->shape[1], num_requests);
    embeddings = embeddings.CreateView({num_requests, 1, embeddings->shape[2]}, embeddings->dtype);

    // - Invoke model decode.
    RECORD_EVENT(trace_recorder_, request_ids, "start decode");
    NDArray logits = models_[0]->BatchDecode(embeddings, request_internal_ids);
    RECORD_EVENT(trace_recorder_, request_ids, "finish decode");
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
    ICHECK_EQ(logits->shape[1], 1);

    // - Sample tokens.
    RECORD_EVENT(trace_recorder_, request_ids, "start sampling");
    std::vector<int32_t> next_tokens =
        sampler_->BatchSampleTokens(logits, models_[0], mstates, generation_cfg, rngs);
    RECORD_EVENT(trace_recorder_, request_ids, "finish sampling");
    ICHECK_EQ(next_tokens.size(), num_requests);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_requests; ++i) {
      mstates[i]->CommitToken(next_tokens[i]);
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return estate->running_queue;
  }

 private:
  /*! \brief Check if the input requests can be decoded under conditions. */
  bool CanDecode(int num_requests) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_requests <= num_available_pages;
  }

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
};

EngineAction EngineAction::BatchDecode(Array<Model> models, Sampler sampler,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<BatchDecodeActionObj>(std::move(models), std::move(sampler),
                                                        std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
