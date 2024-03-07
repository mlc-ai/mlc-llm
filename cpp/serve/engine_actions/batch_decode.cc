/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_decode.cc
 */

#include <numeric>

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
  explicit BatchDecodeActionObj(Array<Model> models, LogitProcessor logit_processor,
                                Sampler sampler, Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Do not run decode when there are multiple models or no running requests.
    if (models_.size() > 1 || estate->running_queue.empty()) {
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

    // NOTE: Right now we only support decode all the running request states at a time.
    int num_rsentries = running_rsentries.size();
    estate->stats.current_total_seq_len += num_rsentries;
    // Collect
    // - the last committed token,
    // - the request id,
    // - the generation config,
    // - the random number generator,
    // of each request state entry.
    std::vector<int> input_tokens;
    Array<String> request_ids;
    std::vector<int64_t> request_internal_ids;
    Array<RequestModelState> mstates;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    input_tokens.reserve(num_rsentries);
    request_ids.reserve(num_rsentries);
    request_internal_ids.reserve(num_rsentries);
    mstates.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    rngs.reserve(num_rsentries);
    for (const RequestStateEntry& rsentry : running_rsentries) {
      input_tokens.push_back(rsentry->mstates[0]->committed_tokens.back().sampled_token_id.first);
      request_ids.push_back(rsentry->request->id);
      request_internal_ids.push_back(rsentry->mstates[0]->internal_id);
      mstates.push_back(rsentry->mstates[0]);
      generation_cfg.push_back(rsentry->request->generation_cfg);
      rngs.push_back(&rsentry->rng);
    }

    // - Compute embeddings.
    RECORD_EVENT(trace_recorder_, request_ids, "start embedding");
    ObjectRef embeddings =
        models_[0]->TokenEmbed({IntTuple{input_tokens.begin(), input_tokens.end()}});
    RECORD_EVENT(trace_recorder_, request_ids, "finish embedding");

    // - Invoke model decode.
    RECORD_EVENT(trace_recorder_, request_ids, "start decode");
    NDArray logits = models_[0]->BatchDecode(embeddings, request_internal_ids);
    RECORD_EVENT(trace_recorder_, request_ids, "finish decode");
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
    std::vector<SampleResult> sample_results = sampler_->BatchSampleTokens(
        probs_on_device, sample_indices, request_ids, generation_cfg, rngs);
    ICHECK_EQ(sample_results.size(), num_rsentries);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_rsentries; ++i) {
      mstates[i]->CommitToken(sample_results[i]);
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return estate->running_queue;
  }

 private:
  /*! \brief Check if the input request state entries can be decoded under conditions. */
  bool CanDecode(int num_rsentries) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries <= num_available_pages;
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
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
};

EngineAction EngineAction::BatchDecode(Array<Model> models, LogitProcessor logit_processor,
                                       Sampler sampler,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(
      make_object<BatchDecodeActionObj>(std::move(models), std::move(logit_processor),
                                        std::move(sampler), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
