/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/eagle_batch_draft.cc
 */

#include <numeric>

#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
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
class EagleBatchDraftActionObj : public EngineActionObj {
 public:
  explicit EagleBatchDraftActionObj(Array<Model> models, LogitProcessor logit_processor,
                                    Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                    Optional<EventTraceRecorder> trace_recorder, int draft_length)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
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
      NDArray hidden_states_nd{nullptr};
      ObjectRef last_hidden_states{nullptr};
      ObjectRef hidden_states = model_workspaces_[model_id].hidden_states;
      // Concat last hidden_states
      std::vector<NDArray> previous_hidden_on_device;
      for (int i = 0; i < num_rsentries; ++i) {
        previous_hidden_on_device.push_back(mstates[i]->draft_last_hidden_on_device.back());
      }
      hidden_states_nd =
          models_[model_id]->ConcatLastHidden(previous_hidden_on_device, &hidden_states);
      ICHECK_EQ(hidden_states_nd->ndim, 2);
      ICHECK_EQ(hidden_states_nd->shape[0], num_rsentries);
      hidden_states_nd = hidden_states_nd.CreateView(
          {hidden_states_nd->shape[0], 1, hidden_states_nd->shape[1]}, hidden_states_nd->dtype);
      last_hidden_states = hidden_states_nd;
      // The first draft token has been generated in prefill/verify stage
      for (int draft_id = 1; draft_id < draft_length_; ++draft_id) {
        // prepare new input tokens
        input_tokens.clear();
        for (int i = 0; i < num_rsentries; ++i) {
          ICHECK(!mstates[i]->draft_output_tokens.empty());
          input_tokens.push_back(mstates[i]->draft_output_tokens.back().sampled_token_id.first);
        }

        // - Compute embeddings.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal embedding");
        ObjectRef embeddings =
            models_[model_id]->TokenEmbed({IntTuple{input_tokens.begin(), input_tokens.end()}});
        RECORD_EVENT(trace_recorder_, request_ids, "finish proposal embedding");

        // - Invoke model decode.
        RECORD_EVENT(trace_recorder_, request_ids, "start proposal decode");
        ObjectRef fused_hidden_states = models_[model_id]->FuseEmbedHidden(
            embeddings, last_hidden_states, /*batch_size*/ num_rsentries, /*seq_len*/ 1);
        hidden_states_nd =
            models_[model_id]->BatchDecodeToLastHidden(fused_hidden_states, request_internal_ids);
        last_hidden_states = hidden_states_nd;
        NDArray logits;
        if (models_[model_id]->CanGetLogits()) {
          logits = models_[model_id]->GetLogits(hidden_states_nd, /*batch_size*/ num_rsentries,
                                                /*seq_len*/ 1);
        } else {
          // - Use base model's head.
          logits =
              models_[0]->GetLogits(hidden_states_nd, /*batch_size*/ num_rsentries, /*seq_len*/ 1);
        }
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
        NDArray renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
            probs_on_device, sample_indices, request_ids, generation_cfg);
        std::vector<SampleResult> sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
            renormalized_probs, sample_indices, request_ids, generation_cfg, rngs, &prob_dist);
        ICHECK_EQ(sample_results.size(), num_rsentries);

        // - Add draft token to the state.
        for (int i = 0; i < num_rsentries; ++i) {
          // - Slice hidden_states_for_sample
          NDArray last_hidden_on_device = GetTokenHidden(hidden_states_nd, i);
          CHECK(i < static_cast<int>(prob_dist.size()));
          CHECK(prob_dist[i].defined());
          mstates[i]->AddDraftToken(sample_results[i], prob_dist[i], last_hidden_on_device);
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

  /*!
   * \brief Get one item from a hidden_states array, which corresponds to the last token.
   * \param hidden_states The hidden_states of all the tokens.
   * \param token_pos The desired token position in the sequence.
   * \return The desired token's hidden_states
   */
  NDArray GetTokenHidden(NDArray hidden_states, int token_pos) {
    ICHECK_EQ(hidden_states->ndim, 3);
    NDArray last_hidden_on_device =
        NDArray::Empty({hidden_states->shape[2]}, hidden_states->dtype, hidden_states->device);

    int64_t ndata = hidden_states->shape[2];
    const int16_t* __restrict p_hidden =
        static_cast<int16_t*>(__builtin_assume_aligned(hidden_states->data, 2)) +
        (token_pos * ndata);

    last_hidden_on_device.CopyFromBytes(p_hidden, ndata * sizeof(int16_t));
    return last_hidden_on_device;
  }

  /*! \brief The model to run draft generation in speculative decoding. */
  Array<Model> models_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Workspace of each model. */
  std::vector<ModelWorkspace> model_workspaces_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Draft proposal length */
  int draft_length_;
};

EngineAction EngineAction::EagleBatchDraft(Array<Model> models, LogitProcessor logit_processor,
                                           Sampler sampler,
                                           std::vector<ModelWorkspace> model_workspaces,
                                           Optional<EventTraceRecorder> trace_recorder,
                                           int draft_length) {
  return EngineAction(make_object<EagleBatchDraftActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(trace_recorder), draft_length));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
