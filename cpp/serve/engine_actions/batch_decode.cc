/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_decode.cc
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
      PreemptLastRunningRequest(estate);
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running requests at a time.
    int num_requests = estate->running_queue.size();
    // Check if the requests ids are in an ascending order.
    for (int i = 1; i < num_requests; ++i) {
      ICHECK_GT(estate->GetRequestState(estate->running_queue[i])->mstates[0]->request_id,
                estate->GetRequestState(estate->running_queue[i - 1])->mstates[0]->request_id);
    }

    estate->stats.current_total_seq_len += num_requests;
    // Collect
    // - the last committed token,
    // - the request states,
    // - the sampling parameters,
    // of each request.
    std::vector<int> input_tokens;
    Array<String> request_ids;
    Array<RequestModelState> mstates;
    Array<GenerationConfig> generation_cfg;
    input_tokens.reserve(num_requests);
    request_ids.reserve(num_requests);
    mstates.reserve(num_requests);
    generation_cfg.reserve(num_requests);
    for (Request request : estate->running_queue) {
      RequestState rstate = estate->GetRequestState(request);
      input_tokens.push_back(rstate->mstates[0]->committed_tokens.back());
      request_ids.push_back(request->id);
      mstates.push_back(rstate->mstates[0]);
      generation_cfg.push_back(request->generation_cfg);
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
    NDArray logits = models_[0]->BatchDecode(embeddings);
    RECORD_EVENT(trace_recorder_, request_ids, "finish decode");
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
    ICHECK_EQ(logits->shape[1], 1);

    // - Sample tokens.
    RECORD_EVENT(trace_recorder_, request_ids, "start sampling");
    std::vector<int32_t> next_tokens =
        sampler_->SampleTokens(logits, models_[0], mstates, generation_cfg);
    RECORD_EVENT(trace_recorder_, request_ids, "finish sampling");
    ICHECK_EQ(next_tokens.size(), num_requests);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_requests; ++i) {
      mstates[i]->committed_tokens.push_back(next_tokens[i]);
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
   * \brief Preempt the last running requests from `running_queue`,
   * moving it from running request set to the foremost of waiting
   * request queue.
   */
  void PreemptLastRunningRequest(EngineState estate) {
    Request request = estate->running_queue.back();

    // Remove from models.
    // - Reset internal `request_id` of states.
    // - Clear model speculation draft.
    // - Update `inputs` for future prefill.
    RequestState rstate = estate->GetRequestState(request);
    RECORD_EVENT(trace_recorder_, rstate->request->id, "preempt");
    int req_id = rstate->mstates[0]->request_id;
    estate->stats.current_total_seq_len -=
        request->input_total_length + rstate->mstates[0]->committed_tokens.size() - 1;
    for (RequestModelState mstate : rstate->mstates) {
      mstate->request_id = -1;
      mstate->draft_output_tokens.clear();
      mstate->draft_output_token_prob.clear();
      mstate->draft_output_prob_dist.clear();
      ICHECK(mstate->inputs.empty());
      ICHECK(!mstate->committed_tokens.empty());

      Array<Data> inputs = request->inputs;
      if (const auto* token_input = inputs.back().as<TokenDataNode>()) {
        // Merge the TokenData so that a single time TokenEmbed is needed.
        std::vector<int> token_ids{token_input->token_ids->data,
                                   token_input->token_ids->data + token_input->token_ids.size()};
        token_ids.insert(token_ids.end(), mstate->committed_tokens.begin(),
                         mstate->committed_tokens.end());
        inputs.Set(inputs.size() - 1, TokenData(token_ids));
      } else {
        inputs.push_back(TokenData(mstate->committed_tokens));
      }
      mstate->inputs = std::move(inputs);
    }
    RemoveRequestFromModel(estate, req_id, models_);

    // Move from running queue to the front of waiting queue.
    estate->running_queue.erase(estate->running_queue.end() - 1);
    estate->waiting_queue.insert(estate->waiting_queue.begin(), request);
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
