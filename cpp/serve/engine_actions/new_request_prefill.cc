/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/new_request_prefill.cc
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
 * \brief The action that prefills requests in the `waiting_queue` of
 * the engine state.
 */
class NewRequestPrefillActionObj : public EngineActionObj {
 public:
  explicit NewRequestPrefillActionObj(Array<Model> models, LogitProcessor logit_processor,
                                      Sampler sampler, KVCacheConfig kv_cache_config,
                                      EngineMode engine_mode,
                                      Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        kv_cache_config_(std::move(kv_cache_config)),
        engine_mode_(std::move(engine_mode)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Find the requests in `waiting_queue` that can prefill in this step.
    auto [rstates, prefill_lengths] = GetRequestStatesToPrefill(estate);
    ICHECK_EQ(rstates.size(), prefill_lengths.size());
    if (rstates.empty()) {
      return {};
    }

    int num_rstates = rstates.size();
    auto tstart = std::chrono::high_resolution_clock::now();

    // - Update status of request states from pending to alive.
    Array<String> request_ids;
    std::vector<RequestState> rstates_of_requests;
    request_ids.reserve(num_rstates);
    rstates_of_requests.reserve(num_rstates);
    for (RequestStateEntry rstate : rstates) {
      const Request& request = rstate->request;
      RequestState request_rstate = estate->GetRequestState(request);
      request_ids.push_back(request->id);
      rstate->status = RequestStateStatus::kAlive;

      // - Remove the request from waiting queue if all its request states are now alive.
      // - Add the request to running queue if all its request states were pending.
      bool alive_state_existed = false;
      for (const RequestStateEntry& request_state : request_rstate->entries) {
        if (request_state->status == RequestStateStatus::kAlive && !request_state.same_as(rstate)) {
          alive_state_existed = true;
        }
      }
      if (!alive_state_existed) {
        estate->running_queue.push_back(request);
      }
      rstates_of_requests.push_back(std::move(request_rstate));
    }

    // - Get embedding and run prefill for each model.
    NDArray logits_for_sample{nullptr};
    for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
      Array<NDArray> embeddings;
      std::vector<int64_t> request_internal_ids;
      embeddings.reserve(num_rstates);
      request_internal_ids.reserve(num_rstates);
      for (int i = 0; i < num_rstates; ++i) {
        RequestModelState mstate = rstates[i]->mstates[model_id];
        ICHECK_EQ(mstate->GetInputLength(), prefill_lengths[i]);
        ICHECK(mstate->draft_output_tokens.empty());
        ICHECK(mstate->draft_output_prob_dist.empty());
        ICHECK(!mstate->inputs.empty());
        // Add the sequence to the model, or fork the sequence from its parent.
        if (rstates[i]->parent_idx == -1) {
          models_[model_id]->AddNewSequence(mstate->internal_id);
        } else {
          models_[model_id]->ForkSequence(rstates_of_requests[i]
                                              ->entries[rstates[i]->parent_idx]
                                              ->mstates[model_id]
                                              ->internal_id,
                                          mstate->internal_id);
        }
        request_internal_ids.push_back(mstate->internal_id);
        RECORD_EVENT(trace_recorder_, rstates[i]->request->id, "start embedding");
        for (int i = 0; i < static_cast<int>(mstate->inputs.size()); ++i) {
          embeddings.push_back(mstate->inputs[i]->GetEmbedding(models_[model_id]));
        }
        RECORD_EVENT(trace_recorder_, rstates[i]->request->id, "finish embedding");
        // Clean up `inputs` after prefill
        mstate->inputs.clear();
      }

      RECORD_EVENT(trace_recorder_, request_ids, "start prefill");
      NDArray logits =
          models_[model_id]->BatchPrefill(embeddings, request_internal_ids, prefill_lengths);
      RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");
      ICHECK_EQ(logits->ndim, 3);
      ICHECK_EQ(logits->shape[0], 1);
      ICHECK_EQ(logits->shape[1], num_rstates);

      if (model_id == 0) {
        // We only need to sample for model 0 in prefill.
        logits_for_sample = logits;
      }
    }

    // - Update logits.
    ICHECK(logits_for_sample.defined());
    Array<GenerationConfig> generation_cfg;
    Array<RequestModelState> mstates_for_logitproc;
    generation_cfg.reserve(num_rstates);
    mstates_for_logitproc.reserve(num_rstates);
    for (int i = 0; i < num_rstates; ++i) {
      generation_cfg.push_back(rstates[i]->request->generation_cfg);
      mstates_for_logitproc.push_back(rstates[i]->mstates[0]);
    }
    logits_for_sample = logits_for_sample.CreateView({num_rstates, logits_for_sample->shape[2]},
                                                     logits_for_sample->dtype);
    logit_processor_->InplaceUpdateLogits(logits_for_sample, generation_cfg, mstates_for_logitproc,
                                          request_ids);

    // - Compute probability distributions.
    NDArray probs_on_device =
        logit_processor_->ComputeProbsFromLogits(logits_for_sample, generation_cfg, request_ids);

    // - Sample tokens.
    //   For rstates which are depended by other states, sample
    //   one token for each rstate that is depending.
    //   Otherwise, sample a token for the current rstate.
    std::vector<int> sample_indices;
    std::vector<RequestStateEntry> rsentries_for_sample;
    std::vector<RandomGenerator*> rngs;
    sample_indices.reserve(num_rstates);
    rsentries_for_sample.reserve(num_rstates);
    rngs.reserve(num_rstates);
    request_ids.clear();
    generation_cfg.clear();
    for (int i = 0; i < num_rstates; ++i) {
      estate->stats.current_total_seq_len += prefill_lengths[i];
      const RequestStateEntry& rstate = rstates[i];
      for (int child_idx : rstate->child_indices) {
        if (rstates_of_requests[i]->entries[child_idx]->mstates[0]->committed_tokens.empty()) {
          // If rstates_of_requests[i][child_idx] has no committed token,
          // the prefill of the current rstate will unblock rstates_of_requests[i][child_idx],
          // and thus we want to sample a token for rstates_of_requests[i][child_idx].
          sample_indices.push_back(i);
          rsentries_for_sample.push_back(rstates_of_requests[i]->entries[child_idx]);
          request_ids.push_back(rstate->request->id);
          generation_cfg.push_back(rstate->request->generation_cfg);
          rngs.push_back(&rstates_of_requests[i]->entries[child_idx]->rng);

          ICHECK(rstates_of_requests[i]->entries[child_idx]->status ==
                 RequestStateStatus::kPending);
          rstates_of_requests[i]->entries[child_idx]->status = RequestStateStatus::kAlive;
          for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
            models_[model_id]->ForkSequence(
                rstate->mstates[model_id]->internal_id,
                rstates_of_requests[i]->entries[child_idx]->mstates[model_id]->internal_id);
          }
        }
      }
      if (rstate->child_indices.empty()) {
        // If rstate has no child, we sample a token for itself.
        sample_indices.push_back(i);
        rsentries_for_sample.push_back(rstate);
        request_ids.push_back(rstate->request->id);
        generation_cfg.push_back(rstate->request->generation_cfg);
        rngs.push_back(&rstate->rng);
      }
    }
    std::vector<SampleResult> sample_results = sampler_->BatchSampleTokens(
        probs_on_device, sample_indices, request_ids, generation_cfg, rngs);
    ICHECK_EQ(sample_results.size(), rsentries_for_sample.size());

    // - Update the committed tokens of states.
    // - If a request is first-time prefilled, set the prefill finish time.
    auto tnow = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
      for (const RequestModelState& mstate : rsentries_for_sample[i]->mstates) {
        mstate->CommitToken(sample_results[i]);
      }
      if (rsentries_for_sample[i]->mstates[0]->committed_tokens.size() == 1) {
        rsentries_for_sample[i]->tprefill_finish = tnow;
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_prefill_time += static_cast<double>((tend - tstart).count()) / 1e9;

    std::vector<Request> processed_requests;
    {
      processed_requests.reserve(num_rstates);
      std::unordered_set<const RequestNode*> dedup_map;
      for (int i = 0; i < static_cast<int>(rstates.size()); ++i) {
        const RequestStateEntry& rstate = rstates[i];
        if (dedup_map.find(rstate->request.get()) != dedup_map.end()) {
          continue;
        }
        dedup_map.insert(rstate->request.get());
        processed_requests.push_back(rstate->request);

        bool pending_state_exists = false;
        for (const RequestStateEntry& request_state : rstates_of_requests[i]->entries) {
          if (request_state->status == RequestStateStatus::kPending) {
            pending_state_exists = true;
            break;
          }
        }
        if (!pending_state_exists) {
          auto it = std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(),
                              rstate->request);
          ICHECK(it != estate->waiting_queue.end());
          estate->waiting_queue.erase(it);
        }
      }
    }
    return processed_requests;
  }

 private:
  /*!
   * \brief Find one or multiple request states to run prefill.
   * \param estate The engine state.
   * \return The requests to prefill, together with their respective
   * state and input length.
   */
  std::tuple<Array<RequestStateEntry>, std::vector<int>> GetRequestStatesToPrefill(
      EngineState estate) {
    if (estate->waiting_queue.empty()) {
      // No request to prefill.
      return {{}, {}};
    }

    // - Try to prefill pending requests.
    std::vector<RequestStateEntry> rsentries_to_prefill;
    std::vector<int> prefill_lengths;
    int total_input_length = 0;
    int total_required_pages = 0;
    int num_available_pages = models_[0]->GetNumAvailablePages();
    int num_running_rsentries = GetRunningRequestStateEntries(estate).size();

    int num_prefill_rsentries = 0;
    for (const Request& request : estate->waiting_queue) {
      RequestState rstate = estate->GetRequestState(request);
      bool prefill_stops = false;
      for (const RequestStateEntry& rsentry : rstate->entries) {
        // A request state entry can be prefilled only when:
        // - it has inputs, and
        // - it is pending, and
        // - it has no parent or its parent is alive.
        if (rsentry->mstates[0]->inputs.empty() ||
            rsentry->status != RequestStateStatus::kPending ||
            (rsentry->parent_idx != -1 &&
             rstate->entries[rsentry->parent_idx]->status == RequestStateStatus::kPending)) {
          continue;
        }

        int input_length = rsentry->mstates[0]->GetInputLength();
        int num_require_pages =
            (input_length + kv_cache_config_->page_size - 1) / kv_cache_config_->page_size;
        total_input_length += input_length;
        total_required_pages += num_require_pages;
        if (CanPrefill(estate, num_prefill_rsentries + 1 + rsentry->child_indices.size(),
                       total_input_length, total_required_pages, num_available_pages,
                       num_running_rsentries)) {
          rsentries_to_prefill.push_back(rsentry);
          prefill_lengths.push_back(input_length);
          num_prefill_rsentries += 1 + rsentry->child_indices.size();
        } else {
          total_input_length -= input_length;
          total_required_pages -= num_require_pages;
          prefill_stops = true;
          break;
        }
      }
      if (prefill_stops) {
        break;
      }
    }

    return {rsentries_to_prefill, prefill_lengths};
  }

  /*! \brief Check if the input requests can be prefilled under conditions. */
  bool CanPrefill(EngineState estate, int num_prefill_rsentries, int total_input_length,
                  int num_required_pages, int num_available_pages, int num_running_rsentries) {
    ICHECK_LE(num_running_rsentries, kv_cache_config_->max_num_sequence);

    // No exceeding of the maximum allowed requests that can
    // run simultaneously.
    int spec_factor = engine_mode_->enable_speculative ? engine_mode_->spec_draft_length : 1;
    if ((num_running_rsentries + num_prefill_rsentries) * spec_factor >
        kv_cache_config_->max_num_sequence) {
      return false;
    }

    // NOTE: The conditions are heuristic and can be revised.
    // Cond 1: total input length <= prefill chunk size.
    // Cond 2: at least one decode can be performed after prefill.
    // Cond 3: number of total tokens after 8 times of decode does not
    // exceed the limit, where 8 is a watermark number can
    // be configured and adjusted in the future.
    int new_batch_size = num_running_rsentries + num_prefill_rsentries;
    return total_input_length <= kv_cache_config_->prefill_chunk_size &&
           num_required_pages + new_batch_size <= num_available_pages &&
           estate->stats.current_total_seq_len + total_input_length + 8 * new_batch_size <=
               kv_cache_config_->max_total_sequence_length;
  }

  /*! \brief The models to run prefill in. */
  Array<Model> models_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The KV cache config to help decide prefill is doable. */
  KVCacheConfig kv_cache_config_;
  /*! \brief The engine operation mode. */
  EngineMode engine_mode_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
};

EngineAction EngineAction::NewRequestPrefill(Array<Model> models, LogitProcessor logit_processor,
                                             Sampler sampler, KVCacheConfig kv_cache_config,
                                             EngineMode engine_mode,
                                             Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<NewRequestPrefillActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler), std::move(kv_cache_config),
      std::move(engine_mode), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
