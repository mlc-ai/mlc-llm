/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action_commons.cc
 */

#include "action_commons.h"

#include <tvm/runtime/nvtx.h>

namespace mlc {
namespace llm {
namespace serve {

void RemoveRequestFromModel(EngineState estate, int64_t req_internal_id, Array<Model> models) {
  // Remove the request from all models (usually the KV cache).
  for (Model model : models) {
    model->RemoveSequence(req_internal_id);
  }
}

void ProcessFinishedRequestStateEntries(std::vector<RequestStateEntry> finished_rsentries,
                                        EngineState estate, Array<Model> models,
                                        int max_single_sequence_length) {
  NVTXScopedRange nvtx_scope("Process finished requests");
  // - Remove the finished request state entries.
  for (const RequestStateEntry& rsentry : finished_rsentries) {
    // The finished entry must be a leaf.
    ICHECK(rsentry->child_indices.empty());
    // Mark the status of this entry as finished.
    rsentry->status = RequestStateStatus::kFinished;
    // Remove the request state entry from all the models.
    if (estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
      if (!rsentry->request->generation_cfg->pinned) {
        // If the request is not pinned, recycle the request.
        estate->prefix_cache->RecycleSequence(
            rsentry->mstates[0]->internal_id, TypedPackedFunc<void()>([estate, models, rsentry]() {
              RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);
              estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
            }),
            /*lazy=*/true);
      }
      // If the request is pinned, do nothing over the prefix cache and KVCache. Let the data be
      // orphan data.
    } else {
      RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);
      estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
    }

    RequestState rstate = estate->GetRequestState(rsentry->request);
    int parent_idx = rsentry->parent_idx;
    while (parent_idx != -1) {
      bool all_children_finished = true;
      for (int child_idx : rstate->entries[parent_idx]->child_indices) {
        if (rstate->entries[child_idx]->status != RequestStateStatus::kFinished) {
          all_children_finished = false;
          break;
        }
      }
      if (!all_children_finished) {
        break;
      }

      // All the children of the parent request state entry have finished.
      // So we mark the parent entry as finished.
      rstate->entries[parent_idx]->status = RequestStateStatus::kFinished;
      // Remove the request state entry from all the models.
      if (estate->prefix_cache->HasSequence(rstate->entries[parent_idx]->mstates[0]->internal_id)) {
        if (!rsentry->request->generation_cfg->pinned) {
          // If the request is not pinned, recycle the request.
          estate->prefix_cache->RecycleSequence(
              rstate->entries[parent_idx]->mstates[0]->internal_id,
              TypedPackedFunc<void()>([estate, models, rstate, parent_idx]() {
                RemoveRequestFromModel(estate, rstate->entries[parent_idx]->mstates[0]->internal_id,
                                       models);
                estate->id_manager.RecycleId(rstate->entries[parent_idx]->mstates[0]->internal_id);
              }),
              /*lazy=*/true);
        }
        // If the request is pinned, do nothing over the prefix cache and KVCache. Let the data be
        // orphan data.
      } else {
        RemoveRequestFromModel(estate, rstate->entries[parent_idx]->mstates[0]->internal_id,
                               models);
        estate->id_manager.RecycleId(rstate->entries[parent_idx]->mstates[0]->internal_id);
      }
      // Climb up to the parent.
      parent_idx = rstate->entries[parent_idx]->parent_idx;
    }

    if (parent_idx == -1) {
      // Remove from running queue and engine state.
      auto it =
          std::find(estate->running_queue.begin(), estate->running_queue.end(), rsentry->request);
      ICHECK(it != estate->running_queue.end());
      estate->running_queue.erase(it);
      estate->request_states.erase(rsentry->request->id);

      // Update engine statistics.
      const RequestStateEntry& root_rsentry = rstate->entries[0];
      auto trequest_finish = std::chrono::high_resolution_clock::now();
      estate->stats.request_total_prefill_time +=
          static_cast<double>((root_rsentry->tprefill_finish - root_rsentry->tadd).count()) / 1e9;
      estate->stats.request_total_decode_time +=
          static_cast<double>((trequest_finish - root_rsentry->tprefill_finish).count()) / 1e9;
      for (const RequestStateEntry& entry : rstate->entries) {
        estate->stats.total_decode_length += entry->mstates[0]->committed_tokens.size();
      }
      // For a request, the first token in committed_tokens is generated by prefilling
      // and the rest are generated by decoding. So we subtract the first token.
      estate->stats.total_decode_length -= rsentry->request->generation_cfg->n;
    }
  }
}

void ActionStepPostProcess(Array<Request> requests, EngineState estate, Array<Model> models,
                           const Tokenizer& tokenizer,
                           FRequestStreamCallback request_stream_callback,
                           int64_t max_single_sequence_length,
                           Optional<EventTraceRecorder> trace_recorder) {
  NVTXScopedRange nvtx_scope("EngineAction postproc");
  std::vector<RequestStateEntry> finished_rsentries;
  finished_rsentries.reserve(requests.size());

  Array<RequestStreamOutput> callback_delta_outputs;
  callback_delta_outputs.reserve(requests.size());

  for (Request request : requests) {
    RequestState rstate = estate->GetRequestState(request);
    for (const RequestStateEntry& rsentry : rstate->entries) {
      if (!rsentry->mstates[0]->prefilled_inputs.empty()) {
        for (Data data : rsentry->mstates[0]->prefilled_inputs) {
          estate->stats.total_prefill_length += data->GetLength();
        }
      }
    }
  }
  for (Request request : requests) {
    RequestState rstate = estate->GetRequestState(request);
    for (const RequestStateEntry& rsentry : rstate->entries) {
      if (estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
        if (!rsentry->mstates[0]->prefilled_inputs.empty()) {
          // Notify the prefix cache of the newly prefilled data.
          for (Data data : rsentry->mstates[0]->prefilled_inputs) {
            const TokenDataNode* token_data = data.as<TokenDataNode>();
            RECORD_EVENT(trace_recorder, rsentry->request->id,
                         "prefix cache extend " + std::to_string(token_data->token_ids.size()) +
                             " prefilled tokens.");
            estate->prefix_cache->ExtendSequence(rsentry->mstates[0]->internal_id,
                                                 token_data->token_ids);
          }
          rsentry->mstates[0]->prefilled_inputs.clear();
        }
        if (rsentry->mstates[0]->cached_committed_tokens <
            rsentry->mstates[0]->committed_tokens.size() - 1) {
          // Notify the prefix cache of the newly decoded data, except the last token as it is not
          // in KVCache yet.
          std::vector<int64_t> tokens;
          tokens.reserve((rsentry->mstates[0]->committed_tokens.size() -
                          rsentry->mstates[0]->cached_committed_tokens));
          for (int i = rsentry->mstates[0]->cached_committed_tokens;
               i < rsentry->mstates[0]->committed_tokens.size() - 1; ++i) {
            tokens.push_back(rsentry->mstates[0]->committed_tokens[i].sampled_token_id.first);
          }
          RECORD_EVENT(trace_recorder, rsentry->request->id,
                       "prefix cache extend " + std::to_string(tokens.size()) + " decoded tokens.");
          estate->prefix_cache->ExtendSequence(rsentry->mstates[0]->internal_id, IntTuple(tokens));
          rsentry->mstates[0]->cached_committed_tokens =
              rsentry->mstates[0]->committed_tokens.size() - 1;
        }
      }
    }
  }

  // - Collect new generated tokens and finish reasons for requests.
  for (Request request : requests) {
    int n = request->generation_cfg->n;
    RequestState rstate = estate->GetRequestState(request);
    Array<IntTuple> group_delta_token_ids;
    Array<Array<String>> group_delta_logprob_json_strs;
    Array<Optional<String>> group_finish_reason;
    group_delta_token_ids.reserve(n);
    group_delta_logprob_json_strs.reserve(n);
    group_finish_reason.reserve(n);

    bool invoke_callback = false;
    for (int i = 0; i < n; ++i) {
      const RequestStateEntry& rsentry = n == 1 ? rstate->entries[0] : rstate->entries[i + 1];
      const DeltaRequestReturn& delta_request_ret =
          rsentry->GetReturnTokenIds(tokenizer, max_single_sequence_length);
      group_delta_token_ids.push_back(IntTuple{delta_request_ret.delta_token_ids.begin(),
                                               delta_request_ret.delta_token_ids.end()});
      group_delta_logprob_json_strs.push_back(delta_request_ret.delta_logprob_json_strs);
      group_finish_reason.push_back(delta_request_ret.finish_reason);
      if (delta_request_ret.finish_reason.defined()) {
        invoke_callback = true;
        finished_rsentries.push_back(rsentry);
      }

      if (!delta_request_ret.delta_token_ids.empty()) {
        invoke_callback = true;
      }
    }

    if (invoke_callback) {
      callback_delta_outputs.push_back(RequestStreamOutput(
          request->id, std::move(group_delta_token_ids),
          request->generation_cfg->logprobs > 0 ? std::move(group_delta_logprob_json_strs)
                                                : Optional<Array<Array<String>>>(),
          std::move(group_finish_reason)));
    }
  }

  {
    NVTXScopedRange nvtx_scope("Call request stream callback");
    // - Invoke the stream callback function once for all collected requests.
    request_stream_callback(callback_delta_outputs);
  }

  ProcessFinishedRequestStateEntries(std::move(finished_rsentries), std::move(estate),
                                     std::move(models), max_single_sequence_length);
}  // namespace serve

RequestStateEntry PreemptLastRunningRequestStateEntry(
    EngineState estate, const Array<Model>& models,
    Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager,
    Optional<EventTraceRecorder> trace_recorder) {
  ICHECK(!estate->running_queue.empty());
  Request request = estate->running_queue.back();

  // Find the last alive request state entry, which is what we want to preempt.
  RequestState rstate = estate->GetRequestState(request);
  int preempt_rstate_idx = -1;
  for (int i = static_cast<int>(rstate->entries.size()) - 1; i >= 0; --i) {
    if (rstate->entries[i]->status == RequestStateStatus::kAlive) {
      preempt_rstate_idx = i;
      break;
    }
  }
  ICHECK_NE(preempt_rstate_idx, -1);
  RequestStateEntry rsentry = rstate->entries[preempt_rstate_idx];
  // When the request state entry still has pending inputs,
  // it means the request is still in the waiting queue.
  bool partially_alive = !rsentry->mstates[0]->inputs.empty();

  // Remove from models.
  // - Clear model speculation draft.
  // - Update `inputs` for future prefill.
  RECORD_EVENT(trace_recorder, rsentry->request->id, "preempt");
  rsentry->status = RequestStateStatus::kPending;
  std::vector<int> draft_token_slots;
  for (RequestModelState mstate : rsentry->mstates) {
    if (draft_token_workspace_manager.defined()) {
      mstate->RemoveAllDraftTokens(&draft_token_slots);
      draft_token_workspace_manager.value()->FreeSlots(draft_token_slots);
    }
    std::vector<int32_t> committed_token_ids;
    committed_token_ids.reserve(mstate->committed_tokens.size());
    for (const SampleResult& committed_token : mstate->committed_tokens) {
      committed_token_ids.push_back(committed_token.sampled_token_id.first);
    }

    Array<Data> inputs;
    if (rsentry->parent_idx == -1) {
      inputs = request->inputs;
      if (const auto* token_input = inputs.back().as<TokenDataNode>()) {
        // Merge the TokenData so that a single time TokenEmbed is needed.
        std::vector<int> token_ids{token_input->token_ids->data,
                                   token_input->token_ids->data + token_input->token_ids.size()};
        token_ids.insert(token_ids.end(), committed_token_ids.begin(), committed_token_ids.end());
        inputs.Set(inputs.size() - 1, TokenData(token_ids));
      } else if (!committed_token_ids.empty()) {
        inputs.push_back(TokenData(committed_token_ids));
      }
    } else if (!committed_token_ids.empty()) {
      inputs.push_back(TokenData(committed_token_ids));
    }
    mstate->inputs = std::move(inputs);
    mstate->prefilled_inputs.clear();
    mstate->cached_committed_tokens = 0;
  }
  if (estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
    estate->prefix_cache->RecycleSequence(
        rsentry->mstates[0]->internal_id, TypedPackedFunc<void()>([estate, models, rsentry]() {
          RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);
        }),
        /*lazy=*/false);
  } else {
    RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);
  }

  if (preempt_rstate_idx == 0) {
    // Remove from running queue.
    estate->running_queue.erase(estate->running_queue.end() - 1);
  }
  if (!partially_alive && preempt_rstate_idx == static_cast<int>(rstate->entries.size()) - 1) {
    // Add to the front of waiting queue.
    estate->waiting_queue.insert(estate->waiting_queue.begin(), request);
  }
  return rsentry;
}

std::pair<NDArray, std::vector<SampleResult>> ApplyLogitProcessorAndSample(
    const LogitProcessor& logit_processor, const Sampler& sampler, const NDArray& logits,
    const Array<GenerationConfig>& generation_cfg, const Array<String>& request_ids,
    const Array<RequestModelState>& mstates, const std::vector<RandomGenerator*>& rngs,
    const std::vector<int>& sample_indices) {
  // - Update logits.
  logit_processor->InplaceUpdateLogits(logits, generation_cfg, mstates, request_ids);

  // - Compute probability distributions.
  NDArray probs_on_device =
      logit_processor->ComputeProbsFromLogits(logits, generation_cfg, request_ids);

  // - Sample tokens.
  NDArray renormalized_probs = sampler->BatchRenormalizeProbsByTopP(probs_on_device, sample_indices,
                                                                    request_ids, generation_cfg);
  std::vector<SampleResult> sample_results = sampler->BatchSampleTokensWithProbAfterTopP(
      renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
  return {std::move(probs_on_device), std::move(sample_results)};
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
