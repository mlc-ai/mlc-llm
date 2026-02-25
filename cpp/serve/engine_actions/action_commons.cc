/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/action_commons.cc
 */

#include "action_commons.h"

#include <tvm/runtime/nvtx.h>

namespace mlc {
namespace llm {
namespace serve {

Array<EngineAction> CreateEngineActions(
    Array<Model> models, EngineConfig engine_config, std::vector<picojson::object> model_configs,
    std::vector<ModelWorkspace> model_workspaces, LogitProcessor logit_processor, Sampler sampler,
    DraftTokenWorkspaceManager draft_token_workspace_manager, Tokenizer tokenizer,
    Optional<EventTraceRecorder> trace_recorder, FRequestStreamCallback request_stream_callback,
    Device device) {
  Array<EngineAction> actions;
  ModelMetadata model_metadata = models[0]->GetMetadata();
  if (engine_config->speculative_mode != SpeculativeMode::kDisable) {
    // Speculative decoding is only possible for more than one model.
    TVM_FFI_ICHECK_GT(models.size(), 1U);
    if (engine_config->speculative_mode == SpeculativeMode::kEagle) {
      CHECK_GT(engine_config->spec_draft_length, 0)
          << "The automatic spec decoding does not support Eagle mode as of now.";
      actions = {EngineAction::EagleNewRequestPrefill(models,                         //
                                                      logit_processor,                //
                                                      sampler,                        //
                                                      model_workspaces,               //
                                                      draft_token_workspace_manager,  //
                                                      engine_config,                  //
                                                      model_configs,                  //
                                                      trace_recorder),
                 EngineAction::EagleBatchDraft(models, logit_processor, sampler, model_workspaces,
                                               draft_token_workspace_manager, engine_config,
                                               trace_recorder),
                 EngineAction::EagleBatchVerify(models, logit_processor, sampler, model_workspaces,
                                                draft_token_workspace_manager, engine_config,
                                                trace_recorder)};
    } else if (engine_config->speculative_mode == SpeculativeMode::kMedusa) {
      CHECK_GT(engine_config->spec_draft_length, 0)
          << "The automatic spec decoding does not support Eagle mode as of now.";
      actions = {EngineAction::EagleNewRequestPrefill(models,                         //
                                                      logit_processor,                //
                                                      sampler,                        //
                                                      model_workspaces,               //
                                                      draft_token_workspace_manager,  //
                                                      engine_config,                  //
                                                      model_configs,                  //
                                                      trace_recorder),
                 EngineAction::EagleBatchVerify(models, logit_processor, sampler, model_workspaces,
                                                draft_token_workspace_manager, engine_config,
                                                trace_recorder)};
    } else if (engine_config->spec_draft_length > 0) {
      // The "small draft" mode speculative decoding.
      // If "engine_config->spec_draft_length" > 0, it means the draft length is
      // configured to be a fixed value.
      actions = {
          EngineAction::NewRequestPrefill(models,            //
                                          logit_processor,   //
                                          sampler,           //
                                          model_workspaces,  //
                                          engine_config,     //
                                          model_configs,     //
                                          trace_recorder),
          EngineAction::BatchDraft(models, logit_processor, sampler, model_workspaces,
                                   draft_token_workspace_manager, engine_config, trace_recorder),
          EngineAction::BatchVerify(models, logit_processor, sampler, model_workspaces,
                                    draft_token_workspace_manager, engine_config, trace_recorder)};
    } else {
      // The "small draft" mode speculative decoding.
      // "engine_config->spec_draft_length" being 0 means we want to enable
      // automatic speculative decoding, which decides the spec decoding draft length
      // automatically.
      actions = {EngineAction::NewRequestPrefill(models,            //
                                                 logit_processor,   //
                                                 sampler,           //
                                                 model_workspaces,  //
                                                 engine_config,     //
                                                 model_configs,     //
                                                 trace_recorder),
                 EngineAction::AutoSpecDecode(
                     /*spec_decode_actions=*/{EngineAction::BatchDraft(
                                                  models, logit_processor, sampler,
                                                  model_workspaces, draft_token_workspace_manager,
                                                  engine_config, trace_recorder),
                                              EngineAction::BatchVerify(
                                                  models, logit_processor, sampler,
                                                  model_workspaces, draft_token_workspace_manager,
                                                  engine_config, trace_recorder)},
                     /*batch_decode_actions=*/
                     {EngineAction::BatchDecode(models, tokenizer, logit_processor, sampler,
                                                engine_config, trace_recorder)},
                     engine_config)};
    }
  } else if (model_metadata.disaggregation) {
    actions = {EngineAction::NewRequestPrefill(models,            //
                                               logit_processor,   //
                                               sampler,           //
                                               model_workspaces,  //
                                               engine_config,     //
                                               model_configs,     //
                                               trace_recorder),
               EngineAction::BatchDecode(models, tokenizer, logit_processor, sampler, engine_config,
                                         trace_recorder)};
  } else {
    // The normal mode.
    actions = {EngineAction::NewRequestPrefill(models,            //
                                               logit_processor,   //
                                               sampler,           //
                                               model_workspaces,  //
                                               engine_config,     //
                                               model_configs,     //
                                               trace_recorder),
               EngineAction::BatchJumpForward(models, tokenizer, trace_recorder),
               EngineAction::BatchDecode(models, tokenizer, logit_processor, sampler, engine_config,
                                         trace_recorder)};
  }

  if (model_metadata.disaggregation) {
    // Insert the disaggregation actions.
    Array<EngineAction> disaggregation_actions = {
        EngineAction::DisaggPrepareReceive(models, engine_config, model_configs, trace_recorder,
                                           request_stream_callback),
        EngineAction::DisaggRemoteSend(models, model_workspaces, engine_config, model_configs,
                                       trace_recorder, request_stream_callback, device)};
    actions.insert(actions.begin(), disaggregation_actions.begin(), disaggregation_actions.end());
  }
  return actions;
}

void RemoveRequestFromModel(EngineState estate, int64_t req_internal_id,
                            const Array<Model>& models) {
  // Remove the request from all models (usually the KV cache).
  for (Model model : models) {
    model->RemoveSequence(req_internal_id);
  }
}

/*!
 * \brief Remove the given request state entry.
 * \param estate The engine state to update after removal.
 * \param models The models to remove the given request from.
 * \param rsentry The request state entry to remove.
 */
void RemoveRequestStateEntry(EngineState estate, const Array<Model>& models,
                             RequestStateEntry rsentry,
                             Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager) {
  if (draft_token_workspace_manager.defined()) {
    std::vector<int> draft_token_slots;
    for (const RequestModelState& mstate : rsentry->mstates) {
      mstate->RemoveAllDraftTokens(&draft_token_slots);
      draft_token_workspace_manager.value()->FreeSlots(draft_token_slots);
    }
  }
  if (estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
    // If the sequence is stored in prefix cache, call prefix cache to remove.
    if (!(rsentry->request->generation_cfg->debug_config.pinned_system_prompt)) {
      // If the request is not pinned, recycle the request.
      estate->prefix_cache->RecycleSequence(rsentry->mstates[0]->internal_id, /*lazy=*/true);
    }
    // If the request is pinned, do nothing over the prefix cache and KVCache.
  } else {
    // If the sequence is not stored in prefix cache, remove it directly.
    RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);
    estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
  }
}

void ProcessFinishedRequestStateEntries(
    const std::vector<RequestStateEntry>& finished_rsentries, EngineState estate,
    const Array<Model>& models, int max_single_sequence_length,
    Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager,
    Array<RequestStreamOutput>* callback_delta_outputs) {
  NVTXScopedRange nvtx_scope("Process finished requests");
  // - Remove the finished request state entries.
  for (const RequestStateEntry& rsentry : finished_rsentries) {
    // The finished entry must be a leaf.
    TVM_FFI_ICHECK(rsentry->child_indices.empty());
    // Mark the status of this entry as finished.
    rsentry->status = RequestStateStatus::kFinished;
    // Remove the request state entry from all the models.
    RemoveRequestStateEntry(estate, models, rsentry, draft_token_workspace_manager);

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

      RemoveRequestStateEntry(estate, models, rstate->entries[parent_idx],
                              draft_token_workspace_manager);
      // Climb up to the parent.
      parent_idx = rstate->entries[parent_idx]->parent_idx;
    }

    if (parent_idx == -1) {
      // Remove from running queue and engine state.
      auto it =
          std::find(estate->running_queue.begin(), estate->running_queue.end(), rsentry->request);
      TVM_FFI_ICHECK(it != estate->running_queue.end());
      estate->running_queue.erase(it);
      estate->request_states.erase(rsentry->request->id);

      // Update engine metrics.
      const RequestStateEntry& root_rsentry = rstate->entries[0];
      auto trequest_finish = std::chrono::high_resolution_clock::now();

      rstate->metrics.finish_time_point = trequest_finish;
      estate->metrics.RequestFinishUpdate(rstate->metrics);

      // always stream back usage in backend
      callback_delta_outputs->push_back(RequestStreamOutput::Usage(
          root_rsentry->request->id, rstate->metrics.AsUsageJSONStr(true)));
    }
    estate->running_rsentries_changed = true;
  }
}

void ActionStepPostProcess(Array<Request> requests, EngineState estate, const Array<Model>& models,
                           const Tokenizer& tokenizer,
                           FRequestStreamCallback request_stream_callback,
                           int64_t max_single_sequence_length,
                           Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager,
                           Optional<EventTraceRecorder> trace_recorder) {
  NVTXScopedRange nvtx_scope("EngineAction postproc");
  int num_requests = requests.size();
  estate->postproc_workspace.finished_rsentries.clear();
  estate->postproc_workspace.callback_delta_outputs.clear();
  estate->postproc_workspace.finished_rsentries.reserve(num_requests);
  estate->postproc_workspace.callback_delta_outputs.reserve(num_requests * 2);

  // - Collect new generated tokens and finish reasons for requests.
  for (int r = 0; r < num_requests; ++r) {
    Request request = requests[r];
    int n = request->generation_cfg->n;
    RequestState rstate = estate->GetRequestState(requests[r]);

    bool invoke_callback = false;
    RequestStreamOutput stream_output = rstate->postproc_states.GetStreamOutput();
    for (int i = 0; i < n; ++i) {
      const RequestStateEntry& rsentry = n == 1 ? rstate->entries[0] : rstate->entries[i + 1];
      rsentry->GetDeltaRequestReturn(tokenizer, max_single_sequence_length, &stream_output, i);
      if (stream_output->group_finish_reason[i].has_value()) {
        invoke_callback = true;
        estate->postproc_workspace.finished_rsentries.push_back(rsentry);
      }
      if (!stream_output->group_delta_token_ids[i].empty() ||
          !stream_output->group_extra_prefix_string[i].empty()) {
        invoke_callback = true;
      }
    }

    if (invoke_callback) {
      stream_output->unpacked = false;
      estate->postproc_workspace.callback_delta_outputs.push_back(std::move(stream_output));
    }

    // Update prefix cache and metrics.
    for (const RequestStateEntry& rsentry : rstate->entries) {
      std::vector<int32_t>& token_ids = rsentry->token_ids_for_prefix_cache_update;
      token_ids.clear();
      if (!rsentry->mstates[0]->prefilled_inputs.empty()) {
        // Notify the prefix cache of the newly prefilled data.
        for (const Data& data : rsentry->mstates[0]->prefilled_inputs) {
          const TokenDataNode* token_data = data.as<TokenDataNode>();
          if (token_data == nullptr) continue;
          token_ids.insert(token_ids.end(), token_data->token_ids->data,
                           token_data->token_ids->data + token_data->token_ids.size());
          // note that we are counting prefill tokens across all branches
          rstate->metrics.prefill_tokens += data->GetLength();
        }
        rsentry->mstates[0]->prefilled_inputs.clear();
      }
      int64_t num_committed_tokens = rsentry->mstates[0]->committed_tokens.size();
      if (rsentry->mstates[0]->cached_committed_tokens < num_committed_tokens - 1) {
        // Notify the prefix cache of the newly decoded data, except the last token as it is not
        // in KVCache yet.
        for (int64_t& i = rsentry->mstates[0]->cached_committed_tokens;
             i < num_committed_tokens - 1; ++i) {
          token_ids.push_back(rsentry->mstates[0]->committed_tokens[i].sampled_token_id.first);
        }
      }
      if (!token_ids.empty()) {
        estate->prefix_cache->ExtendSequence(rsentry->mstates[0]->internal_id, token_ids);
      }
    }

    // - For all disaggregation requests with "remote_send",
    // if it does not appear in the waiting queue, it means the prefill has been finished.
    // In this case, we mark the request as finished.
    if (request->generation_cfg->debug_config.disagg_config.kind ==
        DisaggRequestKind::kRemoteSend) {
      auto it = std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), request);
      if (it == estate->waiting_queue.end()) {
        CHECK_EQ(rstate->entries.size(), 1);
        estate->postproc_workspace.finished_rsentries.push_back(rstate->entries[0]);
      }
    }
  }

  ProcessFinishedRequestStateEntries(estate->postproc_workspace.finished_rsentries, estate, models,
                                     max_single_sequence_length, draft_token_workspace_manager,
                                     &estate->postproc_workspace.callback_delta_outputs);

  if (!estate->postproc_workspace.callback_delta_outputs.empty()) {
    NVTXScopedRange nvtx_scope("Call request stream callback");
    // - Invoke the stream callback function once for all collected requests.
    request_stream_callback(estate->postproc_workspace.callback_delta_outputs);
  }
}  // namespace serve

RequestStateEntry PreemptLastRunningRequestStateEntry(
    EngineState estate, const Array<Model>& models,
    Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager,
    Optional<EventTraceRecorder> trace_recorder) {
  TVM_FFI_ICHECK(!estate->running_queue.empty());
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
  TVM_FFI_ICHECK_NE(preempt_rstate_idx, -1);
  RequestStateEntry rsentry = rstate->entries[preempt_rstate_idx];
  if (estate->disaggregation) {
    AbortRequestImpl(estate, models, request->id, "preempt");
    return rsentry;
  }
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

    // If the commited tokens of the current model lags behind the
    // committed tokens of the main model (models[0]), we commit those
    // new tokens to this model.
    for (size_t i = mstate->committed_tokens.size();
         i < rsentry->mstates[0]->committed_tokens.size(); ++i) {
      mstate->CommitToken(rsentry->mstates[0]->committed_tokens[i]);
    }

    std::vector<int32_t> committed_token_ids;
    committed_token_ids.reserve(mstate->committed_tokens.size());
    for (const SampleResult& committed_token : mstate->committed_tokens) {
      committed_token_ids.push_back(committed_token.GetTokenId());
    }
    mstate->num_prefilled_tokens = 0;

    Array<Data> inputs;
    if (rsentry->parent_idx == -1) {
      inputs = request->inputs;
      if (const auto* token_input = inputs.back().as<TokenDataNode>()) {
        // Merge the TokenData so that a single time TokenEmbed is needed.
        std::vector<int> token_ids{token_input->token_ids->data,
                                   token_input->token_ids->data + token_input->token_ids.size()};
        token_ids.insert(token_ids.end(), committed_token_ids.begin(), committed_token_ids.end());
        inputs.Set(static_cast<int64_t>(inputs.size()) - 1, TokenData(token_ids));
      } else if (!committed_token_ids.empty()) {
        inputs.push_back(TokenData(committed_token_ids));
      }
    } else if (!committed_token_ids.empty()) {
      inputs.push_back(TokenData(committed_token_ids));
    }
    mstate->inputs = std::move(inputs);
    mstate->prefilled_inputs.clear();
    mstate->cached_committed_tokens = 0;
    mstate->num_tokens_for_next_decode = 0;
  }
  if (estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
    estate->prefix_cache->RecycleSequence(rsentry->mstates[0]->internal_id, /*lazy=*/false);
  } else {
    RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);
  }
  // Since the sequence has been removed from model, assign a new sequence ID.
  int64_t new_seq_id = estate->id_manager.GetNewId();
  for (RequestModelState mstate : rsentry->mstates) {
    mstate->internal_id = new_seq_id;
  }

  if (preempt_rstate_idx == 0) {
    // Remove from running queue.
    estate->running_queue.erase(estate->running_queue.end() - 1);
  }
  if (!partially_alive && preempt_rstate_idx == static_cast<int>(rstate->entries.size()) - 1) {
    // Add to the front of waiting queue.
    estate->waiting_queue.insert(estate->waiting_queue.begin(), request);
  }
  estate->running_rsentries_changed = true;
  return rsentry;
}

std::pair<Tensor, std::vector<SampleResult>> ApplyLogitProcessorAndSample(
    const LogitProcessor& logit_processor, const Sampler& sampler, const Tensor& logits,
    const Array<GenerationConfig>& generation_cfg, const Array<String>& request_ids,
    const Array<RequestModelState>& mstates, const std::vector<RandomGenerator*>& rngs,
    const std::vector<int>& sample_indices, const Array<GenerationConfig>& child_generation_cfg,
    const Array<String>& child_request_ids, const std::vector<int>& child_sample_indices) {
  // - Update logits.
  logit_processor->InplaceUpdateLogits(logits, generation_cfg, mstates, request_ids);

  // - Compute probability distributions.
  Tensor probs_on_device =
      logit_processor->ComputeProbsFromLogits(logits, generation_cfg, request_ids);

  // - Sample tokens.
  Tensor renormalized_probs = sampler->BatchRenormalizeProbsByTopP(probs_on_device, sample_indices,
                                                                   request_ids, generation_cfg);
  std::vector<SampleResult> sample_results = sampler->BatchSampleTokensWithProbAfterTopP(
      renormalized_probs, child_sample_indices, child_request_ids, child_generation_cfg, rngs);
  return {std::move(probs_on_device), std::move(sample_results)};
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
