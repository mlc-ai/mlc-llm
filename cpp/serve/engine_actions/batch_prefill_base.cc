/*!
 *  Copyright (c) 2024 by Contributors
 * \file serve/engine_actions/batch_prefill_base.h
 */

#include "batch_prefill_base.h"

namespace mlc {
namespace llm {
namespace serve {

BatchPrefillBaseActionObj::BatchPrefillBaseActionObj(Array<Model> models,
                                                     EngineConfig engine_config,
                                                     Optional<EventTraceRecorder> trace_recorder)
    : models_(models), engine_config_(engine_config), trace_recorder_(trace_recorder) {}

/*!
 * \brief Find one or multiple request state entries to run prefill.
 * \param estate The engine state.
 * \return The request entries to prefill, together with their input lengths.
 */
std::vector<BatchPrefillBaseActionObj::PrefillInput>
BatchPrefillBaseActionObj::GetRequestStateEntriesToPrefill(EngineState estate) {
  if (estate->waiting_queue.empty()) {
    // No request to prefill.
    return {};
  }

  std::vector<PrefillInput> prefill_inputs;

  // - Try to prefill pending requests.
  int total_input_length = 0;
  int total_required_pages = 0;
  int num_available_pages = models_[0]->GetNumAvailablePages();
  int num_running_rsentries = GetRunningRequestStateEntries(estate).size();
  int current_total_seq_len = models_[0]->GetCurrentTotalSequenceLength();
  KVStateKind kv_state_kind = models_[0]->GetMetadata().kv_state_kind;

  int num_prefill_rsentries = 0;
  for (const Request& request : estate->waiting_queue) {
    RequestState rstate = estate->GetRequestState(request);
    bool prefill_stops = false;
    for (const RequestStateEntry& rsentry : rstate->entries) {
      // A request state entry can be prefilled only when:
      // - it has inputs, and
      // - it has no parent or its parent is alive and has no remaining input.
      if (rsentry->mstates[0]->inputs.empty() ||
          (rsentry->parent_idx != -1 &&
           (rstate->entries[rsentry->parent_idx]->status == RequestStateStatus::kPending ||
            !rstate->entries[rsentry->parent_idx]->mstates[0]->inputs.empty()))) {
        continue;
      }

      int input_length = rsentry->mstates[0]->GetInputLength();
      int num_require_pages = (input_length + engine_config_->kv_cache_page_size - 1) /
                              engine_config_->kv_cache_page_size;
      total_input_length += input_length;
      total_required_pages += num_require_pages;
      // - Attempt 1. Check if the entire request state entry can fit for prefill.
      bool can_prefill = false;
      for (int num_child_to_activate = rsentry->child_indices.size(); num_child_to_activate >= 0;
           --num_child_to_activate) {
        if (CanPrefill(estate, num_prefill_rsentries + 1 + num_child_to_activate,
                       total_input_length, total_required_pages, num_available_pages,
                       current_total_seq_len, num_running_rsentries, kv_state_kind)) {
          prefill_inputs.push_back({rsentry, input_length, num_child_to_activate});
          num_prefill_rsentries += 1 + num_child_to_activate;
          can_prefill = true;
          break;
        }
      }
      if (can_prefill) {
        continue;
      }
      total_input_length -= input_length;
      total_required_pages -= num_require_pages;

      // - Attempt 2. Check if the request state entry can partially fit by input chunking.
      ICHECK_LE(total_input_length, engine_config_->prefill_chunk_size);
      if (engine_config_->prefill_chunk_size - total_input_length >= input_length ||
          engine_config_->prefill_chunk_size == total_input_length) {
        // 1. If the input length can fit the remaining prefill chunk size,
        // it means the failure of attempt 1 is not because of the input
        // length being too long, and thus chunking does not help.
        // 2. If the total input length already reaches the prefill chunk size,
        // the current request state entry will not be able to be processed.
        // So we can safely return in either case.
        prefill_stops = true;
        break;
      }
      input_length = engine_config_->prefill_chunk_size - total_input_length;
      num_require_pages = (input_length + engine_config_->kv_cache_page_size - 1) /
                          engine_config_->kv_cache_page_size;
      total_input_length += input_length;
      total_required_pages += num_require_pages;
      if (CanPrefill(estate, num_prefill_rsentries + 1, total_input_length, total_required_pages,
                     num_available_pages, current_total_seq_len, num_running_rsentries,
                     kv_state_kind)) {
        prefill_inputs.push_back({rsentry, input_length, 0});
        num_prefill_rsentries += 1;
      }

      // - Prefill stops here.
      prefill_stops = true;
      break;
    }
    if (prefill_stops) {
      break;
    }
  }

  return prefill_inputs;
}

/*! \brief Check if the input requests can be prefilled under conditions. */
bool BatchPrefillBaseActionObj::CanPrefill(EngineState estate, int num_prefill_rsentries,
                                           int total_input_length, int num_required_pages,
                                           int num_available_pages, int current_total_seq_len,
                                           int num_running_rsentries, KVStateKind kv_state_kind) {
  ICHECK_LE(num_running_rsentries, engine_config_->max_num_sequence);

  // For RNN State, it can prefill as long as it can be instantiated.
  if (kv_state_kind == KVStateKind::kRNNState || kv_state_kind == KVStateKind::kNone) {
    return true;
  }

  // No exceeding of the maximum allowed requests that can
  // run simultaneously.
  int spec_factor = engine_config_->speculative_mode != SpeculativeMode::kDisable
                        ? (engine_config_->spec_draft_length + 1)
                        : 1;
  if ((num_running_rsentries + num_prefill_rsentries) * spec_factor >
      std::min(engine_config_->max_num_sequence, engine_config_->prefill_chunk_size)) {
    return false;
  }

  // NOTE: The conditions are heuristic and can be revised.
  // Cond 1: total input length <= prefill chunk size.
  // Cond 2: at least one decode can be performed after prefill.
  // Cond 3: number of total tokens after 8 times of decode does not
  // exceed the limit, where 8 is a watermark number can
  // be configured and adjusted in the future.
  int new_batch_size = num_running_rsentries + num_prefill_rsentries;
  return total_input_length <= engine_config_->prefill_chunk_size &&
         num_required_pages + new_batch_size <= num_available_pages &&
         current_total_seq_len + total_input_length + 8 * new_batch_size <=
             engine_config_->max_total_sequence_length;
}

/*!
 * \brief Chunk the input of the given RequestModelState for prefill
 * with regard to the provided maximum allowed prefill length.
 * Return the list of input for prefill and the total prefill length.
 * The `inputs` field of the given `mstate` will be mutated to exclude
 * the returned input.
 * \param mstate The RequestModelState whose input data is to be chunked.
 * \param max_prefill_length The maximum allowed prefill length for the mstate.
 * \return The list of input for prefill and the total prefill length.
 */
std::pair<Array<Data>, int> BatchPrefillBaseActionObj::ChunkPrefillInputData(
    const RequestModelState& mstate, int max_prefill_length) {
  if (mstate->inputs.empty()) {
  }
  ICHECK(!mstate->inputs.empty());
  std::vector<Data> inputs;
  int cum_input_length = 0;
  inputs.reserve(mstate->inputs.size());
  for (int i = 0; i < static_cast<int>(mstate->inputs.size()); ++i) {
    inputs.push_back(mstate->inputs[i]);
    int input_length = mstate->inputs[i]->GetLength();
    cum_input_length += input_length;
    // Case 0. the cumulative input length does not reach the maximum prefill length.
    if (cum_input_length < max_prefill_length) {
      continue;
    }

    // Case 1. the cumulative input length equals the maximum prefill length.
    if (cum_input_length == max_prefill_length) {
      if (i == static_cast<int>(mstate->inputs.size()) - 1) {
        // - If `i` is the last input, we just copy and reset `mstate->inputs`.
        mstate->inputs.clear();
      } else {
        // - Otherwise, set the new input array.
        mstate->inputs = Array<Data>{mstate->inputs.begin() + i + 1, mstate->inputs.end()};
      }
      return {inputs, cum_input_length};
    }

    // Case 2. cum_input_length > max_prefill_length
    // The input `i` itself needs chunking if it is TokenData,
    // or otherwise it cannot be chunked.
    Data input = mstate->inputs[i];
    inputs.pop_back();
    cum_input_length -= input_length;
    const auto* token_input = input.as<TokenDataNode>();
    if (token_input == nullptr) {
      // Cannot chunk the input.
      if (i != 0) {
        mstate->inputs = Array<Data>{mstate->inputs.begin() + i, mstate->inputs.end()};
      }
      return {inputs, cum_input_length};
    }

    // Split the token data into two parts.
    // Return the first part for prefill, and keep the second part.
    int chunked_input_length = max_prefill_length - cum_input_length;
    ICHECK_GT(input_length, chunked_input_length);
    TokenData chunked_input(IntTuple{token_input->token_ids.begin(),
                                     token_input->token_ids.begin() + chunked_input_length});
    TokenData remaining_input(IntTuple{token_input->token_ids.begin() + chunked_input_length,
                                       token_input->token_ids.end()});
    inputs.push_back(chunked_input);
    cum_input_length += chunked_input_length;
    std::vector<Data> remaining_inputs{mstate->inputs.begin() + i + 1, mstate->inputs.end()};
    remaining_inputs.insert(remaining_inputs.begin(), remaining_input);
    mstate->inputs = remaining_inputs;
    return {inputs, cum_input_length};
  }

  ICHECK(false) << "Cannot reach here";
}

void BatchPrefillBaseActionObj::UpdateRequestToAlive(
    const std::vector<BatchPrefillBaseActionObj::PrefillInput>& prefill_inputs,
    const EngineState& estate, Array<String>* request_ids,
    std::vector<RequestState>* rstates_of_entries,
    std::vector<RequestStateStatus>* status_before_prefill) {
  int num_rsentries = prefill_inputs.size();
  request_ids->reserve(num_rsentries);
  rstates_of_entries->reserve(num_rsentries);
  status_before_prefill->reserve(num_rsentries);
  for (const PrefillInput& prefill_input : prefill_inputs) {
    const RequestStateEntry& rsentry = prefill_input.rsentry;
    const Request& request = rsentry->request;
    RequestState request_rstate = estate->GetRequestState(request);
    request_ids->push_back(request->id);
    status_before_prefill->push_back(rsentry->status);
    rsentry->status = RequestStateStatus::kAlive;

    if (status_before_prefill->back() == RequestStateStatus::kPending) {
      // - Add the request to running queue if the request state
      // status was pending and all its request states were pending.
      bool alive_state_existed = false;
      for (const RequestStateEntry& rsentry_ : request_rstate->entries) {
        if (rsentry_->status == RequestStateStatus::kAlive && !rsentry_.same_as(rsentry)) {
          alive_state_existed = true;
        }
      }
      if (!alive_state_existed) {
        estate->running_queue.push_back(request);
      }
    }
    rstates_of_entries->push_back(std::move(request_rstate));
  }
}

std::vector<Request> BatchPrefillBaseActionObj::RemoveProcessedRequests(
    const std::vector<BatchPrefillBaseActionObj::PrefillInput>& prefill_inputs,
    const EngineState& estate, const std::vector<RequestState>& rstates_of_entries) {
  // - Remove the request from waiting queue if all its request states
  // are now alive and have no remaining chunked inputs.
  std::vector<Request> processed_requests;
  int num_rsentries = prefill_inputs.size();
  processed_requests.reserve(num_rsentries);
  std::unordered_set<const RequestNode*> dedup_map;
  for (int i = 0; i < num_rsentries; ++i) {
    const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
    if (dedup_map.find(rsentry->request.get()) != dedup_map.end()) {
      continue;
    }
    dedup_map.insert(rsentry->request.get());
    processed_requests.push_back(rsentry->request);

    bool pending_state_exists = false;
    for (const RequestStateEntry& rsentry_ : rstates_of_entries[i]->entries) {
      if (rsentry_->status == RequestStateStatus::kPending ||
          !rsentry_->mstates[0]->inputs.empty()) {
        pending_state_exists = true;
        break;
      }
    }
    if (!pending_state_exists) {
      auto it =
          std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), rsentry->request);
      ICHECK(it != estate->waiting_queue.end());
      estate->waiting_queue.erase(it);
    }
  }
  return processed_requests;
}

void BatchPrefillBaseActionObj::UpdateRequestStateEntriesWithSampleResults(
    const std::vector<RequestStateEntry>& rsentries_for_sample,
    const std::vector<bool>& rsentry_activated, const std::vector<SampleResult>& sample_results) {
  auto tnow = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
    // Update all model states of the request state entry.
    for (const RequestModelState& mstate : rsentries_for_sample[i]->mstates) {
      mstate->CommitToken(sample_results[i]);
      if (!rsentry_activated[i]) {
        // When the child rsentry is not activated,
        // add the sampled token as an input of the mstate for prefill.
        mstate->inputs.push_back(
            TokenData(std::vector<int64_t>{sample_results[i].sampled_token_id.first}));
      }
    }
    if (rsentries_for_sample[i]->mstates[0]->committed_tokens.size() == 1) {
      rsentries_for_sample[i]->tprefill_finish = tnow;
    }
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
