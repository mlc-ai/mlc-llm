/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/batch_prefill_base.h
 */

#include "batch_prefill_base.h"

#include <numeric>

#include "../../support/json_parser.h"

namespace mlc {
namespace llm {
namespace serve {

bool HasPrefillSpace(int num_required_pages, bool sliding_window_enabled, int new_batch_size,
                     int num_available_pages, int current_total_seq_len, int total_input_length,
                     int max_total_sequence_length) {
  return num_required_pages + (!sliding_window_enabled ? new_batch_size : 0) <=
             num_available_pages &&
         (sliding_window_enabled ||
          current_total_seq_len + total_input_length + 8 * new_batch_size <=
              max_total_sequence_length);
}

BatchPrefillBaseActionObj::BatchPrefillBaseActionObj(Array<Model> models,
                                                     EngineConfig engine_config,
                                                     std::vector<picojson::object> model_configs,
                                                     Optional<EventTraceRecorder> trace_recorder)
    : models_(std::move(models)),
      engine_config_(std::move(engine_config)),
      trace_recorder_(std::move(trace_recorder)) {
  TVM_FFI_ICHECK_EQ(models_.size(), model_configs.size());
  sliding_window_sizes_.reserve(models_.size());
  for (const picojson::object& model_config : model_configs) {
    // "-1" means the sliding window is disabled.
    sliding_window_sizes_.push_back(
        json::LookupOrDefault<int64_t>(model_config, "sliding_window_size", -1));
  }
  kv_state_kind_ = models_[0]->GetMetadata().kv_state_kind;
}

/*!
 * \brief Find one or multiple request state entries to run prefill.
 * \param estate The engine state.
 * \return The request entries to prefill, together with their input lengths.
 */
std::vector<BatchPrefillBaseActionObj::PrefillInput>
BatchPrefillBaseActionObj::GetRequestStateEntriesToPrefill(EngineState estate) {
  // Preempt request state entries when decode cannot apply.
  const std::vector<RequestStateEntry>* running_rsentries;
  {
    NVTXScopedRange nvtx_scope("BatchDecode getting requests");
    running_rsentries = &estate->GetRunningRequestStateEntries();
    if (!(running_rsentries->size() <= models_[0]->GetNumAvailablePages())) {
      // Even the decode cannot be performed.
      // As a result, directly return without doing prefill.
      return {};
    }
  }

  if (estate->waiting_queue.empty()) {
    // No request to prefill.
    return {};
  }

  std::vector<std::vector<PrefillInput>> prefill_inputs_for_all_models;
  prefill_inputs_for_all_models.reserve(models_.size());

  int num_decode_inputs = static_cast<int>(running_rsentries->size());

  // We first collect the inputs that can be prefilled for each model.
  // Then we make a reduction to return the maximum common inputs.
  for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
    std::vector<PrefillInput> prefill_inputs;
    // - Try to prefill pending requests.
    int total_input_length = 0;
    for (const RequestStateEntry& rsentry : *running_rsentries) {
      total_input_length += rsentry->mstates[i]->num_tokens_for_next_decode;
    }
    int total_required_pages = num_decode_inputs;
    int num_available_pages;
    int num_running_rsentries = num_decode_inputs;
    int current_total_seq_len;
    {
      NVTXScopedRange nvtx_scope("KV cache GetNumAvailablePages");
      num_available_pages = models_[i]->GetNumAvailablePages();
    }
    {
      NVTXScopedRange nvtx_scope("KV cache GetCurrentTotalSequenceLength");
      current_total_seq_len = models_[i]->GetCurrentTotalSequenceLength();
    }

    int num_prefill_rsentries = 0;
    for (const Request& request : estate->waiting_queue) {
      NVTXScopedRange nvtx_scope("Process request " + request->id);
      if (request->generation_cfg->debug_config.disagg_config.kind != DisaggRequestKind::kNone) {
        continue;
      }
      RequestState rstate = estate->GetRequestState(request);
      bool prefill_stops = false;
      for (const RequestStateEntry& rsentry : rstate->entries) {
        // A request state entry can be prefilled only when:
        // - it has inputs, and
        // - it has no parent or its parent is alive and has no remaining input.
        if (rsentry->mstates[i]->inputs.empty() ||
            (rsentry->parent_idx != -1 &&
             (rstate->entries[rsentry->parent_idx]->status == RequestStateStatus::kPending ||
              !rstate->entries[rsentry->parent_idx]->mstates[i]->inputs.empty()))) {
          continue;
        }

        int input_length = rsentry->mstates[i]->GetInputLength();
        int num_require_pages = (input_length + engine_config_->kv_cache_page_size - 1) /
                                engine_config_->kv_cache_page_size;
        bool sliding_window_enabled = sliding_window_sizes_[i] != -1;
        int num_required_pages_under_sliding_window = std::numeric_limits<int>::max();
        if (sliding_window_enabled) {
          // Sliding window for model i is enabled.
          int max_single_request_page_requirement =
              1 + (sliding_window_sizes_[i] + engine_config_->kv_cache_page_size - 1) /
                      engine_config_->kv_cache_page_size;
          int num_total_prefilled_tokens = rsentry->mstates[i]->num_prefilled_tokens;
          int parent_ptr = rsentry->parent_idx;
          while (parent_ptr != -1) {
            num_total_prefilled_tokens +=
                rstate->entries[parent_ptr]->mstates[i]->num_prefilled_tokens;
            parent_ptr = rstate->entries[parent_ptr]->parent_idx;
          }

          int num_pages_in_use = (std::min(num_total_prefilled_tokens, sliding_window_sizes_[i]) +
                                  engine_config_->kv_cache_page_size - 1) /
                                 engine_config_->kv_cache_page_size;
          num_required_pages_under_sliding_window =
              max_single_request_page_requirement - num_pages_in_use;
          num_require_pages = std::min(num_require_pages, num_required_pages_under_sliding_window);
          TVM_FFI_ICHECK_GE(num_require_pages, 0);
        }

        total_input_length += input_length;
        total_required_pages += num_require_pages;
        // - Attempt 1. Check if the entire request state entry can fit for prefill.
        bool can_prefill = false;
        {
          NVTXScopedRange nvtx_scope("Attempt 1");
          for (int num_child_to_activate = rsentry->child_indices.size();
               num_child_to_activate >= 0; --num_child_to_activate) {
            while (!HasPrefillSpace(total_required_pages, sliding_window_enabled,
                                    (num_running_rsentries + num_prefill_rsentries),
                                    num_available_pages, current_total_seq_len, total_input_length,
                                    engine_config_->max_total_sequence_length)) {
              if (!estate->prefix_cache->TryFreeMemory()) break;
              // Update number of available pages after memory free.
              num_available_pages = models_[i]->GetNumAvailablePages();
              current_total_seq_len = models_[i]->GetCurrentTotalSequenceLength();
            }
            if (CanPrefill(estate, num_prefill_rsentries + 1 + num_child_to_activate,
                           total_input_length, total_required_pages, num_available_pages,
                           current_total_seq_len, num_running_rsentries, kv_state_kind_,
                           sliding_window_enabled)) {
              prefill_inputs.push_back(
                  {rsentry, input_length, num_child_to_activate, /*is_decode=*/false});
              num_prefill_rsentries += 1 + num_child_to_activate;
              can_prefill = true;
              break;
            }
          }
        }
        if (can_prefill) {
          continue;
        }
        total_input_length -= input_length;
        total_required_pages -= num_require_pages;

        // - Attempt 2. Check if the request state entry can partially fit by input chunking.
        TVM_FFI_ICHECK_LE(total_input_length, engine_config_->prefill_chunk_size);
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
        if (sliding_window_enabled) {
          // Sliding window for model i is enabled.
          num_require_pages = std::min(num_require_pages, num_required_pages_under_sliding_window);
          TVM_FFI_ICHECK_GE(num_require_pages, 0);
        }

        {
          NVTXScopedRange nvtx_scope("Attempt 2");
          total_input_length += input_length;
          total_required_pages += num_require_pages;
          if (CanPrefill(estate, num_prefill_rsentries + 1, total_input_length,
                         total_required_pages, num_available_pages, current_total_seq_len,
                         num_running_rsentries, kv_state_kind_, sliding_window_enabled)) {
            prefill_inputs.push_back({rsentry, input_length, 0, /*is_decode=*/false});
          }
        }

        // - Prefill stops here.
        prefill_stops = true;
        break;
      }
      if (prefill_stops) {
        break;
      }
    }
    prefill_inputs_for_all_models.push_back(prefill_inputs);
  }

  // Reduce over the prefill inputs of all models.
  TVM_FFI_ICHECK(!prefill_inputs_for_all_models.empty());
  int num_prefill_inputs = prefill_inputs_for_all_models[0].size();
  for (int i = 1; i < static_cast<int>(prefill_inputs_for_all_models.size()); ++i) {
    num_prefill_inputs =
        std::min(num_prefill_inputs, static_cast<int>(prefill_inputs_for_all_models[i].size()));
  }

  if (num_prefill_inputs == 0) {
    return {};
  }

  // Add the decode requests to the prefill inputs if prefill mode is hybrid.
  std::vector<PrefillInput> prefill_inputs(prefill_inputs_for_all_models[0].begin(),
                                           prefill_inputs_for_all_models[0].end());
  if (engine_config_->prefill_mode == PrefillMode::kHybrid) {
    prefill_inputs.reserve(num_decode_inputs + num_prefill_inputs);
    for (const RequestStateEntry& rsentry : *running_rsentries) {
      prefill_inputs.push_back(
          {rsentry, rsentry->mstates[0]->num_tokens_for_next_decode, 0, /*is_decode=*/true});
    }
  }
  {
    NVTXScopedRange nvtx_scope("reduction");
    for (int i = 1; i < static_cast<int>(prefill_inputs_for_all_models.size()); ++i) {
      // Prefill input lengths except the last one are supposed to be the same for all models.
      for (int j = 0; j < num_prefill_inputs - 1; ++j) {
        TVM_FFI_ICHECK(
            prefill_inputs_for_all_models[i][j].rsentry.same_as(prefill_inputs[j].rsentry));
        TVM_FFI_ICHECK_EQ(prefill_inputs_for_all_models[i][j].max_prefill_length,
                          prefill_inputs[j].max_prefill_length);
        prefill_inputs[j].num_child_to_activate =
            std::min(prefill_inputs[j].num_child_to_activate,
                     prefill_inputs_for_all_models[i][j].num_child_to_activate);
      }
      // The input length of the last input is the minimum among all models.
      TVM_FFI_ICHECK(prefill_inputs_for_all_models[i][num_prefill_inputs - 1].rsentry.same_as(
          prefill_inputs[num_prefill_inputs - 1].rsentry));
      prefill_inputs[num_prefill_inputs - 1].max_prefill_length =
          std::min(prefill_inputs[num_prefill_inputs - 1].max_prefill_length,
                   prefill_inputs_for_all_models[i][num_prefill_inputs - 1].max_prefill_length);
      prefill_inputs[num_prefill_inputs - 1].num_child_to_activate =
          std::min(prefill_inputs[num_prefill_inputs - 1].num_child_to_activate,
                   prefill_inputs_for_all_models[i][num_prefill_inputs - 1].num_child_to_activate);
    }
  }

  return prefill_inputs;
}

bool BatchPrefillBaseActionObj::CanPrefill(EngineState estate, int num_prefill_rsentries,
                                           int total_input_length, int num_required_pages,
                                           int num_available_pages, int current_total_seq_len,
                                           int num_running_rsentries, KVStateKind kv_state_kind,
                                           bool sliding_window_enabled) {
  TVM_FFI_ICHECK_LE(num_running_rsentries, engine_config_->max_num_sequence);

  // For RNN State, it can prefill as long as it can be instantiated.
  if (kv_state_kind == KVStateKind::kRNNState || kv_state_kind == KVStateKind::kNone) {
    return true;
  }

  // No exceeding of the maximum allowed requests that can
  // run simultaneously.
  int spec_factor = engine_config_->speculative_mode != SpeculativeMode::kDisable
                        ? (estate->spec_draft_length + 1)
                        : 1;
  if ((num_running_rsentries + num_prefill_rsentries) * spec_factor >
      std::min(static_cast<int64_t>(engine_config_->max_num_sequence),
               engine_config_->prefill_chunk_size)) {
    return false;
  }

  // NOTE: The conditions are heuristic and can be revised.
  // Cond 1: total input length <= prefill chunk size.
  // Cond 2: at least one decode can be performed after prefill.
  // Cond 3: number of total tokens after 8 times of decode does not
  // exceed the limit, where 8 is a watermark number can
  // be configured and adjusted in the future.
  return total_input_length <= engine_config_->prefill_chunk_size &&
         HasPrefillSpace(num_required_pages, sliding_window_enabled,
                         (num_running_rsentries + num_prefill_rsentries), num_available_pages,
                         current_total_seq_len, total_input_length,
                         engine_config_->max_total_sequence_length);
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
    // If the request is a hybrid decode request
    TVM_FFI_ICHECK(mstate->num_tokens_for_next_decode > 0);
    int num_tokens = mstate->num_tokens_for_next_decode;
    mstate->num_tokens_for_next_decode = 0;
    std::vector<int32_t> decode_tokens;
    decode_tokens.reserve(num_tokens);
    for (auto begin = mstate->committed_tokens.end() - num_tokens;
         begin != mstate->committed_tokens.end(); ++begin) {
      decode_tokens.push_back(begin->GetTokenId());
    }
    return {{TokenData(decode_tokens)}, num_tokens};
  }
  TVM_FFI_ICHECK(!mstate->inputs.empty());
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
    TVM_FFI_ICHECK_GT(input_length, chunked_input_length);
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

  TVM_FFI_ICHECK(false) << "Cannot reach here";
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
    if (dedup_map.find(rsentry->request.operator->()) != dedup_map.end()) {
      continue;
    }
    dedup_map.insert(rsentry->request.operator->());
    processed_requests.push_back(rsentry->request);

    bool pending_state_exists = false;
    for (const RequestStateEntry& rsentry_ : rstates_of_entries[i]->entries) {
      if (rsentry_->status == RequestStateStatus::kPending ||
          !rsentry_->mstates[0]->inputs.empty()) {
        pending_state_exists = true;
        break;
      }
    }
    if (!pending_state_exists &&
        std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), rsentry->request) !=
            estate->waiting_queue.end()) {
      auto it =
          std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), rsentry->request);
      if (it != estate->waiting_queue.end()) {
        estate->waiting_queue.erase(it);
      }
    }
  }
  return processed_requests;
}

void BatchPrefillBaseActionObj::UpdateRequestStateEntriesWithSampleResults(
    const std::vector<RequestStateEntry>& rsentries_for_sample,
    const std::vector<bool>& rsentry_activated, const std::vector<SampleResult>& sample_results) {
  auto tnow = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
    // If the request is a hybrid decode request
    if (rsentries_for_sample[i]->status == RequestStateStatus::kAlive &&
        rsentries_for_sample[i]->child_indices.empty() &&
        rsentries_for_sample[i]->mstates[0]->inputs.empty()) {
      for (const RequestModelState& mstate : rsentries_for_sample[i]->mstates) {
        CHECK(!mstate->require_retokenization_in_next_decode);
        mstate->CommitToken(sample_results[i]);
        // live update the output metrics
        rsentries_for_sample[i]->rstate->metrics.completion_tokens += 1;
        rsentries_for_sample[i]->rstate->metrics.prefill_end_time_point = tnow;
      }
      continue;
    }

    // Update all model states of the request state entry.
    for (const RequestModelState& mstate : rsentries_for_sample[i]->mstates) {
      mstate->CommitToken(sample_results[i]);
      if (!rsentry_activated[i]) {
        // When the child rsentry is not activated,
        // add the sampled token as an input of the mstate for prefill.
        mstate->inputs.push_back(TokenData(std::vector<int64_t>{sample_results[i].GetTokenId()}));
      }
    }
    // prefill has finished
    if (rsentries_for_sample[i]->mstates[0]->committed_tokens.size() == 1) {
      TVM_FFI_ICHECK(rsentries_for_sample[i]->rstate != nullptr);
      rsentries_for_sample[i]->rstate->metrics.prefill_end_time_point = tnow;
    }
  }
}

std::vector<int32_t> BatchPrefillBaseActionObj::GetConcatPrefillInputData(
    const RequestModelState& mstate) {
  std::vector<int32_t> tokens;
  for (Data data : mstate->inputs) {
    if (const TokenDataNode* token_data = data.as<TokenDataNode>()) {
      tokens.reserve(tokens.size() + token_data->GetLength());
      tokens.insert(tokens.end(), token_data->token_ids.begin(), token_data->token_ids.end());
    } else {
      return {};
    }
  }
  return tokens;
}

void BatchPrefillBaseActionObj::PopPrefillInputData(const RequestModelState& mstate,
                                                    size_t num_tokens) {
  while (mstate->inputs[0]->GetLength() <= num_tokens) {
    num_tokens -= mstate->inputs[0]->GetLength();
    mstate->inputs.erase(mstate->inputs.begin());
  }
  if (num_tokens) {
    const TokenDataNode* token_data = mstate->inputs[0].as<TokenDataNode>();
    std::vector<int32_t> tokens;
    tokens.reserve(token_data->GetLength() - num_tokens);
    tokens.insert(tokens.begin(), token_data->token_ids.begin() + num_tokens,
                  token_data->token_ids.end());
    mstate->inputs.erase(mstate->inputs.begin());
    mstate->inputs.insert(mstate->inputs.begin(), TokenData(tokens));
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
