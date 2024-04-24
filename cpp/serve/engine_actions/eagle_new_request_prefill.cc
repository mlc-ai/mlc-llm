/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/eagle_new_request_prefill.cc
 */

#include <tvm/runtime/nvtx.h>

#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that prefills requests in the `waiting_queue` of
 * the engine state.
 */
class EagleNewRequestPrefillActionObj : public EngineActionObj {
 public:
  explicit EagleNewRequestPrefillActionObj(Array<Model> models, LogitProcessor logit_processor,
                                           Sampler sampler,
                                           std::vector<ModelWorkspace> model_workspaces,
                                           DraftTokenWorkspaceManager draft_token_workspace_manager,
                                           EngineConfig engine_config,
                                           Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Find the requests in `waiting_queue` that can prefill in this step.
    std::vector<PrefillInput> prefill_inputs;
    {
      NVTXScopedRange nvtx_scope("NewRequestPrefill getting requests");
      prefill_inputs = GetRequestStateEntriesToPrefill(estate);
      if (prefill_inputs.empty()) {
        return {};
      }
    }

    int num_rsentries = prefill_inputs.size();
    auto tstart = std::chrono::high_resolution_clock::now();

    // - Update status of request states from pending to alive.
    Array<String> request_ids;
    std::vector<RequestState> rstates_of_entries;
    std::vector<RequestStateStatus> status_before_prefill;
    request_ids.reserve(num_rsentries);
    rstates_of_entries.reserve(num_rsentries);
    status_before_prefill.reserve(num_rsentries);
    for (const PrefillInput& prefill_input : prefill_inputs) {
      const RequestStateEntry& rsentry = prefill_input.rsentry;
      const Request& request = rsentry->request;
      RequestState request_rstate = estate->GetRequestState(request);
      request_ids.push_back(request->id);
      status_before_prefill.push_back(rsentry->status);
      rsentry->status = RequestStateStatus::kAlive;

      if (status_before_prefill.back() == RequestStateStatus::kPending) {
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
      rstates_of_entries.push_back(std::move(request_rstate));
    }

    // - Get embedding and run prefill for each model.
    std::vector<int> prefill_lengths;
    prefill_lengths.resize(/*size=*/num_rsentries, /*value=*/-1);
    ObjectRef hidden_states_for_input{nullptr};
    ObjectRef hidden_states_for_sample{nullptr};
    NDArray logits_for_sample{nullptr};
    // A map used to record the entry and child_idx pair needed to fork sequence.
    // The base model (id 0) should record all the pairs and all the small models
    // fork sequences according to this map.
    std::unordered_map<int, std::unordered_set<int>> fork_rsentry_child_map;
    for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
      std::vector<int64_t> request_internal_ids;
      request_internal_ids.reserve(num_rsentries);
      ObjectRef embeddings = model_workspaces_[model_id].embeddings;
      int cum_prefill_length = 0;
      bool single_input =
          num_rsentries == 1 && prefill_inputs[0].rsentry->mstates[model_id]->inputs.size() == 1;
      for (int i = 0; i < num_rsentries; ++i) {
        const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
        RequestModelState mstate = rsentry->mstates[model_id];
        auto [input_data, input_length] =
            ChunkPrefillInputData(mstate, prefill_inputs[i].max_prefill_length);
        if (prefill_lengths[i] == -1) {
          prefill_lengths[i] = input_length;
        } else {
          ICHECK_EQ(prefill_lengths[i], input_length);
        }

        ICHECK(mstate->draft_output_tokens.empty());
        ICHECK(mstate->draft_token_slots.empty());
        if (status_before_prefill[i] == RequestStateStatus::kPending) {
          // Add the sequence to the model, or fork the sequence from its parent.
          if (rsentry->parent_idx == -1) {
            models_[model_id]->AddNewSequence(mstate->internal_id);
          } else {
            models_[model_id]->ForkSequence(
                rstates_of_entries[i]->entries[rsentry->parent_idx]->mstates[model_id]->internal_id,
                mstate->internal_id);
          }
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            models_[model_id]->EnableSlidingWindowForSeq(mstate->internal_id);
          }
        }
        request_internal_ids.push_back(mstate->internal_id);
        RECORD_EVENT(trace_recorder_, prefill_inputs[i].rsentry->request->id, "start embedding");
        // Speculative models shift left the input tokens by 1 when base model has committed tokens.
        // Note: for n > 1 cases Eagle doesn't work because parent entry doesn't shift input tokens.
        int embed_offset =
            prefill_inputs[i].rsentry->mstates[model_id]->committed_tokens.empty() ? 0 : 1;
        for (int j = 0; j < static_cast<int>(input_data.size()); ++j) {
          if (j == static_cast<int>(input_data.size()) - 1) {
            std::vector<int32_t> tail_tokens;
            TokenData tk_data = Downcast<TokenData>(input_data[j]);
            CHECK(tk_data.defined());
            for (int k = embed_offset; k < static_cast<int>(tk_data->token_ids.size()); ++k) {
              tail_tokens.push_back(tk_data->token_ids[k]);
            }
            embeddings = models_[model_id]->TokenEmbed(
                {tail_tokens.begin(), tail_tokens.end()},
                /*dst=*/!single_input ? &model_workspaces_[model_id].embeddings : nullptr,
                /*offset=*/cum_prefill_length);
            cum_prefill_length += input_data[j]->GetLength();
            cum_prefill_length -= embed_offset;
          } else {
            embeddings = input_data[i]->GetEmbedding(
                models_[model_id],
                /*dst=*/!single_input ? &model_workspaces_[model_id].embeddings : nullptr,
                /*offset=*/cum_prefill_length);
            cum_prefill_length += input_data[j]->GetLength();
          }
        }
        if (embed_offset > 0) {
          std::vector<int32_t> new_tokens = {prefill_inputs[i]
                                                 .rsentry->mstates[model_id]
                                                 ->committed_tokens.back()
                                                 .sampled_token_id.first};
          embeddings =
              models_[model_id]->TokenEmbed({new_tokens.begin(), new_tokens.end()},
                                            /*dst=*/&model_workspaces_[model_id].embeddings,
                                            /*offset=*/cum_prefill_length);
          cum_prefill_length += new_tokens.size();
        }
        RECORD_EVENT(trace_recorder_, rsentry->request->id, "finish embedding");
      }

      RECORD_EVENT(trace_recorder_, request_ids, "start prefill");
      ObjectRef embedding_or_hidden_states{nullptr};
      if (model_id == 0) {
        embedding_or_hidden_states = embeddings;
      } else {
        embedding_or_hidden_states = models_[model_id]->FuseEmbedHidden(
            embeddings, hidden_states_for_input, /*batch_size*/ 1, /*seq_len*/ cum_prefill_length);
      }
      // hidden_states: (b * s, h)
      ObjectRef hidden_states = models_[model_id]->BatchPrefillToLastHidden(
          embedding_or_hidden_states, request_internal_ids, prefill_lengths);
      RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");

      if (model_id == 0) {
        // We only need to sample for model 0 in prefill.
        hidden_states_for_input = hidden_states;
      }

      // Whether to use base model to get logits.
      int sample_model_id = !models_[model_id]->CanGetLogits() ? 0 : model_id;

      std::vector<int> logit_positions;
      {
        // Prepare the logit positions
        logit_positions.reserve(prefill_lengths.size());
        int total_len = 0;
        for (int i = 0; i < prefill_lengths.size(); ++i) {
          total_len += prefill_lengths[i];
          logit_positions.push_back(total_len - 1);
        }
      }
      // hidden_states_for_sample: (b * s, h)
      hidden_states_for_sample = models_[sample_model_id]->GatherHiddenStates(
          hidden_states, logit_positions, &model_workspaces_[model_id].hidden_states);
      // logits_for_sample: (b * s, v)
      logits_for_sample =
          models_[sample_model_id]->GetLogits(hidden_states_for_sample, 1, num_rsentries);
      // - Update logits.
      ICHECK(logits_for_sample.defined());
      Array<GenerationConfig> generation_cfg;
      Array<RequestModelState> mstates_for_logitproc;
      generation_cfg.reserve(num_rsentries);
      mstates_for_logitproc.reserve(num_rsentries);
      for (int i = 0; i < num_rsentries; ++i) {
        generation_cfg.push_back(prefill_inputs[i].rsentry->request->generation_cfg);
        mstates_for_logitproc.push_back(prefill_inputs[i].rsentry->mstates[sample_model_id]);
      }
      logits_for_sample = logits_for_sample.CreateView({num_rsentries, logits_for_sample->shape[2]},
                                                       logits_for_sample->dtype);
      logit_processor_->InplaceUpdateLogits(logits_for_sample, generation_cfg,
                                            mstates_for_logitproc, request_ids);

      // - Compute probability distributions.
      NDArray probs_on_device =
          logit_processor_->ComputeProbsFromLogits(logits_for_sample, generation_cfg, request_ids);

      // - Sample tokens.
      //   For prefill_inputs which have children, sample
      //   one token for each rstate that is depending.
      //   Otherwise, sample a token for the current rstate.
      std::vector<int> sample_indices;
      std::vector<RequestStateEntry> rsentries_for_sample;
      std::vector<RandomGenerator*> rngs;
      std::vector<bool> rsentry_activated;
      sample_indices.reserve(num_rsentries);
      rsentries_for_sample.reserve(num_rsentries);
      rngs.reserve(num_rsentries);
      rsentry_activated.reserve(num_rsentries);
      request_ids.clear();
      generation_cfg.clear();
      for (int i = 0; i < num_rsentries; ++i) {
        const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
        int remaining_num_child_to_activate = prefill_inputs[i].num_child_to_activate;
        for (int child_idx : rsentry->child_indices) {
          // Only use base model to judge if we need to add child entries.
          if (rstates_of_entries[i]->entries[child_idx]->status == RequestStateStatus::kPending &&
              (rstates_of_entries[i]->entries[child_idx]->mstates[0]->committed_tokens.empty() ||
               fork_rsentry_child_map[i].count(child_idx))) {
            // If rstates_of_entries[i]->entries[child_idx] has no committed token,
            // the prefill of the current rsentry will unblock
            // rstates_of_entries[i]->entries[child_idx],
            // and thus we want to sample a token for rstates_of_entries[i]->entries[child_idx].
            fork_rsentry_child_map[i].insert(child_idx);
            sample_indices.push_back(i);
            rsentries_for_sample.push_back(rstates_of_entries[i]->entries[child_idx]);
            request_ids.push_back(rsentry->request->id);
            generation_cfg.push_back(rsentry->request->generation_cfg);
            rngs.push_back(&rstates_of_entries[i]->entries[child_idx]->rng);

            // We only fork the first `num_child_to_activate` children.
            // The children not being forked will be forked via later prefills.
            // Usually `num_child_to_activate` is the same as the number of children.
            // But it can be fewer subject to the KV cache max num sequence limit.
            if (remaining_num_child_to_activate == 0) {
              rsentry_activated.push_back(false);
              continue;
            }
            rsentry_activated.push_back(true);
            --remaining_num_child_to_activate;
            if (model_id == 0) {
              ICHECK(rstates_of_entries[i]->entries[child_idx]->status ==
                     RequestStateStatus::kPending);
              rstates_of_entries[i]->entries[child_idx]->status = RequestStateStatus::kAlive;
            }
            int64_t child_internal_id =
                rstates_of_entries[i]->entries[child_idx]->mstates[model_id]->internal_id;
            models_[model_id]->ForkSequence(rsentry->mstates[model_id]->internal_id,
                                            child_internal_id);
            // Enable sliding window for the child sequence if the child is not a parent.
            if (rstates_of_entries[i]->entries[child_idx]->child_indices.empty()) {
              models_[model_id]->EnableSlidingWindowForSeq(child_internal_id);
            }
          }
        }
        if (rsentry->child_indices.empty()) {
          // If rsentry has no child, we sample a token for itself.
          sample_indices.push_back(i);
          rsentries_for_sample.push_back(rsentry);
          request_ids.push_back(rsentry->request->id);
          generation_cfg.push_back(rsentry->request->generation_cfg);
          rngs.push_back(&rsentry->rng);
          rsentry_activated.push_back(true);
        }
      }

      NDArray renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
          probs_on_device, sample_indices, request_ids, generation_cfg);
      std::vector<SampleResult> sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
          renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
      ICHECK_EQ(sample_results.size(), rsentries_for_sample.size());

      // - Update the committed tokens of states.
      // - If a request is first-time prefilled, set the prefill finish time.
      auto tnow = std::chrono::high_resolution_clock::now();
      if (model_id == 0) {
        for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
          for (int mid = 0; mid < static_cast<int>(models_.size()); ++mid) {
            rsentries_for_sample[i]->mstates[mid]->CommitToken(sample_results[i]);
            if (!rsentry_activated[i]) {
              // When the child rsentry is not activated,
              // add the sampled token as an input of the mstate for prefill.
              rsentries_for_sample[i]->mstates[mid]->inputs.push_back(
                  TokenData(std::vector<int64_t>{sample_results[i].sampled_token_id.first}));
            }
          }
          // Only base model trigger timing records.
          if (rsentries_for_sample[i]->mstates[0]->committed_tokens.size() == 1) {
            rsentries_for_sample[i]->tprefill_finish = tnow;
          }
        }
      } else {
        // - Slice and save hidden_states_for_sample
        draft_token_workspace_manager_->AllocSlots(rsentries_for_sample.size(),
                                                   &draft_token_slots_);
        models_[model_id]->ScatterDraftProbs(renormalized_probs, draft_token_slots_,
                                             &model_workspaces_[0].draft_probs_storage);
        if (engine_config_->spec_draft_length > 1) {
          models_[model_id]->ScatterHiddenStates(hidden_states_for_sample, draft_token_slots_,
                                                 &model_workspaces_[0].draft_hidden_states_storage);
        }
        for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
          rsentries_for_sample[i]->mstates[model_id]->AddDraftToken(sample_results[i],
                                                                    draft_token_slots_[i]);
          estate->stats.total_draft_length += 1;
        }
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_prefill_time += static_cast<double>((tend - tstart).count()) / 1e9;

    // - Remove the request from waiting queue if all its request states
    // are now alive and have no remaining chunked inputs.
    std::vector<Request> processed_requests;
    {
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
          auto it = std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(),
                              rsentry->request);
          ICHECK(it != estate->waiting_queue.end());
          estate->waiting_queue.erase(it);
        }
      }
    }
    return processed_requests;
  }

 private:
  /*! \brief The class of request state entry and its maximum allowed length for prefill. */
  struct PrefillInput {
    RequestStateEntry rsentry;
    int max_prefill_length = 0;
    int num_child_to_activate = 0;
  };

  /*!
   * \brief Find one or multiple request state entries to run prefill.
   * \param estate The engine state.
   * \return The request entries to prefill, together with their input lengths.
   */
  std::vector<PrefillInput> GetRequestStateEntriesToPrefill(EngineState estate) {
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
                         current_total_seq_len, num_running_rsentries)) {
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
                       num_available_pages, current_total_seq_len, num_running_rsentries)) {
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
  bool CanPrefill(EngineState estate, int num_prefill_rsentries, int total_input_length,
                  int num_required_pages, int num_available_pages, int current_total_seq_len,
                  int num_running_rsentries) {
    ICHECK_LE(num_running_rsentries, engine_config_->max_num_sequence);

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
  std::pair<Array<Data>, int> ChunkPrefillInputData(const RequestModelState& mstate,
                                                    int max_prefill_length) {
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

  /*! \brief The models to run prefill in. */
  Array<Model> models_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Workspace of each model. */
  std::vector<ModelWorkspace> model_workspaces_;
  /*! \brief The draft token workspace manager. */
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Temporary buffer to store the slots of the current draft tokens */
  std::vector<int> draft_token_slots_;
};

EngineAction EngineAction::EagleNewRequestPrefill(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces,
    DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
    Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<EagleNewRequestPrefillActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
