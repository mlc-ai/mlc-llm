/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/batch_decode.cc
 */

#include <tvm/runtime/nvtx.h>

#include <numeric>

#include "../../support/random.h"
#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
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
  explicit BatchDecodeActionObj(Array<Model> models, Tokenizer tokenizer,
                                LogitProcessor logit_processor, Sampler sampler,
                                EngineConfig engine_config,
                                Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        tokenizer_(std::move(tokenizer)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Do not run decode when there is no running request.
    if (estate->running_queue.empty()) {
      return {};
    }

    // Preempt request state entries when decode cannot apply.
    std::vector<RequestStateEntry> running_rsentries;
    {
      NVTXScopedRange nvtx_scope("BatchDecode getting requests");
      running_rsentries = estate->GetRunningRequestStateEntries();
      while (!CanDecode(running_rsentries.size())) {
        if (estate->prefix_cache->TryFreeMemory()) continue;
        RequestStateEntry preempted =
            PreemptLastRunningRequestStateEntry(estate, models_, std::nullopt, trace_recorder_);
        if (preempted.same_as(running_rsentries.back())) {
          running_rsentries.pop_back();
        }
      }
      while (running_rsentries.size() >
             std::min(static_cast<int64_t>(engine_config_->max_num_sequence),
                      engine_config_->prefill_chunk_size)) {
        running_rsentries.pop_back();
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running request states at a time.
    int num_rsentries = running_rsentries.size();
    TVM_FFI_ICHECK_GT(num_rsentries, 0)
        << "There should be at least one request state entry that can run decode. "
           "Possible failure reason: none of the prefill phase of the running requests is finished";
    TVM_FFI_ICHECK_LE(num_rsentries, engine_config_->max_num_sequence)
        << "The number of running requests exceeds the max number of sequence in EngineConfig. "
           "Possible failure reason: the prefill action allows new sequence in regardless of the "
           "max num sequence.";
    // Collect
    // - the last committed token,
    // - the request id,
    // - the generation config,
    // - the random number generator,
    // of each request state entry.
    std::vector<int> input_tokens;
    std::vector<int> lengths;
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

    {
      NVTXScopedRange nvtx_scope("BatchDecode setting batch info");
      for (const RequestStateEntry& rsentry : running_rsentries) {
        auto mstate = rsentry->mstates[0];
        TVM_FFI_ICHECK(mstate->num_tokens_for_next_decode > 0 &&
                       mstate->num_tokens_for_next_decode <=
                           static_cast<int>(mstate->committed_tokens.size()));

        for (auto begin = mstate->committed_tokens.end() - mstate->num_tokens_for_next_decode;
             begin != mstate->committed_tokens.end(); ++begin) {
          input_tokens.push_back(begin->GetTokenId());
        }

        lengths.push_back(mstate->num_tokens_for_next_decode);
        mstate->num_tokens_for_next_decode = 0;

        request_ids.push_back(rsentry->request->id);
        request_internal_ids.push_back(mstate->internal_id);
        mstates.push_back(mstate);
        generation_cfg.push_back(rsentry->request->generation_cfg);
        rngs.push_back(&rsentry->rng);
      }
    }

    // - Compute embeddings.
    RECORD_EVENT(trace_recorder_, request_ids, "start embedding");
    ObjectRef embeddings =
        models_[0]->TokenEmbed({IntTuple(input_tokens.begin(), input_tokens.end())});
    RECORD_EVENT(trace_recorder_, request_ids, "finish embedding");

    // - Invoke model decode.
    // If every request only requires to process one token, batch decode kernel is called.
    // Otherwise, batch prefill kernel is called.
    bool is_every_request_single_token =
        std::all_of(lengths.begin(), lengths.end(), [](int len) { return len == 1; });
    RECORD_EVENT(trace_recorder_, request_ids, "start decode");
    Tensor logits;
    if (is_every_request_single_token) {
      logits = models_[0]->BatchDecode(embeddings, request_internal_ids);
      TVM_FFI_ICHECK_EQ(logits->ndim, 3);
      TVM_FFI_ICHECK_EQ(logits->shape[0], num_rsentries);
      TVM_FFI_ICHECK_EQ(logits->shape[1], 1);
    } else {
      logits = models_[0]->BatchPrefill(embeddings, request_internal_ids, lengths);
      TVM_FFI_ICHECK_EQ(logits->ndim, 3);
      TVM_FFI_ICHECK_EQ(logits->shape[0], 1);
      TVM_FFI_ICHECK_EQ(logits->shape[1], num_rsentries);
    }
    RECORD_EVENT(trace_recorder_, request_ids, "finish decode");

    // - Update logits.
    logits = logits.CreateView({num_rsentries, logits->shape[2]}, logits->dtype);
    logit_processor_->InplaceUpdateLogits(logits, generation_cfg, mstates, request_ids);

    // - Compute probability distributions.
    Tensor probs_on_device =
        logit_processor_->ComputeProbsFromLogits(logits, generation_cfg, request_ids);

    // - Commit the prefix cache changes from previous round of action.
    // Note: we commit prefix cache changes here to overlap this commit with the GPU execution.
    estate->prefix_cache->CommitSequenceExtention();

    // - Sample tokens.
    // Fill range [0, num_rsentries) into `sample_indices`.
    std::vector<int> sample_indices(num_rsentries);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
        probs_on_device, sample_indices, request_ids, generation_cfg);
    std::vector<SampleResult> sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
        renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
    TVM_FFI_ICHECK_EQ(sample_results.size(), num_rsentries);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_rsentries; ++i) {
      auto mstate = mstates[i];

      if (!mstate->require_retokenization_in_next_decode) {
        mstates[i]->CommitToken(sample_results[i]);
        // live update the output metrics
        running_rsentries[i]->rstate->metrics.completion_tokens += 1;
      } else {
        // Retokenize and commit tokens.
        CommitTokenMayRetokenize(running_rsentries[i], mstate, sample_results[i]);
        mstate->require_retokenization_in_next_decode = false;
      }

      running_rsentries[i]->rstate->metrics.decode_tokens += lengths[i];
    }

    double elapsed_time;
    {
      NVTXScopedRange nvtx_scope("BatchDecode get time");
      auto tend = std::chrono::high_resolution_clock::now();
      elapsed_time = static_cast<double>((tend - tstart).count()) / 1e9;
    }
    estate->metrics.engine_decode_time_sum += elapsed_time;
    estate->metrics.UpdateDecodeTimeByBatchSize(num_rsentries, elapsed_time);

    return estate->running_queue;
  }

 private:
  /*! \brief Check if the input request state entries can be decoded under conditions. */
  bool CanDecode(int num_rsentries) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries <= num_available_pages;
  }

  /*!
   * \brief Retokenize the past tokens with a new token.
   * \param mstate The model state.
   * \param token_id The new token id.
   * \param max_rollback_tokens The maximum number of tokens to rollback.
   * \return The number of tokens to rollback and the new tokens.
   */
  std::pair<int, std::vector<int32_t>> RetokenizeWithNewToken(RequestModelState mstate,
                                                              int32_t token_id,
                                                              int max_rollback_tokens) {
    // Step 1. Get past tokens
    // past_tokens = mstate[-max_rollback_tokens:]
    // past_string = detokenize(past_tokens)
    const auto& token_table = tokenizer_->PostProcessedTokenTable();
    std::vector<int32_t> past_tokens;
    std::string past_string;
    auto past_begin_it = mstate->committed_tokens.size() >= max_rollback_tokens
                             ? mstate->committed_tokens.end() - max_rollback_tokens
                             : mstate->committed_tokens.begin();
    for (auto it = past_begin_it; it != mstate->committed_tokens.end(); ++it) {
      past_tokens.push_back(it->GetTokenId());
      past_string += token_table[it->GetTokenId()];
    }

    // Step 2. Retokenize
    // Compare tokenize(past_string + new_string) and past_tokens
    auto new_tokens = tokenizer_->EncodeNoPrependSpace(past_string + token_table[token_id]);

    int first_differ_idx = past_tokens.size();
    for (int i = 0; i < static_cast<int>(past_tokens.size()); ++i) {
      if (i == static_cast<int>(new_tokens.size()) || past_tokens[i] != new_tokens[i]) {
        first_differ_idx = i;
        break;
      }
    }

    return {past_tokens.size() - first_differ_idx,
            std::vector<int32_t>(new_tokens.begin() + first_differ_idx, new_tokens.end())};
  }

  /*!
   * \brief Commit the token and may retokenize the past tokens.
   * \param rsentry The request state entry.
   * \param mstate The model state.
   * \param sample_result The sampled token.
   */
  void CommitTokenMayRetokenize(RequestStateEntry rsentry, RequestModelState mstate,
                                const SampleResult& sample_result) {
    auto generation_cfg = rsentry->request->generation_cfg;
    // 1. If EOS token is generated, jump commit it
    if (!generation_cfg->debug_config.ignore_eos &&
        std::any_of(generation_cfg->stop_token_ids.begin(), generation_cfg->stop_token_ids.end(),
                    [&](int32_t token) { return token == sample_result.GetTokenId(); })) {
      mstate->CommitToken(sample_result);
      rsentry->rstate->metrics.completion_tokens += 1;
      return;
    }

    // 2. Check retokenization
    const auto& committed_tokens = mstate->committed_tokens;
    auto [rollback_cnt, new_tokens] =
        RetokenizeWithNewToken(mstate, sample_result.GetTokenId(), MAX_ROLLBACK_TOKENS_);

    // 3. Handle output when retokenization happens
    if (rollback_cnt >
        static_cast<int>(committed_tokens.size()) - rsentry->next_callback_token_pos) {
      const auto& token_table = tokenizer_->PostProcessedTokenTable();
      for (auto i = rsentry->next_callback_token_pos; i < committed_tokens.size(); ++i) {
        auto token_id = committed_tokens[i].GetTokenId();
        rsentry->extra_prefix_string += token_table[token_id];
      }
      rsentry->extra_prefix_string += token_table[sample_result.GetTokenId()];
      rsentry->next_callback_token_pos = static_cast<int>(committed_tokens.size()) - rollback_cnt +
                                         static_cast<int>(new_tokens.size());
    }

    if (rollback_cnt > 0) {
      mstate->RollbackTokens(rollback_cnt);
      models_[0]->PopNFromKVCache(mstate->internal_id, rollback_cnt);
    }

    for (auto token_id : new_tokens) {
      mstate->CommitToken({{token_id, 1.0}, {}});
    }

    rsentry->rstate->metrics.completion_tokens +=
        static_cast<int>(new_tokens.size()) - rollback_cnt;
  }

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The tokenizer of the engine. */
  Tokenizer tokenizer_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief The maximum number of tokens to retokenize and may be rolled back. */
  const int MAX_ROLLBACK_TOKENS_ = 10;
};

EngineAction EngineAction::BatchDecode(Array<Model> models, Tokenizer tokenizer,
                                       LogitProcessor logit_processor, Sampler sampler,
                                       EngineConfig engine_config,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<BatchDecodeActionObj>(
      std::move(models), std::move(tokenizer), std::move(logit_processor), std::move(sampler),
      std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
