/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.cc
 * \brief The implementation for runtime module of serving engine module in MLC LLM.
 */
#define __STDC_FORMAT_MACROS

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "model.h"
#include "request.h"
#include "request_state.h"
#include "sampler.h"
#include "tokenizer.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

class EngineModule;

/*!
 * \brief The engine for request serving in MLC LLM.
 * The engine can run one or multiple LLM models internally for
 * text generation. Usually, when there are multiple models,
 * speculative inference will be activated, where the first model
 * (index 0) is the main "large model" that has better generation
 * quality, and all other models are "small" models that used for
 * speculation.
 * The engine receives requests from the "AddRequest" method. For
 * an given request, the engine will keep generating new tokens for
 * the request until finish (under certain criterion). After finish,
 * the engine will return the generation result through the callback
 * function provided by the request.
 * \note For now only one model run in the engine is supported.
 * Multiple model support such as speculative inference will
 * be followed soon in the future.
 */
class Engine {
  friend class EngineModule;

 public:
  /*!
   * \brief (Re)initialize the engine with the given lists of
   * models and KV cache config.
   * \param reload_libs The model libraries of the input models.
   * \param model_paths The weight/config directories of the input models.
   * \param devices The devices where each of the input model runs.
   * \param kv_cache_config_json The page KV cache configuration.
   * \note `reload_libs`, `model_paths` and `devices` should have the same size.
   */
  void Reload(std::vector<TVMArgValue> reload_libs, std::vector<String> model_paths,
              std::vector<DLDevice> devices, String kv_cache_config_json) {
    int num_models = reload_libs.size();
    ICHECK_GE(num_models, 1);
    ICHECK_EQ(model_paths.size(), num_models);
    ICHECK_EQ(devices.size(), num_models);
    devices_ = std::move(devices);

    // Step 1. Create models.
    ICHECK(models_.empty());
    models_.reserve(num_models);
    for (int i = 0; i < num_models; ++i) {
      models_.push_back(CreateModelModule(reload_libs[i], model_paths[i], devices_[i]));
    }
    // Step 2. Fetch max single sequence length from models.
    max_single_sequence_length_ = std::numeric_limits<int>::max();
    for (Module model : models_) {
      int max_window_size = model->GetFunction("get_max_window_size")();
      max_single_sequence_length_ = std::min(max_single_sequence_length_, max_window_size);
    }
    // Step 3. Process KV cache config json string.
    kv_cache_config_ = KVCacheConfig(kv_cache_config_json, max_single_sequence_length_);
    // Step 4. Create KV cache for each model.
    for (Module model : models_) {
      model->GetFunction("create_kv_cache")(kv_cache_config_);
    }
    // Step 5. Create sampler and tokenizer.
    //         The sampler is created one per model on each device.
    //         The tokenizer is created from the first model.
    //         We assume all models have the same tokenizer, which is the basic
    //         requirement of speculative encoding.
    for (int i = 0; i < num_models; ++i) {
      samplers_.push_back(CreateSamplerModule(devices_[i]));
    }
    tokenizer_ = CreateTokenizerModule(model_paths[0]);

    ResetEngine();
  }

  /*!
   * \brief Add a new request to the engine.
   * \param request The request to add.
   */
  void AddRequest(Request request) {
    waiting_queue_.push_back(request);
    request_states_.emplace(request, RequestState(models_.size(), request->inputs));
  }

  /*! \brief Abort the input request. */
  void AbortRequest(Request request) { abort_queue_.push_back(request); }

  /*!
   * \brief The main function that the engine takes a step of action.
   * At each step, the engine may decide to
   * - run prefill for one (or more) requests,
   * - run one-step decode for the all existing requests
   * ...
   * In the end of certain actions (e.g., decode), the engine will
   * check if any request has finished, and will return the
   * generation results for those finished requests.
   */
  void Step() {
    // - Abort requests.
    while (!abort_queue_.empty()) {
      StepAbort(abort_queue_.front());
      abort_queue_.erase(abort_queue_.begin());
    }

    // - Action 1. Prefill the front-most waiting request.
    bool prefill_processed = StepPrefill();
    if (prefill_processed) {
      return;
    }

    // - Action 2. Run speculation step for small models.
    // NOTE: Right now we do not really support speculation.
    //       Here we just reserve room for extension.
    bool speculate_processed = StepSpeculate();
    if (speculate_processed) {
      return;
    }
    // - Action 3. Run speculation verification step.
    bool verify_processed = StepVerify();
    if (verify_processed) {
      UpdateFinishedRequest();
      return;
    }

    // - Action 4. Run decode step.
    bool decode_processed = StepDecode();
    if (decode_processed) {
      UpdateFinishedRequest();
      return;
    }

    // - Action 5. Preempt the last running sequence.
    if (!running_queue_.empty()) {
      ICHECK_GT(static_cast<int>(running_queue_.size()), 1);
      StepPreempt(running_queue_.back());
    }
  }

  /*! \brief Reset the engine and clean up all running data. */
  void ResetEngine() {
    running_queue_.clear();
    waiting_queue_.clear();
    abort_queue_.clear();
    request_states_.clear();

    for (Module model : models_) {
      model->GetFunction("reset")();
    }

    current_total_seq_len_ = 0;
    tokenize_cache_.clear();
  }

 private:
  /*! \brief Pick applicable requests and run prefill. */
  bool StepPrefill() {
    auto [requests, sample_new_token] = GetRequestsToPrefill();
    ICHECK_EQ(requests.size(), sample_new_token.size());
    if (requests.empty()) {
      return false;
    }

    for (Request request : requests) {
      int req_id = running_queue_.size();
      auto it = std::find(waiting_queue_.begin(), waiting_queue_.end(), request);
      if (it == waiting_queue_.end()) {
        continue;
      }

      // - Move request from waiting queue to running queue.
      waiting_queue_.erase(it);
      running_queue_.push_back(request);
      // - Assign request id for the requests.
      AssignIDForRequest(request, req_id);
    }

    // NOTE: Right now only single-sequence prefill is supported.
    //       So we prefill the requests one by one for now.
    for (int req_idx = 0; req_idx < static_cast<int>(requests.size()); ++req_idx) {
      Request request = requests[req_idx];
      RequestState& state = request_states_.at(request);
      ICHECK_EQ(state.mstates.size(), models_.size());
      NDArray logits{nullptr};
      current_total_seq_len_ += GetRequestPrefillInputLength(request);
      // - Prefill the inputs for each model.
      for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
        Module model = models_[model_id];
        RequestModelState mstate = state.mstates[model_id];
        ICHECK(mstate->draft_output_tokens.empty());
        ICHECK(mstate->draft_output_token_prob.empty());
        ICHECK(mstate->draft_output_prob_dist.empty());
        if (mstate->inputs.empty()) {
          continue;
        }
        for (int i = 0; i < static_cast<int>(mstate->inputs.size()); ++i) {
          NDArray embedding = GetEmbedding(mstate->inputs[i], model);
          NDArray output_logits =
              model->GetFunction("single_seq_prefill")(embedding, mstate->request_id);

          // Only the output logits of the last input on the first model will be send for sampling.
          if (model_id == 0 && i == static_cast<int>(mstate->inputs.size()) - 1) {
            logits = output_logits;
          }
        }
        // Clean up `inputs` after prefill
        mstate->inputs.clear();
      }
      if (!sample_new_token[req_idx]) {
        continue;
      }

      ICHECK(logits.defined());
      ICHECK_EQ(logits->ndim, 3);
      ICHECK_EQ(logits->shape[0], 1);
      ICHECK_EQ(logits->shape[1], 1);

      std::vector<int32_t> next_token = SampleTokens(logits, models_[0], samplers_[0],
                                                     {state.mstates[0]}, {request->generation_cfg});
      ICHECK_EQ(next_token.size(), 1);

      // - Update the committed tokens of states.
      for (RequestModelState mstate : state.mstates) {
        mstate->committed_tokens.push_back(next_token[0]);
      }
    }
    return true;
  }

  /*! \brief Pick applicable requests and run decode. */
  bool StepDecode() {
    // - Do not run decode when there are multiple models.
    if (models_.size() > 1) {
      return false;
    }

    Array<Request> requests = GetRequestsToDecode();
    if (requests.empty()) {
      return false;
    }

    // NOTE: Right now we only support decode all the running requests at a time.
    ICHECK_EQ(requests.size(), running_queue_.size());
    int num_requests = requests.size();
    // Check if the requests ids are in an ascending order.
    for (int i = 1; i < num_requests; ++i) {
      ICHECK_GT(request_states_.at(requests[i]).mstates[0]->request_id,
                request_states_.at(requests[i - 1]).mstates[0]->request_id);
    }

    Module model = models_[0];
    current_total_seq_len_ += num_requests;
    // Collect
    // - the last committed token,
    // - the request states,
    // - the sampling parameters,
    // of each request.
    Array<Data> inputs;
    Array<RequestModelState> mstates;
    Array<GenerationConfig> generation_cfg;
    inputs.reserve(num_requests);
    mstates.reserve(num_requests);
    generation_cfg.reserve(num_requests);
    for (Request request : requests) {
      RequestState& state = request_states_.at(request);
      inputs.push_back(TokenData(ShapeTuple({state.mstates[0]->committed_tokens.back()})));
      mstates.push_back(state.mstates[0]);
      generation_cfg.push_back(request->generation_cfg);
    }

    // - Compute embeddings.
    NDArray embeddings = GetTokenEmbeddings(inputs, model, /*return_flattened_view=*/false);

    // - Invoke model decode.
    NDArray logits = model->GetFunction("decode")(embeddings);
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
    ICHECK_EQ(logits->shape[1], 1);

    // - Sample tokens.
    std::vector<int32_t> next_tokens =
        SampleTokens(logits, model, samplers_[0], mstates, generation_cfg);
    ICHECK_EQ(next_tokens.size(), num_requests);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_requests; ++i) {
      mstates[i]->committed_tokens.push_back(next_tokens[i]);
    }
    return true;
  }

  /*! \brief Pick applicable requests and run speculation. */
  bool StepSpeculate() {
    // - No speculate when there is only one model.
    if (models_.size() == 1) {
      return false;
    }

    // NOTE: We do not support speculation right now.
    // The following is the possible sketch implementation for speculation step.
    //
    //   Array<RequestModelState> mstates = GetRequestStatesToSpeculate();
    //   if (mstates.empty()) {
    //     return false;
    //   }
    //   ...
    ICHECK(false) << "Cannot reach here at this moment.";
    return true;
  }

  /*! \brief Pick applicable requests and run verification of speculation results. */
  bool StepVerify() {
    // - No verification when there is only one model.
    if (models_.size() == 1) {
      return false;
    }

    // NOTE: We do not support speculation and verification right now.
    // The following is the possible sketch implementation for speculation step.
    //
    //   Array<Request> requests = GetRequestsToVerify();
    //   if (requests.empty()) {
    //     return false;
    //   }
    //   ...
    ICHECK(false) << "Cannot reach here at this moment.";
    return true;
  }

  /*! \brief Abort the generation of the given request. */
  void StepAbort(Request request) {
    auto it_running = std::find(running_queue_.begin(), running_queue_.end(), request);
    auto it_waiting = std::find(waiting_queue_.begin(), waiting_queue_.end(), request);
    ICHECK(it_running != running_queue_.end() || it_waiting != waiting_queue_.end());
    if (it_running != running_queue_.end()) {
      // The request to abort is in running queue
      int req_id = it_running - running_queue_.begin();
      running_queue_.erase(it_running);
      current_total_seq_len_ -= GetRequestRawInputLength(request) +
                                request_states_.at(request).mstates[0]->committed_tokens.size() - 1;
      RemoveSequenceFromModels(req_id);
      UpdateRequestIDAfterRemoval(req_id);
    } else {
      // The request to abort is in waiting queue
      waiting_queue_.erase(it_waiting);
    }
    request_states_.erase(request);
  }

  /*!
   * \brief Preempt the generation of the given request, moving
   * it from running request set to the foremost of waiting
   * request queue.
   */
  void StepPreempt(Request request) {
    auto it = std::find(running_queue_.begin(), running_queue_.end(), request);
    ICHECK(it != running_queue_.end());

    // Remove from models.
    int req_id = it - running_queue_.begin();
    // - Reset `request_id` of states.
    // - Clear model speculation draft.
    // - Update `inputs` for future prefill.
    RequestState& state = request_states_.at(request);
    current_total_seq_len_ -=
        GetRequestRawInputLength(request) + state.mstates[0]->committed_tokens.size() - 1;
    for (RequestModelState mstate : state.mstates) {
      mstate->request_id = -1;
      mstate->draft_output_tokens.clear();
      mstate->draft_output_token_prob.clear();
      mstate->draft_output_prob_dist.clear();
      ICHECK(mstate->inputs.empty());
      ICHECK(!mstate->committed_tokens.empty());
      mstate->inputs = request->inputs;
      mstate->inputs.push_back(TokenData(mstate->committed_tokens));
    }
    RemoveSequenceFromModels(req_id);
    UpdateRequestIDAfterRemoval(req_id);

    // Move from running queue to the front of waiting queue.
    running_queue_.erase(it);
    waiting_queue_.insert(waiting_queue_.begin(), request);
  }

  /*!
   * \brief Find one or multiple requests to run prefill.
   * \return The requests to prefill. For each request, we
   * additionally return a boolean flag indicating if a new
   * token needs to be sampled from logits after prefill.
   */
  std::pair<Array<Request>, std::vector<bool>> GetRequestsToPrefill() {
    // NOTE: Right now we only support single-sequence prefill.
    if (!waiting_queue_.empty()) {
      Array<Request> prefill_requests{waiting_queue_.front()};
      if (CanPrefill(prefill_requests)) {
        // Need to sample a new token for waiting requests.
        return {prefill_requests, {true}};
      }
    }

    for (Request request : running_queue_) {
      Array<RequestModelState> mstates = request_states_.at(request).mstates;
      for (int i = 0; i < static_cast<int>(mstates.size()); ++i) {
        if (!mstates[i]->inputs.empty()) {
          ICHECK_NE(i, 0);
          // This return happens only for "small" models in
          // speculative inference settings.
          // Therefore no need to sample new token from logits.
          return {{request}, {false}};
        }
      }
    }

    return {};
  }

  /*! \brief Find requests to decode. */
  Array<Request> GetRequestsToDecode() {
    if (running_queue_.empty()) {
      return {};
    }

    // NOTE: Right now we only support decode all the running requests at a time.
    Array<Request> requests(running_queue_);
    if (CanDecode(requests)) {
      return requests;
    }
    return {};
  }

  /*! \brief Assign the given id for the given request. */
  void AssignIDForRequest(Request request, int req_id) {
    // Set id in the request state.
    RequestState& state = request_states_.at(request);
    for (RequestModelState mstate : state.mstates) {
      mstate->request_id = req_id;
    }
    // Add a new sequence to each model.
    for (Module model : models_) {
      int seq_id_in_model = model->GetFunction("add_new_sequence")();
      ICHECK_EQ(seq_id_in_model, req_id);
    }
  }

  /*! \brief Check if the input requests can be prefilled under conditions. */
  bool CanPrefill(Array<Request> requests) {
    int num_running_requests = running_queue_.size();
    ICHECK_LE(num_running_requests, kv_cache_config_->max_num_sequence);

    // No exceeding of the maximum allowed requests that can
    // run simultaneously.
    if (num_running_requests == kv_cache_config_->max_num_sequence) {
      return false;
    }

    // The total length + the maximum allowed single-sequence length does
    // not exceed the maximum allowed total length.
    // NOTE: this condition is heuristic and can be revised.
    int total_input_length = 0;
    for (Request request : requests) {
      total_input_length += GetRequestPrefillInputLength(request);
    }
    return current_total_seq_len_ + max_single_sequence_length_ <
           kv_cache_config_->max_total_sequence_length;
  }

  /*! \brief Check if the input requests can be decoded under conditions. */
  bool CanDecode(Array<Request> requests) {
    return current_total_seq_len_ + requests.size() < kv_cache_config_->max_total_sequence_length;
  }

  /*!
   * \brief Get the total equivalent **input length to prefill**
   * of the given request's current state.
   */
  int GetRequestPrefillInputLength(const Request& request) {
    auto it = request_states_.find(request);
    ICHECK(it != request_states_.end());

    const RequestState& state = it->second;
    ICHECK_EQ(state.mstates.size(), models_.size());
    int input_length = -1;
    for (const RequestModelState& mstate : state.mstates) {
      int length_sum = 0;
      for (Data input : mstate->inputs) {
        length_sum += GetInputLength(input);
      }
      if (input_length == -1) {
        input_length = length_sum;
      } else {
        ICHECK_EQ(length_sum, input_length);
      }
    }
    ICHECK_NE(input_length, -1);
    return input_length;
  }

  /*! \brief Get the total equivalent **input length** of the given request. */
  int GetRequestRawInputLength(const Request& request) {
    int length_sum = 0;
    for (Data input : request->inputs) {
      length_sum += GetInputLength(input);
    }
    return length_sum;
  }

  /*! \brief Get the equivalent length of the given input. */
  int GetInputLength(const Data& input) {
    // Dispatch according to input type.
    if (const auto* text_input = input.as<TextDataNode>()) {
      return Tokenize(text_input->text).size();
    } else if (const auto* tokens_input = input.as<TokenDataNode>()) {
      return tokens_input->token_ids.size();
    } else {
      ICHECK(false) << "Cannot reach here";
    }
  }

  /*!
   * \brief Tokenize the input text string using tokenizer.
   * \note We use an engine-wise tokenize cache.
   * The cache will be reset once its size reaches the full capacity.
   */
  ShapeTuple Tokenize(String text) {
    auto it = tokenize_cache_.find(text);
    if (it != tokenize_cache_.end()) {
      return it->second;
    }
    ShapeTuple token_ids = tokenizer_->GetFunction("tokenize")(text);
    tokenize_cache_.emplace(text, token_ids);

    // Clean up cache to avoid unlimited growth.
    static constexpr int max_cache_size = 100000;
    if (tokenize_cache_.size() == max_cache_size) {
      tokenize_cache_.clear();
    }

    return token_ids;
  }

  /*!
   * \brief Compute the embedding of the given **single** input with
   * regard to the given model.
   */
  NDArray GetEmbedding(Data input, Module model) {
    // Dispatch according to input type.
    if (const auto* text_input = input.as<TextDataNode>()) {
      ShapeTuple token_ids = Tokenize(text_input->text);
      return model->GetFunction("token_embed")(Array<ShapeTuple>{token_ids});
    } else if (const auto* tokens_input = input.as<TokenDataNode>()) {
      return model->GetFunction("token_embed")(Array<ShapeTuple>{tokens_input->token_ids});
    } else {
      ICHECK(false) << "Cannot reach here";
    }
  }

  /*!
   * \brief Get token embeddings for all inputs in a **batched style**.
   * It requires all inputs are either TextData or TokenData.
   * This function is usually called for batch-wise actions such as decode
   * (or batched prefill if supported in the future).
   * \param inputs The inputs to compute embeddings.
   * \param model The model to compute embeddings with regard to.
   * \param return_flattened_view A boolean indicating if flatten the
   * embeddings across the batch dimension or not. For batch decode we
   * do not flatten, and for batch prefill we usually flatten to handle
   * raggedness.
   * \return The computed embeddings with regard to the required view.
   */
  NDArray GetTokenEmbeddings(Array<Data> inputs, Module model, bool return_flattened_view) {
    CHECK(!inputs.empty());
    int num_inputs = inputs.size();
    Array<ShapeTuple> token_ids;
    token_ids.reserve(num_inputs);
    for (Data input : inputs) {
      if (const auto* text_input = input.as<TextDataNode>()) {
        token_ids.push_back(Tokenize(text_input->text));
      } else if (const auto* tokens_input = input.as<TokenDataNode>()) {
        token_ids.push_back(tokens_input->token_ids);
      } else {
        CHECK(false) << "Input type " << input->GetTypeKey() << " is not accepted";
      }
    }

    // - If it is expected to return in a flattened view, just return the embeddings.
    NDArray embeddings = model->GetFunction("token_embed")(token_ids);
    if (return_flattened_view) {
      return embeddings;
    }
    // - Otherwise, it is required that each input has the same length.
    // Because we cannot return embeddings with raggedness in an
    // unflattened way.
    int input_length = token_ids[0].size();
    for (ShapeTuple ids : token_ids) {
      CHECK_EQ(ids.size(), input_length)
          << "When it is required not to return flattened embeddings, "
             "all inputs are supposed to have the same length";
    }
    ICHECK_EQ(embeddings->ndim, 3);
    ICHECK_EQ(embeddings->shape[0], 1);
    ICHECK_EQ(embeddings->shape[1], input_length * num_inputs);
    return embeddings.CreateView({num_inputs, input_length, embeddings->shape[2]},
                                 embeddings->dtype);
  }

  /*!
   * \brief Sample tokens from the input logits.
   * \param logits_on_device The logits to sample tokens from.
   * \param model The LLM model module which contains the softmax
   * function on device that might be helpful.
   * \param sampler The sampler module to run sampling.
   * \param request_mstates The request states of each sequence in
   * the batch with regard to the given model.
   * \param generation_cfg The generation config of each request
   * in the input batch.
   * \return The sampled tokens, one for each request in the batch.
   */
  std::vector<int32_t> SampleTokens(NDArray logits_on_device, Module model, Module sampler,
                                    Array<RequestModelState> request_mstates,
                                    Array<GenerationConfig> generation_cfg) {
    ICHECK(logits_on_device.defined());
    ICHECK_EQ(logits_on_device->ndim, 3);
    ICHECK_EQ(logits_on_device->shape[1], 1);
    ICHECK_EQ(logits_on_device->shape[0], generation_cfg.size());
    ICHECK_EQ(request_mstates.size(), generation_cfg.size());

    int num_sequence = logits_on_device->shape[0];
    bool require_gpu_softmax = sampler->GetFunction("require_gpu_softmax")(generation_cfg);

    // - Compute probabilities from logits.
    std::vector<NDArray> probs;
    std::vector<int> prob_token_offsets;
    probs.reserve(num_sequence);
    prob_token_offsets.reserve(num_sequence);
    if (require_gpu_softmax) {
      NDArray probs_on_device =
          model->GetFunction("softmax_with_temperature")(logits_on_device, generation_cfg);
      for (int i = 0; i < num_sequence; ++i) {
        probs.push_back(probs_on_device);
        prob_token_offsets.push_back(i);
      }
    } else {
      for (int i = 0; i < num_sequence; ++i) {
        NDArray probs_on_cpu = sampler->GetFunction("compute_probs_from_logits")(
            logits_on_device, /*token_offset=*/i, request_mstates[i], generation_cfg[i]);
        ICHECK_EQ(probs_on_cpu->ndim, 1);
        probs.push_back(probs_on_cpu);
        prob_token_offsets.push_back(0);
      }
    }

    // - Sample tokens from probabilities.
    // NOTE: Though we have the probability field in RequestModelState,
    //       we do not save the probabilities right now.
    //       We will handle this in the future when we work on speculation.
    std::vector<int32_t> new_tokens;
    new_tokens.reserve(num_sequence);
    for (int i = 0; i < num_sequence; ++i) {
      int32_t token_id = sampler->GetFunction("sample_token_from_probs")(
          probs[i], prob_token_offsets[i], generation_cfg[i]);
      new_tokens.push_back(token_id);
    }
    return new_tokens;
  }

  /*! \brief Remove the given request from all models (usually the KV cache). */
  void RemoveSequenceFromModels(int req_id) {
    for (Module model : models_) {
      model->GetFunction("remove_sequence")(req_id);
    }
  }

  /*!
   * \brief Update the request ids of all running requests after
   * the removal of the given request.
   */
  void UpdateRequestIDAfterRemoval(int removed_req_id) {
    for (auto& it : request_states_) {
      RequestState& state = it.second;
      for (RequestModelState mstate : state.mstates) {
        ICHECK_NE(mstate->request_id, removed_req_id);
        if (mstate->request_id > removed_req_id) {
          --mstate->request_id;
        }
      }
    }
  }

  /*!
   * \brief For each request, check if the request has finished
   * its generation. And update the state and return the generation
   * result for the finished requests.
   * \note This function removes requests from the running request
   * queue.
   */
  void UpdateFinishedRequest() {
    // - Collect finished requests.
    //   We don't remove on the fly to avoid concurrent modification.
    std::vector<Request> request_to_remove;
    for (Request request : running_queue_) {
      if (RequestIsFinished(request)) {
        request_to_remove.push_back(request);
      }
    }

    // - Remove the finished request.
    for (Request request : request_to_remove) {
      auto it = std::find(running_queue_.begin(), running_queue_.end(), request);
      ICHECK(it != running_queue_.end());
      int req_id = it - running_queue_.begin();
      running_queue_.erase(it);

      RequestState& state = request_states_.at(request);
      current_total_seq_len_ -=
          GetRequestRawInputLength(request) + state.mstates[0]->committed_tokens.size() - 1;
      for (RequestModelState mstate : state.mstates) {
        ICHECK_EQ(mstate->request_id, req_id);
        mstate->request_id = -1;
      }
      RemoveSequenceFromModels(req_id);
      UpdateRequestIDAfterRemoval(req_id);

      // NOTE: right now we only return the generated text.
      // In the future we might optional return text or token ids.
      request->fcallback(request, TextData(state.output));

      // Remove the request from states.
      request_states_.erase(request);
    }
  }

  /*! \brief Check if the given request is finished under conditions. */
  bool RequestIsFinished(Request request) {
    RequestState& state = request_states_.at(request);

    // - Case 0. There is remaining draft output ==> Unfinished
    //   All draft outputs are supposed to be processed before finish.
    for (RequestModelState mstate : state.mstates) {
      if (!mstate->draft_output_tokens.empty()) {
        return false;
      }
    }

    // - Decode committed tokens.
    const std::vector<int32_t>& committed_tokens = state.mstates[0]->committed_tokens;
    String output = tokenizer_->GetFunction("decode")(
        ShapeTuple(committed_tokens.begin(), committed_tokens.end()));
    state.output = output.operator std::string();

    // Case 1. Any of the stop strings appears in output ==> Finished
    for (String stop_str : request->generation_cfg->stop_strs) {
      if (state.output.rfind(stop_str) != std::string::npos) {
        return true;
      }
    }
    // Case 2. Any of the stop tokens appears in the committed tokens ===> Finished
    if (std::any_of(request->generation_cfg->stop_tokens.begin(),
                    request->generation_cfg->stop_tokens.end(), [&committed_tokens](int32_t token) {
                      return token == committed_tokens.back();
                    })) {
      return true;
    }
    // Case 3. Generation reaches the specified max generation length ==> Finished
    if (static_cast<int>(committed_tokens.size()) >= request->generation_cfg->max_new_tokens) {
      return true;
    }
    // Case 4. Total length of the request reaches the maximum single sequence length ==> Finished
    if (GetRequestRawInputLength(request) + static_cast<int>(committed_tokens.size()) >=
        max_single_sequence_length_) {
      return true;
    }
    return false;
  }

  // Request queues
  std::vector<Request> running_queue_;
  std::vector<Request> waiting_queue_;
  std::vector<Request> abort_queue_;
  // Request states
  std::unordered_map<Request, RequestState, ObjectPtrHash, ObjectPtrEqual> request_states_;

  // Models
  Array<Module> models_;
  Array<Module> samplers_;
  Module tokenizer_;
  // Device corresponding to each model
  std::vector<DLDevice> devices_;

  // Runtime statistics
  int64_t current_total_seq_len_;
  // Tokenization cache
  std::unordered_map<String, ShapeTuple, ObjectPtrHash, ObjectPtrEqual> tokenize_cache_;

  // Configurations
  KVCacheConfig kv_cache_config_;
  int max_single_sequence_length_ = -1;
};

class EngineModule : public ModuleNode {
 public:
  // clear global memory manager
  static void ClearGlobalMemoryManager() {
    // Step 0. Clear the previously allocated memory.
    const PackedFunc* fclear_memory_manager =
        tvm::runtime::Registry::Get("vm.builtin.memory_manager.clear");
    ICHECK(fclear_memory_manager) << "Cannot find env function vm.builtin.memory_manager.clear";
    (*fclear_memory_manager)();
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "reload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        // The args of `reload` is expected to be in the following pattern.
        // Assume we want to load `n` models in the engine, then there
        // is supposed to have (4n + 1) arguments.
        // For each i (i from 0 to n),
        // - args[4 * i    ] denotes the model lib,
        // - args[4 * i + 1] denotes the model path (for weights/config),
        // - args[4 * i + 2] denotes the device type,
        // - args[4 * i + 3] denotes the device id.
        // And the last argument denotes the KV cache config in JSON string.

        engine_ = nullptr;
        ClearGlobalMemoryManager();
        engine_ = std::make_unique<Engine>(Engine());
        // num_models x (model lib, model path, device type, device id) + kv_cache_config
        std::vector<TVMArgValue> reload_libs;
        std::vector<String> model_paths;
        std::vector<DLDevice> devices;
        CHECK_EQ(args.size() % 4, 1)
            << "Unexpected number of reload arguments. "
               "Reload arguments should be one or many of (reload_lib, "
               "model_path, device_type, device_id) with a trailing KV cache config JSON string.";

        int num_models = args.size() / 4;
        reload_libs.reserve(num_models);
        model_paths.reserve(num_models);
        devices.reserve(num_models);
        for (int i = 0; i < num_models; ++i) {
          reload_libs.push_back(args[i * 4]);
          model_paths.push_back(args[i * 4 + 1]);
          int device_type = args[i * 4 + 2];
          int device_id = args[i * 4 + 3];
          devices.push_back(DLDevice{static_cast<DLDeviceType>(device_type), device_id});
        }
        engine_->Reload(reload_libs, model_paths, devices, args[num_models * 4]);
      });
    } else if (name == "unload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        engine_ = nullptr;
        ClearGlobalMemoryManager();
      });
    } else if (name == "add_request") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1);
        GetEngine()->AddRequest(args[0]);
      });
    } else if (name == "abort") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        GetEngine()->AbortRequest(args[0]);
      });
    } else if (name == "step") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        GetEngine()->Step();
      });
    } else if (name == "stats") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        // TODO: stats is not implemented and will be added soon in followup PRs.
      });
    } else if (name == "reset") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        GetEngine()->ResetEngine();
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  Engine* GetEngine() {
    ICHECK(engine_ != nullptr) << "Engine is not initialized via reload";
    return engine_.get();
  }

  const char* type_key() const final { return "mlc.serve.engine"; }

 private:
  std::unique_ptr<Engine> engine_ = nullptr;
};

tvm::runtime::Module CreateEngineModule() {
  ObjectPtr<EngineModule> n = make_object<EngineModule>();
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.serve.create_engine").set_body_typed([]() {
  return CreateEngineModule();
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
