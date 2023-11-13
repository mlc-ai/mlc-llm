/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.cc
 * \brief The implementation for runtime module of serving engine module in MLC LLM.
 */
#define __STDC_FORMAT_MACROS
#define PICOJSON_USE_INT64

#include <picojson.h>
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

    // Step 1. Create models and their PackedFuncs.
    ICHECK(models_.empty());
    models_.reserve(num_models);
    fmodel_batch_prefill_.clear();
    fmodel_decode_.clear();
    fmodel_token_embed_.clear();
    fmodel_add_new_sequence_.clear();
    fmodel_remove_sequence_.clear();
    fmodel_softmax_with_temperature_.clear();
    fmodel_get_num_available_pages_.clear();
    for (int i = 0; i < num_models; ++i) {
      Module model = CreateModelModule(reload_libs[i], model_paths[i], devices_[i]);
      models_.push_back(model);
      fmodel_batch_prefill_.push_back(model->GetFunction("batch_prefill"));
      fmodel_decode_.push_back(model->GetFunction("decode"));
      fmodel_token_embed_.push_back(model->GetFunction("token_embed"));
      fmodel_add_new_sequence_.push_back(model->GetFunction("add_new_sequence"));
      fmodel_remove_sequence_.push_back(model->GetFunction("remove_sequence"));
      fmodel_softmax_with_temperature_.push_back(model->GetFunction("softmax_with_temperature"));
      fmodel_get_num_available_pages_.push_back(model->GetFunction("get_num_available_pages"));
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
    fsampler_require_gpu_softmax_.clear();
    fsampler_compute_probs_from_logits_inplace_.clear();
    fsampler_sample_token_from_probs_.clear();
    for (int i = 0; i < num_models; ++i) {
      Module sampler = CreateSamplerModule(devices_[i]);
      samplers_.push_back(sampler);
      fsampler_require_gpu_softmax_.push_back(sampler->GetFunction("require_gpu_softmax"));
      fsampler_compute_probs_from_logits_inplace_.push_back(
          sampler->GetFunction("compute_probs_from_logits_inplace"));
      fsampler_sample_token_from_probs_.push_back(sampler->GetFunction("sample_token_from_probs"));
    }
    tokenizer_ = CreateTokenizerModule(model_paths[0]);
    ftokenizer_tokenize = tokenizer_->GetFunction("tokenize");
    ftokenizer_decode = tokenizer_->GetFunction("decode");

    ResetEngine();
  }

  /*!
   * \brief Add a new request to the engine.
   * \param request The request to add.
   */
  void AddRequest(Request request) {
    waiting_queue_.push_back(request);
    request_states_.emplace(
        request, RequestState(models_.size(), request->inputs, GetInputLength(request->inputs)));
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

  /*! \brief Reset the engine, clean up all running data and statistics. */
  void ResetEngine() {
    running_queue_.clear();
    waiting_queue_.clear();
    abort_queue_.clear();
    request_states_.clear();

    for (Module model : models_) {
      model->GetFunction("reset")();
    }

    current_total_seq_len_ = 0;
    request_total_prefill_time_ = 0.0f;
    request_total_decode_time_ = 0.0f;
    engine_total_prefill_time_ = 0.0f;
    engine_total_decode_time_ = 0.0f;
    total_prefill_length_ = 0;
    total_decode_length_ = 0;
    tokenize_cache_.clear();
  }

  /*!
   * \brief Return the engine runtime statistics in JSON string.
   * We collect the following entries:
   * - single token prefill latency (s/tok): avg latency of processing one token in prefill
   * - single token decode latency (s/tok): avg latency of processing one token in decode
   * - engine time for prefill (sec)
   * - engine time for decode (sec)
   * - total number of processed tokens in prefill.
   * - total number of processed tokens in decode.
   * \return The statistics in JSON string.
   */
  String StatisticsJSON() {
    picojson::object config;
    config["single_token_prefill_latency"] =
        picojson::value(request_total_prefill_time_ / total_prefill_length_);
    config["single_token_decode_latency"] =
        picojson::value(request_total_decode_time_ / total_decode_length_);
    config["engine_total_prefill_time"] = picojson::value(engine_total_prefill_time_);
    config["engine_total_decode_time"] = picojson::value(engine_total_decode_time_);
    config["total_prefill_tokens"] = picojson::value(total_prefill_length_);
    config["total_decode_tokens"] = picojson::value(total_decode_length_);
    return picojson::value(config).serialize(true);
  }

 private:
  /*! \brief Pick applicable requests and run prefill. */
  bool StepPrefill() {
    auto [requests, states, sample_new_token] = GetRequestsToPrefill();
    if (requests.empty()) {
      return false;
    }

    auto tstart = std::chrono::high_resolution_clock::now();

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

    int sum_prefill_lengths = 0;
    NDArray logits_for_sample{nullptr};
    Array<RequestModelState> mstates_for_sample;
    Array<GenerationConfig> generation_cfg_for_sample;
    mstates_for_sample.reserve(requests.size());
    generation_cfg_for_sample.reserve(requests.size());
    for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
      Module model = models_[model_id];
      auto [request_list, mstates, prefill_lengths] =
          FilterPrefillRequests(requests, states, model_id);
      Array<NDArray> embeddings;
      std::vector<int> request_ids;
      embeddings.reserve(request_list.size());
      request_ids.reserve(request_list.size());
      for (int i = 0; i < static_cast<int>(request_list.size()); ++i) {
        Request request = request_list[i];
        int prefill_length = prefill_lengths[i];
        RequestModelState mstate = mstates[i];
        if (model_id == 0) {
          // Accumulate the sequence length.
          sum_prefill_lengths += prefill_length;
          current_total_seq_len_ += prefill_length;
          mstates_for_sample.push_back(mstate);
          generation_cfg_for_sample.push_back(request->generation_cfg);
        }
        ICHECK(mstate->draft_output_tokens.empty());
        ICHECK(mstate->draft_output_token_prob.empty());
        ICHECK(mstate->draft_output_prob_dist.empty());
        ICHECK(!mstate->inputs.empty());
        request_ids.push_back(mstate->request_id);
        for (int i = 0; i < static_cast<int>(mstate->inputs.size()); ++i) {
          embeddings.push_back(GetEmbedding(mstate->inputs[i], fmodel_token_embed_[model_id]));
        }
        // Clean up `inputs` after prefill
        mstate->inputs.clear();
      }

      NDArray logits = fmodel_batch_prefill_[model_id](
          embeddings, ShapeTuple(request_ids.begin(), request_ids.end()), prefill_lengths);
      ICHECK_EQ(logits->ndim, 3);
      ICHECK_EQ(logits->shape[0], 1);
      ICHECK_EQ(logits->shape[1], request_list.size());

      if (model_id == 0) {
        // We only need to sample for model 0 in prefill.
        logits_for_sample = logits;
      }
    }

    if (sample_new_token) {
      // - Sample tokens.
      int num_requests = requests.size();
      ICHECK(logits_for_sample.defined());
      ICHECK_EQ(logits_for_sample->shape[1], num_requests);
      ICHECK_EQ(mstates_for_sample.size(), num_requests);
      ICHECK_EQ(generation_cfg_for_sample.size(), num_requests);
      logits_for_sample = logits_for_sample.CreateView(
          {num_requests, 1, logits_for_sample->shape[2]}, logits_for_sample->dtype);
      ShapeTuple next_tokens = SampleTokens(logits_for_sample, /*model_id=*/0, /*sampler_id=*/0,
                                            mstates_for_sample, generation_cfg_for_sample);
      ICHECK_EQ(next_tokens.size(), num_requests);
      // - Update the committed tokens of states.
      // - If a request is first-time prefilled, set the prefill finish time.
      auto tnow = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < num_requests; ++i) {
        mstates_for_sample[i]->committed_tokens.push_back(next_tokens[i]);
        if (mstates_for_sample[i]->committed_tokens.size() == 1) {
          request_states_.at(requests[i])->tprefill_finish = tnow;
        }
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    engine_total_prefill_time_ += static_cast<double>((tend - tstart).count()) / 1e9;

    return true;
  }

  /*! \brief Pick applicable requests and run decode. */
  bool StepDecode() {
    // - Do not run decode when there are multiple models.
    if (models_.size() > 1) {
      return false;
    }

    PreemptUnfittableRequests();
    if (running_queue_.empty()) {
      return false;
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running requests at a time.
    int num_requests = running_queue_.size();
    // Check if the requests ids are in an ascending order.
    for (int i = 1; i < num_requests; ++i) {
      ICHECK_GT(request_states_.at(running_queue_[i])->mstates[0]->request_id,
                request_states_.at(running_queue_[i - 1])->mstates[0]->request_id);
    }

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
    for (Request request : running_queue_) {
      RequestState& state = request_states_.at(request);
      inputs.push_back(TokenData(ShapeTuple({state->mstates[0]->committed_tokens.back()})));
      mstates.push_back(state->mstates[0]);
      generation_cfg.push_back(request->generation_cfg);
    }

    // - Compute embeddings.
    NDArray embeddings = GetTokenEmbeddings(inputs, fmodel_token_embed_[0],
                                            /*return_flattened_view=*/false);

    // - Invoke model decode.
    NDArray logits = fmodel_decode_[0](embeddings);
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
    ICHECK_EQ(logits->shape[1], 1);

    // - Sample tokens.
    ShapeTuple next_tokens =
        SampleTokens(logits, /*model_id=*/0, /*sampler_id=*/0, mstates, generation_cfg);
    ICHECK_EQ(next_tokens.size(), num_requests);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_requests; ++i) {
      mstates[i]->committed_tokens.push_back(next_tokens[i]);
    }

    auto tend = std::chrono::high_resolution_clock::now();
    engine_total_decode_time_ += static_cast<double>((tend - tstart).count()) / 1e9;

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
      RequestState state = request_states_.at(request);
      current_total_seq_len_ -=
          state->raw_input_length + state->mstates[0]->committed_tokens.size() - 1;
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
        state->raw_input_length + state->mstates[0]->committed_tokens.size() - 1;
    for (RequestModelState mstate : state->mstates) {
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
  std::tuple<Array<Request>, Array<RequestState>, bool> GetRequestsToPrefill() {
    // - Try to prefill pending requests.
    std::vector<Request> prefill_requests;
    std::vector<RequestState> states;
    if (!waiting_queue_.empty()) {
      int total_input_length = 0;
      int total_required_pages = 0;
      ICHECK(fmodel_get_num_available_pages_[0].defined());
      int num_available_pages = fmodel_get_num_available_pages_[0]();

      for (int i = 0; i < static_cast<int>(waiting_queue_.size()); ++i) {
        Request request = waiting_queue_[i];
        RequestState state = request_states_.at(request);
        int input_length = GetInputLength(state->mstates[0]->inputs);
        int num_require_pages =
            (input_length + kv_cache_config_->page_size - 1) / kv_cache_config_->page_size;
        total_input_length += input_length;
        total_required_pages += num_require_pages;
        if (CanPrefill(i + 1, total_input_length, total_required_pages, num_available_pages)) {
          prefill_requests.push_back(request);
          states.push_back(state);
        } else {
          total_input_length -= input_length;
          total_required_pages -= num_require_pages;
          break;
        }
      }
      if (!prefill_requests.empty()) {
        // Need to sample a new token for waiting requests.
        return {prefill_requests, states, true};
      }
    }

    // Try to prefill for small models.
    for (Request request : running_queue_) {
      RequestState state = request_states_.at(request);
      Array<RequestModelState> mstates = state->mstates;
      for (int i = 0; i < static_cast<int>(mstates.size()); ++i) {
        if (!mstates[i]->inputs.empty()) {
          ICHECK_NE(i, 0);
          prefill_requests.push_back(request);
          states.push_back(state);
          break;
        }
      }
    }
    // This return happens only for "small" models in
    // speculative inference settings.
    // Therefore no need to sample new token from logits.
    return {prefill_requests, states, false};
  }

  /*! \brief Preempt the requests unfittable for decode. */
  void PreemptUnfittableRequests() {
    if (running_queue_.empty()) {
      return;
    }

    int num_available_pages = fmodel_get_num_available_pages_[0]();
    while (true) {
      if (CanDecode(running_queue_.size())) {
        break;
      }
      StepPreempt(running_queue_.back());
    }
  }

  /*! \brief Assign the given id for the given request. */
  void AssignIDForRequest(Request request, int req_id) {
    // Set id in the request state.
    RequestState& state = request_states_.at(request);
    for (RequestModelState mstate : state->mstates) {
      mstate->request_id = req_id;
    }
    // Add a new sequence to each model.
    for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
      Module model = models_[i];
      int seq_id_in_model = fmodel_add_new_sequence_[i]();
      ICHECK_EQ(seq_id_in_model, req_id);
    }
  }

  /*! \brief Check if the input requests can be prefilled under conditions. */
  bool CanPrefill(int num_prefill_req, int total_input_length, int num_required_pages,
                  int num_available_pages) {
    int num_running_requests = running_queue_.size();
    ICHECK_LE(num_running_requests, kv_cache_config_->max_num_sequence);

    // No exceeding of the maximum allowed requests that can
    // run simultaneously.
    if (num_running_requests + num_prefill_req > kv_cache_config_->max_num_sequence) {
      return false;
    }

    // NOTE: The conditions are heuristic and can be revised.
    // Cond 1: total input length <= max allowed single sequence length.
    // Cond 2: remaining pages >= 10, where 10 is a watermark number can
    // be configured and adjusted in the future.
    // Cond 3: at least one decode can be performed after prefill.
    // Todo: move watermark to config.
    int new_batch_size = num_running_requests + num_prefill_req;
    return total_input_length <= max_single_sequence_length_ &&
           num_required_pages + new_batch_size <= num_available_pages &&
           current_total_seq_len_ + total_input_length + 8 * new_batch_size <=
               kv_cache_config_->max_total_sequence_length;
  }

  /*! \brief Check if the input requests can be decoded under conditions. */
  bool CanDecode(int num_requests) {
    int num_available_pages = fmodel_get_num_available_pages_[0]();
    return num_requests <= num_available_pages;
  }

  /*! \brief Filter the requests to prefill on the given model. */
  std::tuple<Array<Request>, Array<RequestModelState>, ShapeTuple> FilterPrefillRequests(
      Array<Request> requests, Array<RequestState> states, int model_id) {
    ICHECK_EQ(requests.size(), states.size());
    int num_requests = requests.size();
    Array<Request> filtered_requests;
    Array<RequestModelState> filtered_mstates;
    std::vector<int> prefill_length;
    filtered_requests.reserve(num_requests);
    filtered_mstates.reserve(num_requests);
    prefill_length.reserve(num_requests);

    for (int i = 0; i < num_requests; ++i) {
      int length = GetInputLength(states[i]->mstates[model_id]->inputs);
      if (length > 0) {
        filtered_requests.push_back(requests[i]);
        filtered_mstates.push_back(states[i]->mstates[model_id]);
        prefill_length.push_back(length);
      }
    }
    return {filtered_requests, filtered_mstates,
            ShapeTuple(prefill_length.begin(), prefill_length.end())};
  }

  /*! \brief Get the total input length of the given inputs. */
  int GetInputLength(Array<Data> inputs) {
    int length_sum = 0;
    for (Data input : inputs) {
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
      throw;
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
    ShapeTuple token_ids = ftokenizer_tokenize(text);
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
  NDArray GetEmbedding(Data input, PackedFunc fmodel_token_embed) {
    // Dispatch according to input type.
    if (const auto* text_input = input.as<TextDataNode>()) {
      ShapeTuple token_ids = Tokenize(text_input->text);
      return fmodel_token_embed(Array<ShapeTuple>{token_ids});
    } else if (const auto* tokens_input = input.as<TokenDataNode>()) {
      return fmodel_token_embed(Array<ShapeTuple>{tokens_input->token_ids});
    } else {
      ICHECK(false) << "Cannot reach here";
      throw;
    }
  }

  /*!
   * \brief Get token embeddings for all inputs in a **batched style**.
   * It requires all inputs are either TextData or TokenData.
   * This function is usually called for batch-wise actions such as decode
   * (or batched prefill if supported in the future).
   * \param inputs The inputs to compute embeddings.
   * \param fmodel_token_embed The token embedding function of the model of interest.
   * \param return_flattened_view A boolean indicating if flatten the
   * embeddings across the batch dimension or not. For batch decode we
   * do not flatten, and for batch prefill we usually flatten to handle
   * raggedness.
   * \return The computed embeddings with regard to the required view.
   */
  NDArray GetTokenEmbeddings(Array<Data> inputs, PackedFunc fmodel_token_embed,
                             bool return_flattened_view) {
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
    NDArray embeddings = fmodel_token_embed(token_ids);
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
   * \param model_id The id of the LLM model module which contains the softmax
   * function on device that might be helpful.
   * \param sampler_id The id of the sampler module to run sampling.
   * \param request_mstates The request states of each sequence in
   * the batch with regard to the given model.
   * \param generation_cfg The generation config of each request
   * in the input batch.
   * \return The sampled tokens, one for each request in the batch.
   */
  ShapeTuple SampleTokens(NDArray logits_on_device, int model_id, int sampler_id,
                          Array<RequestModelState> request_mstates,
                          Array<GenerationConfig> generation_cfg) {
    ICHECK(logits_on_device.defined());
    ICHECK_EQ(logits_on_device->ndim, 3);
    ICHECK_EQ(logits_on_device->shape[1], 1)
        << "Multi-token sampling for one sequence is not supported yet.";
    ICHECK_EQ(logits_on_device->shape[0], generation_cfg.size());
    ICHECK_EQ(request_mstates.size(), generation_cfg.size());

    Module model = models_[model_id];
    Module sampler = samplers_[sampler_id];

    int num_sequence = logits_on_device->shape[0];
    bool require_gpu_softmax = fsampler_require_gpu_softmax_[sampler_id](generation_cfg);

    // - Compute probabilities from logits.
    NDArray logits_or_probs_on_cpu{nullptr};
    if (require_gpu_softmax) {
      NDArray probs_on_device =
          fmodel_softmax_with_temperature_[model_id](logits_on_device, generation_cfg);
      logits_or_probs_on_cpu = CopyLogitsOrProbsToCPU(probs_on_device);
    } else {
      logits_or_probs_on_cpu = CopyLogitsOrProbsToCPU(logits_on_device);
      // The "compute_probs_from_logits_inplace" function updates
      // `logits_or_probs_on_cpu` in place.
      fsampler_compute_probs_from_logits_inplace_[sampler_id](
          logits_or_probs_on_cpu, std::move(request_mstates), generation_cfg);
    }
    // `CopyLogitsOrProbsToCPU` flattens the first two dimensions.
    ICHECK_EQ(logits_or_probs_on_cpu->ndim, 2);

    // - Sample tokens from probabilities.
    // NOTE: Though we have the probability field in RequestModelState,
    //       we do not save the probabilities right now.
    //       We will handle this in the future when we work on speculation.
    ShapeTuple new_tokens =
        fsampler_sample_token_from_probs_[sampler_id](logits_or_probs_on_cpu, generation_cfg);
    return new_tokens;
  }

  /*!
   * \brief Copy logits or prob distributions from device to CPU.
   * The input array is in layout (b, n, v).
   * This function flattens the first dimension, returns an NDArray
   * in shape (b * n, v).
   */
  NDArray CopyLogitsOrProbsToCPU(NDArray arr_on_device) {
    // arr_on_device: (b, n, v)
    ICHECK_EQ(arr_on_device->ndim, 3);
    ICHECK(!logits_or_probs_on_cpu_.defined() || (logits_or_probs_on_cpu_)->ndim == 2);
    ICHECK(arr_on_device->device.device_type != kDLCPU);
    if (logits_or_probs_on_cpu_.defined()) {
      ICHECK_EQ(logits_or_probs_on_cpu_->shape[1], arr_on_device->shape[2]);
    }

    int64_t init_size = logits_or_probs_on_cpu_.defined() ? logits_or_probs_on_cpu_->shape[0] : 32;
    int64_t num_tokens = arr_on_device->shape[0] * arr_on_device->shape[1];
    int64_t vocab_size = arr_on_device->shape[2];
    while (init_size < num_tokens) {
      init_size *= 2;
    }
    if (!logits_or_probs_on_cpu_.defined() || init_size != logits_or_probs_on_cpu_->shape[0]) {
      logits_or_probs_on_cpu_ =
          NDArray::Empty({init_size, vocab_size}, arr_on_device->dtype, DLDevice{kDLCPU, 0});
    }
    ICHECK_LE(num_tokens, logits_or_probs_on_cpu_->shape[0]);
    NDArray view =
        logits_or_probs_on_cpu_.CreateView({num_tokens, vocab_size}, arr_on_device->dtype);
    view.CopyFrom(arr_on_device);
    return view;
  }

  /*! \brief Remove the given request from all models (usually the KV cache). */
  void RemoveSequenceFromModels(int req_id) {
    for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
      fmodel_remove_sequence_[i](req_id);
    }
  }

  /*!
   * \brief Update the request ids of all running requests after
   * the removal of the given request.
   */
  void UpdateRequestIDAfterRemoval(int removed_req_id) {
    for (auto& it : request_states_) {
      RequestState& state = it.second;
      for (RequestModelState mstate : state->mstates) {
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
      int num_input_tokens = state->raw_input_length;
      int num_output_tokens = state->mstates[0]->committed_tokens.size() - 1;
      current_total_seq_len_ -= num_input_tokens + num_output_tokens;
      for (RequestModelState mstate : state->mstates) {
        ICHECK_EQ(mstate->request_id, req_id);
        mstate->request_id = -1;
      }
      RemoveSequenceFromModels(req_id);
      UpdateRequestIDAfterRemoval(req_id);

      auto trequest_finish = std::chrono::high_resolution_clock::now();
      request_total_prefill_time_ +=
          static_cast<double>((state->tprefill_finish - state->tadd).count()) / 1e9;
      total_prefill_length_ += num_input_tokens;
      request_total_decode_time_ +=
          static_cast<double>((trequest_finish - state->tprefill_finish).count()) / 1e9;
      total_decode_length_ += num_output_tokens;

      // NOTE: right now we only return the generated text.
      // In the future we might optional return text or token ids.
      String output = ftokenizer_decode(ShapeTuple(state->mstates[0]->committed_tokens.begin(),
                                                   state->mstates[0]->committed_tokens.end()));
      state->output = output.operator std::string();
      request->fcallback(request, TextData(state->output));

      // Remove the request from states.
      request_states_.erase(request);
    }
  }

  /*! \brief Check if the given request is finished under conditions. */
  bool RequestIsFinished(Request request) {
    RequestState& state = request_states_.at(request);

    // - Case 0. There is remaining draft output ==> Unfinished
    //   All draft outputs are supposed to be processed before finish.
    for (RequestModelState mstate : state->mstates) {
      if (!mstate->draft_output_tokens.empty()) {
        return false;
      }
    }

    // - Decode committed tokens.
    const std::vector<int32_t>& committed_tokens = state->mstates[0]->committed_tokens;

    // Case 1. Any of the stop strings appears in output ==> Finished
    // Todo: handle stop_str by tokenizing. So that we don't detokenize during check

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
    if (state->raw_input_length + static_cast<int>(committed_tokens.size()) >=
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

  /*! \brief Shared array for logits and probability distributions on cpu. */
  NDArray logits_or_probs_on_cpu_{nullptr};

  // PackedFuncs from model/tokenizer/sampler/env.
  std::vector<PackedFunc> fmodel_batch_prefill_;
  std::vector<PackedFunc> fmodel_decode_;
  std::vector<PackedFunc> fmodel_token_embed_;
  std::vector<PackedFunc> fmodel_add_new_sequence_;
  std::vector<PackedFunc> fmodel_remove_sequence_;
  std::vector<PackedFunc> fmodel_softmax_with_temperature_;
  std::vector<PackedFunc> fmodel_get_num_available_pages_;
  std::vector<PackedFunc> fsampler_require_gpu_softmax_;
  std::vector<PackedFunc> fsampler_compute_probs_from_logits_inplace_;
  std::vector<PackedFunc> fsampler_sample_token_from_probs_;
  PackedFunc ftokenizer_tokenize;
  PackedFunc ftokenizer_decode;

  // Runtime statistics
  int64_t current_total_seq_len_;
  /*! \brief The sum of "prefill time of each request". */
  double request_total_prefill_time_ = 0.0f;
  /*! \brief The sum of "decode time of each request". */
  double request_total_decode_time_ = 0.0f;
  /*! \brief The total engine time on prefill. */
  double engine_total_prefill_time_ = 0.0f;
  /*! \brief The total engine time on decode. */
  double engine_total_decode_time_ = 0.0f;
  /*! \brief The total number of processed tokens in prefill. */
  int64_t total_prefill_length_ = 0;
  /*! \brief The total number of processed tokens in decode. */
  int64_t total_decode_length_ = 0;

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
        *rv = GetEngine()->StatisticsJSON();
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
