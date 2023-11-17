/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.cc
 * \brief The implementation for runtime module of serving engine module in MLC LLM.
 */
#define __STDC_FORMAT_MACROS

#include <tokenizers_cpp.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../tokenizers.h"
#include "engine_stats.h"
#include "model.h"
#include "request.h"
#include "request_state.h"
#include "sampler.h"

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
 *
 * The public interface of Engine has the following three categories:
 * - engine management,
 * - high-level request management,
 * - engine "step" action.
 *
 * The internal implementation of Engine has the following categories:
 * - internal request management,
 * - actions and request schedule policy (such as prefill, decode, etc.)
 */
class Engine {
  friend class EngineModule;

 public:
  /********************** Engine Management **********************/

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

    // Step 1. Create models and their PackedFuncs.
    ICHECK(models_.empty());
    models_.reserve(num_models);
    for (int i = 0; i < num_models; ++i) {
      models_.push_back(Model::Create(reload_libs[i], model_paths[i], devices[i]));
    }
    // Step 2. Fetch max single sequence length from models.
    max_single_sequence_length_ = std::numeric_limits<int>::max();
    for (Model model : models_) {
      int max_window_size = model->GetMaxWindowSize();
      max_single_sequence_length_ = std::min(max_single_sequence_length_, max_window_size);
    }
    // Step 3. Process KV cache config json string.
    kv_cache_config_ = KVCacheConfig(kv_cache_config_json, max_single_sequence_length_);
    // Step 4. Create KV cache for each model.
    for (Model model : models_) {
      model->CreateKVCache(kv_cache_config_);
    }
    // Step 5. Create sampler and tokenizer.
    //         The tokenizer is created from the first model.
    //         We assume all models have the same tokenizer, which is the basic
    //         requirement of speculative encoding.
    sampler_ = Sampler::Create(/*sampler_kind=*/"cpu");
    tokenizer_ = TokenizerFromPath(model_paths[0]);

    ResetEngine();
  }

  /*! \brief Reset the engine, clean up all running data and statistics. */
  void ResetEngine() {
    running_queue_.clear();
    waiting_queue_.clear();
    abort_queue_.clear();
    request_states_.clear();
    stats_.Reset();
    for (Model model : models_) {
      model->Reset();
    }
  }

  /***************** High-level Request Management *****************/

  /*!
   * \brief Add a new request to the engine.
   * \param request The request to add.
   */
  void AddRequest(Request request) {
    // Get a request copy where all text inputs are tokenized.
    request = Request::FromUntokenized(request, tokenizer_);
    ICHECK_NE(request->input_total_length, -1);
    // Append to the waiting queue and create the request state.
    waiting_queue_.push_back(request);
    request_states_.emplace(request->id, RequestState(request, models_.size()));
  }

  /*! \brief Abort the input request. */
  void AbortRequest(Request request) { abort_queue_.push_back(request); }

  /*********************** Engine Action ***********************/

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

    // - Action 2. Run decode step.
    bool decode_processed = StepDecode();
    if (decode_processed) {
      ProcessFinishedRequest();
      return;
    }

    ICHECK(running_queue_.empty())
        << "Not taking any action in a step is not expected with running requests.";
  }

 private:
  /***************** Internal Request Management *****************/

  /*! \brief Assign the given internal id for the given request. */
  void AssignIDForRequest(Request request, int req_id) {
    // Set internal id in the request state.
    RequestState state = request_states_.at(request->id);
    for (RequestModelState mstate : state->mstates) {
      mstate->request_id = req_id;
    }
    // Add a new sequence to each model.
    for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
      int seq_id_in_model = models_[i]->AddNewSequence();
      ICHECK_EQ(seq_id_in_model, req_id);
    }
  }

  /*!
   * \brief Remove the given request from models and update request states.
   * \param req_id The internal id of the request to remove.
   */
  void RemoveRequestFromModel(int req_id) {
    // Remove the request from all models (usually the KV cache).
    for (Model model : models_) {
      model->RemoveSequence(req_id);
    }
    // Update the internal request id of other requests.
    for (auto& it : request_states_) {
      RequestState state = it.second;
      for (RequestModelState mstate : state->mstates) {
        ICHECK_NE(mstate->request_id, req_id);
        if (mstate->request_id > req_id) {
          --mstate->request_id;
        }
      }
    }
  }

  /*!
   * \brief Preempt the generation of the given request, moving
   * it from running request set to the foremost of waiting
   * request queue.
   */
  void PreemptRequest(std::vector<Request>::iterator request_it) {
    Request request = *request_it;

    // Remove from models.
    // - Reset `request_id` of states.
    // - Clear model speculation draft.
    // - Update `inputs` for future prefill.
    RequestState state = request_states_.at(request->id);
    int req_id = state->mstates[0]->request_id;
    stats_.current_total_seq_len -=
        request->input_total_length + state->mstates[0]->committed_tokens.size() - 1;
    for (RequestModelState mstate : state->mstates) {
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
    RemoveRequestFromModel(req_id);

    // Move from running queue to the front of waiting queue.
    running_queue_.erase(request_it);
    waiting_queue_.insert(waiting_queue_.begin(), request);
  }

  /*!
   * \brief For each request, check if the request has finished
   * its generation. And update the state and return the generation
   * result for the finished requests.
   * \note This function removes requests from the running request
   * queue.
   */
  void ProcessFinishedRequest() {
    // - Collect finished requests.
    //   We don't remove on the fly to avoid concurrent modification.
    std::vector<Request> request_to_remove;
    for (Request request : running_queue_) {
      if (request_states_.at(request->id)->GenerationFinished(max_single_sequence_length_)) {
        request_to_remove.push_back(request);
      }
    }

    // - Remove the finished request.
    for (Request request : request_to_remove) {
      // Remove from running queue.
      auto it = std::find(running_queue_.begin(), running_queue_.end(), request);
      ICHECK(it != running_queue_.end());
      running_queue_.erase(it);

      // Update engine states.
      RequestState state = request_states_.at(request->id);
      int req_id = state->mstates[0]->request_id;
      for (RequestModelState mstate : state->mstates) {
        ICHECK_EQ(mstate->request_id, req_id);
        mstate->request_id = -1;
      }
      RemoveRequestFromModel(req_id);
      request_states_.erase(request->id);

      // Update engine statistics.
      int num_input_tokens = request->input_total_length;
      int num_output_tokens = state->mstates[0]->committed_tokens.size() - 1;
      stats_.current_total_seq_len -= num_input_tokens + num_output_tokens;
      auto trequest_finish = std::chrono::high_resolution_clock::now();
      stats_.request_total_prefill_time +=
          static_cast<double>((state->tprefill_finish - state->tadd).count()) / 1e9;
      stats_.total_prefill_length += num_input_tokens;
      stats_.request_total_decode_time +=
          static_cast<double>((trequest_finish - state->tprefill_finish).count()) / 1e9;
      stats_.total_decode_length += num_output_tokens;

      // NOTE: right now we only return the generated text.
      // In the future we might optional return text or token ids.
      String output = tokenizer_->Decode(state->mstates[0]->committed_tokens);
      request->fcallback(request, TextData(output));
    }
  }

  /************** Engine Actions and Request Schedule Policy **************/

  /*********************
   * Action 1. Prefill *
   *********************/

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
      Model model = models_[model_id];
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
          stats_.current_total_seq_len += prefill_length;
          mstates_for_sample.push_back(mstate);
          generation_cfg_for_sample.push_back(request->generation_cfg);
        }
        ICHECK(mstate->draft_output_tokens.empty());
        ICHECK(mstate->draft_output_token_prob.empty());
        ICHECK(mstate->draft_output_prob_dist.empty());
        ICHECK(!mstate->inputs.empty());
        request_ids.push_back(mstate->request_id);
        for (int i = 0; i < static_cast<int>(mstate->inputs.size()); ++i) {
          embeddings.push_back(mstate->inputs[i]->GetEmbedding(model));
        }
        // Clean up `inputs` after prefill
        mstate->inputs.clear();
      }

      NDArray logits = model->BatchPrefill(embeddings, request_ids, prefill_lengths);
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
      std::vector<int32_t> next_tokens = sampler_->SampleTokens(
          logits_for_sample, models_[0], mstates_for_sample, generation_cfg_for_sample);
      ICHECK_EQ(next_tokens.size(), num_requests);
      // - Update the committed tokens of states.
      // - If a request is first-time prefilled, set the prefill finish time.
      auto tnow = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < num_requests; ++i) {
        mstates_for_sample[i]->committed_tokens.push_back(next_tokens[i]);
        if (mstates_for_sample[i]->committed_tokens.size() == 1) {
          request_states_.at(requests[i]->id)->tprefill_finish = tnow;
        }
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    stats_.engine_total_prefill_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return true;
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
      int num_available_pages = models_[0]->GetNumAvailablePages();

      for (int i = 0; i < static_cast<int>(waiting_queue_.size()); ++i) {
        Request request = waiting_queue_[i];
        RequestState state = request_states_.at(request->id);
        int input_length = state->mstates[0]->GetInputLength();
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
      RequestState state = request_states_.at(request->id);
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
    // Cond 2: at least one decode can be performed after prefill.
    // Cond 3: number of total tokens after 8 times of decode does not
    // exceed the limit, where 8 is a watermark number can
    // be configured and adjusted in the future.
    int new_batch_size = num_running_requests + num_prefill_req;
    return total_input_length <= max_single_sequence_length_ &&
           num_required_pages + new_batch_size <= num_available_pages &&
           stats_.current_total_seq_len + total_input_length + 8 * new_batch_size <=
               kv_cache_config_->max_total_sequence_length;
  }

  /*! \brief Filter the requests to prefill on the given model. */
  std::tuple<Array<Request>, Array<RequestModelState>, std::vector<int>> FilterPrefillRequests(
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
      int length = states[i]->mstates[model_id]->GetInputLength();
      if (length > 0) {
        filtered_requests.push_back(requests[i]);
        filtered_mstates.push_back(states[i]->mstates[model_id]);
        prefill_length.push_back(length);
      }
    }
    return {filtered_requests, filtered_mstates, prefill_length};
  }

  /********************
   * Action 2. Decode *
   ********************/

  /*! \brief Pick applicable requests and run decode. */
  bool StepDecode() {
    // - Do not run decode when there are multiple models.
    if (models_.size() > 1) {
      return false;
    }

    if (running_queue_.empty()) {
      return false;
    }

    // Preempt requests when decode cannot apply.
    while (!CanDecode(running_queue_.size())) {
      PreemptRequest(running_queue_.end() - 1);
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running requests at a time.
    int num_requests = running_queue_.size();
    // Check if the requests ids are in an ascending order.
    for (int i = 1; i < num_requests; ++i) {
      ICHECK_GT(request_states_.at(running_queue_[i]->id)->mstates[0]->request_id,
                request_states_.at(running_queue_[i - 1]->id)->mstates[0]->request_id);
    }

    stats_.current_total_seq_len += num_requests;
    // Collect
    // - the last committed token,
    // - the request states,
    // - the sampling parameters,
    // of each request.
    std::vector<int> input_tokens;
    Array<RequestModelState> mstates;
    Array<GenerationConfig> generation_cfg;
    input_tokens.reserve(num_requests);
    mstates.reserve(num_requests);
    generation_cfg.reserve(num_requests);
    for (Request request : running_queue_) {
      RequestState state = request_states_.at(request->id);
      input_tokens.push_back(state->mstates[0]->committed_tokens.back());
      mstates.push_back(state->mstates[0]);
      generation_cfg.push_back(request->generation_cfg);
    }

    // - Compute embeddings.
    NDArray embeddings =
        models_[0]->TokenEmbed({IntTuple{input_tokens.begin(), input_tokens.end()}});
    ICHECK_EQ(embeddings->ndim, 3);
    ICHECK_EQ(embeddings->shape[0], 1);
    ICHECK_EQ(embeddings->shape[1], num_requests);
    embeddings = embeddings.CreateView({num_requests, 1, embeddings->shape[2]}, embeddings->dtype);

    // - Invoke model decode.
    NDArray logits = models_[0]->BatchDecode(embeddings);
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
    ICHECK_EQ(logits->shape[1], 1);

    // - Sample tokens.
    std::vector<int32_t> next_tokens =
        sampler_->SampleTokens(logits, models_[0], mstates, generation_cfg);
    ICHECK_EQ(next_tokens.size(), num_requests);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_requests; ++i) {
      mstates[i]->committed_tokens.push_back(next_tokens[i]);
    }

    auto tend = std::chrono::high_resolution_clock::now();
    stats_.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return true;
  }

  /*! \brief Check if the input requests can be decoded under conditions. */
  bool CanDecode(int num_requests) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_requests <= num_available_pages;
  }

  /*******************
   * Action 3. Abort *
   *******************/

  /*! \brief Abort the generation of the given request. */
  void StepAbort(Request request) {
    auto it_running = std::find(running_queue_.begin(), running_queue_.end(), request);
    auto it_waiting = std::find(waiting_queue_.begin(), waiting_queue_.end(), request);
    ICHECK(it_running != running_queue_.end() || it_waiting != waiting_queue_.end());
    if (it_running != running_queue_.end()) {
      // The request to abort is in running queue
      int req_id = it_running - running_queue_.begin();
      running_queue_.erase(it_running);
      RequestState state = request_states_.at(request->id);
      stats_.current_total_seq_len -=
          request->input_total_length + state->mstates[0]->committed_tokens.size() - 1;
      RemoveRequestFromModel(req_id);
    } else {
      // The request to abort is in waiting queue
      waiting_queue_.erase(it_waiting);
    }
    request_states_.erase(request->id);
  }

  /***************** Engine Data Structures *****************/

  // Request queues
  std::vector<Request> running_queue_;
  std::vector<Request> waiting_queue_;
  std::vector<Request> abort_queue_;
  // Request states
  std::unordered_map<String, RequestState, ObjectPtrHash, ObjectPtrEqual> request_states_;

  // Models, sampler and tokenizer.
  Array<Model> models_;
  Sampler sampler_;
  std::unique_ptr<Tokenizer> tokenizer_;

  // Runtime statistics
  EngineStats stats_;

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
        *rv = GetEngine()->stats_.AsJSON();
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
TVM_REGISTER_GLOBAL("mlc.serve.create_engine").set_body_typed(CreateEngineModule);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
