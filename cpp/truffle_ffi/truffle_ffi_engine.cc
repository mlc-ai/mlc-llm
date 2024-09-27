#include "truffle_ffi_engine.h"

#include <picojson.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <filesystem>
#include <fstream>

#include "../serve/model.h"
#include "../support/json_parser.h"
#include "../support/result.h"
#include <cstdio>

namespace mlc {
namespace llm {
namespace truffle_ffi {

using namespace tvm::runtime;

TruffleFFIEngine::TruffleFFIEngine() { engine_ = serve::ThreadedEngine::Create(); }

bool TruffleFFIEngine::Completion(std::string request_json_str, std::string request_id) {
  bool success = this->AddRequest(request_json_str, request_id);
  if (!success) {
    this->StreamBackError(request_id);
  }
  return success;
}

void TruffleFFIEngine::StreamBackError(std::string request_id) {
  TruffleResponse response;
  response.content = this->err_;
 
  response.finish_reason = FinishReason::error;
  picojson::array response_arr;
  response.id = request_id;
  picojson::object dummy_usage;
  dummy_usage["prompt_tokens"] = picojson::value(static_cast<int64_t>(0));
  dummy_usage["completion_tokens"] = picojson::value(static_cast<int64_t>(0));
  dummy_usage["total_tokens"] = picojson::value(static_cast<int64_t>(0));
  response.usage = picojson::value(dummy_usage);
  response_arr.push_back(picojson::value(response.AsJSON()));

  std::string stream_back_json = picojson::value(response_arr).serialize();
  this->request_stream_callback_(stream_back_json);
}

bool TruffleFFIEngine::AddRequest(std::string request_json_str, std::string request_id) {
  Result<TruffleRequest> request_res = TruffleRequest::FromJSON(request_json_str);
  if (request_res.IsErr()) {
    err_ = request_res.UnwrapErr();
    return false;
  }
  TruffleRequest request = request_res.Unwrap();
  Array<Data> inputs;
  Array<String> stop_strs;
  

    inputs.push_back(TextData(request.context));

    stop_strs.push_back("<|eot_id|>");
    if (request.stop.has_value()) {
      stop_strs.reserve(stop_strs.size() + request.stop.value().size());
      for (const std::string& stop_str : request.stop.value()) {
        stop_strs.push_back(stop_str);
      }
    }
  
  // create a generation config from request
  const auto& default_gen_cfg = default_generation_config_;
  auto gen_cfg = tvm::runtime::make_object<GenerationConfigNode>();
  gen_cfg->n = 1;
  gen_cfg->temperature = request.temperature.value_or(default_gen_cfg->temperature);
  gen_cfg->top_p = request.top_p.value_or(default_gen_cfg->top_p);
  gen_cfg->frequency_penalty =
      request.frequency_penalty.value_or(default_gen_cfg->frequency_penalty);
  gen_cfg->presence_penalty = request.presence_penalty.value_or(default_gen_cfg->presence_penalty);
 // gen_cfg->logprobs = request.logprobs;
  //gen_cfg->top_logprobs = request.top_logprobs;
  gen_cfg->logit_bias = default_gen_cfg->logit_bias;
  gen_cfg->seed = std::random_device{}();
  gen_cfg->max_tokens = request.max_tokens.value_or(default_gen_cfg->max_tokens);
  gen_cfg->stop_strs = std::move(stop_strs);
  gen_cfg->stop_token_ids = std::vector<int>{};
  gen_cfg->response_format = ResponseFormat();
  gen_cfg->debug_config = DebugConfig();

  Result<GenerationConfig> res_gen_config = GenerationConfig::Validate(GenerationConfig(gen_cfg));
  if (res_gen_config.IsErr()) {
    err_ = res_gen_config.UnwrapErr();
    return false;
  }

  Request engine_request(request_id, inputs, res_gen_config.Unwrap());

  // setup request state
  RequestState rstate;
  rstate.model = ""; 
  rstate.streamer.reserve(gen_cfg->n);
  for (int i = 0; i < gen_cfg->n; ++i) {
    rstate.streamer.push_back(TextStreamer(tokenizer_));
  }
  request_map_[request_id] = std::move(rstate);

  this->engine_->AddRequest(engine_request);
  return true;
}

bool TruffleFFIEngine::Abort(std::string request_id) {
  this->engine_->AbortRequest(request_id);
  auto it = request_map_.find(request_id);
  if (it != request_map_.end()) {
    request_map_.erase(it);
  }
  return true;
}

std::string TruffleFFIEngine::GetLastError() { return err_; }

void TruffleFFIEngine::ExitBackgroundLoop() { this->engine_->ExitBackgroundLoop(); }

TruffleFFIEngine::~TruffleFFIEngine() { this->ExitBackgroundLoop(); }

class TruffleFFIEngineImpl : public TruffleFFIEngine, public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.truffle_ffi");
  TVM_MODULE_VTABLE_ENTRY("init_background_engine", &TruffleFFIEngineImpl::InitBackgroundEngine);
  TVM_MODULE_VTABLE_ENTRY("reload", &TruffleFFIEngineImpl::Reload);
  TVM_MODULE_VTABLE_ENTRY("unload", &TruffleFFIEngineImpl::Unload);
  TVM_MODULE_VTABLE_ENTRY("reset", &TruffleFFIEngineImpl::Reset);
  TVM_MODULE_VTABLE_ENTRY("completion", &TruffleFFIEngineImpl::Completion);
  TVM_MODULE_VTABLE_ENTRY("abort", &TruffleFFIEngineImpl::Abort);
  TVM_MODULE_VTABLE_ENTRY("get_last_error", &TruffleFFIEngineImpl::GetLastError);
  TVM_MODULE_VTABLE_ENTRY("run_background_loop", &TruffleFFIEngineImpl::RunBackgroundLoop);
  TVM_MODULE_VTABLE_ENTRY("run_background_stream_back_loop",
                          &TruffleFFIEngineImpl::RunBackgroundStreamBackLoop);
  TVM_MODULE_VTABLE_ENTRY("exit_background_loop", &TruffleFFIEngineImpl::ExitBackgroundLoop);
  TVM_MODULE_VTABLE_END();

  void InitBackgroundEngine(int device_type, int device_id,
                            Optional<PackedFunc> request_stream_callback) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    this->device_ = device;
    CHECK(request_stream_callback.defined())
        << "TruffleFFIEngine requires request stream callback function, but it is not given.";
    this->request_stream_callback_ = request_stream_callback.value();

    auto frequest_stream_callback_wrapper = [this](TVMArgs args, TVMRetValue* ret) {
      ICHECK_EQ(args.size(), 1);
      Array<RequestStreamOutput> delta_outputs = args[0];
      std::string responses = this->GetResponseFromStreamOutput(delta_outputs);
      this->request_stream_callback_(responses);
    };

    request_stream_callback = PackedFunc(frequest_stream_callback_wrapper);
    this->engine_->InitThreadedEngine(device, std::move(request_stream_callback), NullOpt);
  }

  void Reload(String engine_config_json_str) {
    this->engine_->Reload(engine_config_json_str);
    this->default_generation_config_ = this->engine_->GetDefaultGenerationConfig();
    auto engine_config = this->engine_->GetCompleteEngineConfig();

    // Load conversation template.
    Result<picojson::object> model_config_json =
        serve::Model::LoadModelConfig(engine_config->model);
    CHECK(model_config_json.IsOk()) << model_config_json.UnwrapErr();
    const picojson::object& model_config_json_unwrapped = model_config_json.Unwrap();
   
    this->model_config_ = ModelConfig::FromJSON(
        json::Lookup<picojson::object>(model_config_json_unwrapped, "model_config"));
    this->tokenizer_ = Tokenizer::FromPath(engine_config->model);
  }

  void Unload() { this->engine_->Unload(); }

  void Reset() { this->engine_->Reset(); }

  void RunBackgroundLoop() { this->engine_->RunBackgroundLoop(); }

  void RunBackgroundStreamBackLoop() { this->engine_->RunBackgroundStreamBackLoop(); }

  String GetResponseFromStreamOutput(Array<RequestStreamOutput> delta_outputs) {
    picojson::array json_response_arr;
    for (const auto& delta_output : delta_outputs) {
      std::string request_id = delta_output->request_id;
      auto request_state_it = request_map_.find(request_id);
      if (request_state_it == request_map_.end()) continue;
      RequestState& rstate = request_state_it->second;

      // build the final usage messages
      // invariant, we can always let other messages to come first
      // then the final usage messages, as final usage is always last
      if (delta_output->request_final_usage_json_str.defined()) {
        TruffleResponse response;
        response.id = request_id;
  
        std::string usage_json_str = delta_output->request_final_usage_json_str.value();
        picojson::value usage_json;
        std::string err = picojson::parse(usage_json, usage_json_str);
        if (!err.empty()) {
          err_ = err;
        } else {
          response.usage = usage_json;
        }
        json_response_arr.push_back(picojson::value(response.AsJSON()));
        request_map_.erase(request_state_it);
        continue;
      }
      ICHECK_NE(delta_output->group_finish_reason.size(), 0);
      ICHECK_EQ(delta_output->group_delta_token_ids.size(),
                delta_output->group_finish_reason.size());
      ICHECK_EQ(delta_output->group_delta_token_ids.size(), rstate.streamer.size());

      TruffleResponse response;
      response.id = request_id;
     

      for (size_t i = 0; i < delta_output->group_finish_reason.size(); ++i) {
        // choice
        
        Optional<String> finish_reason = delta_output->group_finish_reason[i];
        if (finish_reason.defined()) {
          if (finish_reason.value() == "stop") {
            response.finish_reason = FinishReason::stop;
          } else if (finish_reason.value() == "length") {
            response.finish_reason = FinishReason::length;
          } else if (finish_reason.value() == "tool_calls") {
            response.finish_reason = FinishReason::tool_calls;
          } else if (finish_reason.value() == "error") {
            response.finish_reason = FinishReason::error;
          }
        } else {
          response.finish_reason = std::nullopt;
        }
       
        std::string delta;
        // Size of delta_output->group_delta_token_ids Array should be 1
        const IntTuple& delta_token_ids = delta_output->group_delta_token_ids[i];
        std::vector<int32_t> delta_token_ids_vec(delta_token_ids.begin(), delta_token_ids.end());
        std::string content = rstate.streamer[i]->Put(delta_token_ids_vec);
       
        if (finish_reason.defined()) {
          content += rstate.streamer[i]->Finish();
        }
        if (!content.empty()) {
          delta = content;
          
        }
      
        if (!delta.empty() || response.finish_reason.has_value()) {
          response.content.append(delta);
          
        }
      }
      // if it is not the usage block, choices cannot be empty
      if (!response.content.empty()) {
        json_response_arr.push_back(picojson::value(response.AsJSON()));
      }
    }
    return picojson::value(json_response_arr).serialize();
  }
};

TVM_REGISTER_GLOBAL("mlc.truffle_ffi.CreateTruffleFFIEngine").set_body_typed([]() {
  return Module(make_object<TruffleFFIEngineImpl>());
});

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
