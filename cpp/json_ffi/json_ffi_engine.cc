#include "json_ffi_engine.h"

#include <picojson.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <filesystem>
#include <fstream>

#include "../serve/model.h"
#include "../support/json_parser.h"
#include "../support/result.h"

namespace mlc {
namespace llm {
namespace json_ffi {

using namespace tvm::runtime;

JSONFFIEngine::JSONFFIEngine() { engine_ = serve::ThreadedEngine::Create(); }

bool JSONFFIEngine::ChatCompletion(std::string request_json_str, std::string request_id) {
  bool success = this->AddRequest(request_json_str, request_id);
  if (!success) {
    this->StreamBackError(request_id);
  }
  return success;
}

void JSONFFIEngine::StreamBackError(std::string request_id) {
  ChatCompletionMessage delta;
  delta.content = std::vector<std::unordered_map<std::string, std::string>>{
      {{"type", "text"}, {"text", this->err_}}};
  delta.role = "assistant";

  ChatCompletionStreamResponseChoice choice;
  choice.finish_reason = FinishReason::error;
  choice.index = 0;
  choice.delta = delta;

  ChatCompletionStreamResponse response;
  response.id = request_id;
  response.choices = std::vector<ChatCompletionStreamResponseChoice>{choice};
  response.model = "json_ffi";  // TODO: Return model name from engine (or from args)
  response.system_fingerprint = "";

  picojson::array response_arr;
  response_arr.push_back(picojson::value(response.AsJSON()));
  std::string stream_back_json = picojson::value(response_arr).serialize();
  this->request_stream_callback_(stream_back_json);
}

bool JSONFFIEngine::AddRequest(std::string request_json_str, std::string request_id) {
  Result<ChatCompletionRequest> request_res = ChatCompletionRequest::FromJSON(request_json_str);
  if (request_res.IsErr()) {
    err_ = request_res.UnwrapErr();
    return false;
  }
  ChatCompletionRequest request = request_res.Unwrap();
  // get prompt: note, assistant was appended in the end.
  Result<std::vector<Data>> inputs_obj =
      CreatePrompt(this->conv_template_, request, this->model_config_, this->device_);
  if (inputs_obj.IsErr()) {
    err_ = inputs_obj.UnwrapErr();
    return false;
  }
  Array<Data> inputs = inputs_obj.Unwrap();

  // generation_cfg
  Array<String> stop_strs;
  stop_strs.reserve(this->conv_template_.stop_str.size());
  for (const std::string& stop_str : this->conv_template_.stop_str) {
    stop_strs.push_back(stop_str);
  }
  if (request.stop.has_value()) {
    stop_strs.reserve(stop_strs.size() + request.stop.value().size());
    for (const std::string& stop_str : request.stop.value()) {
      stop_strs.push_back(stop_str);
    }
  }

  // create a generation config from request
  const auto& default_gen_cfg = default_generation_config_;
  auto gen_cfg = tvm::runtime::make_object<GenerationConfigNode>();
  gen_cfg->n = request.n;
  gen_cfg->temperature = request.temperature.value_or(default_gen_cfg->temperature);
  gen_cfg->top_p = request.top_p.value_or(default_gen_cfg->top_p);
  gen_cfg->frequency_penalty =
      request.frequency_penalty.value_or(default_gen_cfg->frequency_penalty);
  gen_cfg->presence_penalty = request.presence_penalty.value_or(default_gen_cfg->presence_penalty);
  gen_cfg->logprobs = request.logprobs;
  gen_cfg->top_logprobs = request.top_logprobs;
  gen_cfg->logit_bias = request.logit_bias.value_or(default_gen_cfg->logit_bias);
  gen_cfg->seed = request.seed.value_or(default_gen_cfg->seed);
  gen_cfg->max_tokens = request.seed.value_or(default_gen_cfg->max_tokens);
  gen_cfg->stop_strs = std::move(stop_strs);
  gen_cfg->stop_token_ids = conv_template_.stop_token_ids;
  gen_cfg->debug_config = request.debug_config.value_or(DebugConfig());

  Result<GenerationConfig> res_gen_config = GenerationConfig::Validate(GenerationConfig(gen_cfg));
  if (res_gen_config.IsErr()) {
    err_ = res_gen_config.UnwrapErr();
    return false;
  }

  Request engine_request(request_id, inputs, res_gen_config.Unwrap());
  this->engine_->AddRequest(engine_request);
  return true;
}

bool JSONFFIEngine::Abort(std::string request_id) {
  this->engine_->AbortRequest(request_id);
  return true;
}

std::string JSONFFIEngine::GetLastError() { return err_; }

std::string JSONFFIEngine::JSONMetrics() { return this->engine_->JSONMetrics(); }

void JSONFFIEngine::ExitBackgroundLoop() { this->engine_->ExitBackgroundLoop(); }

JSONFFIEngine::~JSONFFIEngine() { this->ExitBackgroundLoop(); }

class JSONFFIEngineImpl : public JSONFFIEngine, public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.json_ffi");
  TVM_MODULE_VTABLE_ENTRY("init_background_engine", &JSONFFIEngineImpl::InitBackgroundEngine);
  TVM_MODULE_VTABLE_ENTRY("reload", &JSONFFIEngineImpl::Reload);
  TVM_MODULE_VTABLE_ENTRY("unload", &JSONFFIEngineImpl::Unload);
  TVM_MODULE_VTABLE_ENTRY("reset", &JSONFFIEngineImpl::Reset);
  TVM_MODULE_VTABLE_ENTRY("chat_completion", &JSONFFIEngineImpl::ChatCompletion);
  TVM_MODULE_VTABLE_ENTRY("abort", &JSONFFIEngineImpl::Abort);
  TVM_MODULE_VTABLE_ENTRY("get_last_error", &JSONFFIEngineImpl::GetLastError);
  TVM_MODULE_VTABLE_ENTRY("json_metrics", &JSONFFIEngineImpl::JSONMetrics);
  TVM_MODULE_VTABLE_ENTRY("run_background_loop", &JSONFFIEngineImpl::RunBackgroundLoop);
  TVM_MODULE_VTABLE_ENTRY("run_background_stream_back_loop",
                          &JSONFFIEngineImpl::RunBackgroundStreamBackLoop);
  TVM_MODULE_VTABLE_ENTRY("exit_background_loop", &JSONFFIEngineImpl::ExitBackgroundLoop);
  TVM_MODULE_VTABLE_END();

  void InitBackgroundEngine(int device_type, int device_id,
                            Optional<PackedFunc> request_stream_callback) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    this->device_ = device;
    CHECK(request_stream_callback.defined())
        << "JSONFFIEngine requires request stream callback function, but it is not given.";
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
    Result<Conversation> conv_template = Conversation::FromJSON(
        json::Lookup<picojson::object>(model_config_json_unwrapped, "conv_template"));
    CHECK(!conv_template.IsErr()) << "Invalid conversation template JSON: "
                                  << conv_template.UnwrapErr();
    this->conv_template_ = conv_template.Unwrap();
    this->model_config_ = ModelConfig::FromJSON(
        json::Lookup<picojson::object>(model_config_json_unwrapped, "model_config"));

    // Create streamer.
    // Todo(mlc-team): Create one streamer for each request, instead of a global one.
    this->streamer_ = TextStreamer(Tokenizer::FromPath(engine_config->model));
  }

  void Unload() { this->engine_->Unload(); }

  void Reset() { this->engine_->Reset(); }

  void RunBackgroundLoop() { this->engine_->RunBackgroundLoop(); }

  void RunBackgroundStreamBackLoop() { this->engine_->RunBackgroundStreamBackLoop(); }

  String GetResponseFromStreamOutput(Array<RequestStreamOutput> delta_outputs) {
    std::unordered_map<std::string, std::vector<ChatCompletionStreamResponseChoice>> response_map;
    for (const auto& delta_output : delta_outputs) {
      std::string request_id = delta_output->request_id;
      if (response_map.find(request_id) == response_map.end()) {
        response_map[request_id] = std::vector<ChatCompletionStreamResponseChoice>();
      }
      ChatCompletionStreamResponseChoice choice;

      if (delta_output->group_finish_reason.size() != 1) {
        // Only support n = 1 in ChatCompletionStreamResponse for now
        this->err_ += "Group finish reason should have exactly one element";
      }
      Optional<String> finish_reason = delta_output->group_finish_reason[0];
      if (finish_reason.defined()) {
        if (finish_reason.value() == "stop") {
          choice.finish_reason = FinishReason::stop;
        } else if (finish_reason.value() == "length") {
          choice.finish_reason = FinishReason::length;
        } else if (finish_reason.value() == "tool_calls") {
          choice.finish_reason = FinishReason::tool_calls;
        } else if (finish_reason.value() == "error") {
          choice.finish_reason = FinishReason::error;
        }
      } else {
        choice.finish_reason = std::nullopt;
      }

      choice.index = response_map[request_id].size();

      ChatCompletionMessage delta;
      // Size of delta_output->group_delta_token_ids Array should be 1
      IntTuple delta_token_ids = delta_output->group_delta_token_ids[0];
      std::vector<int32_t> delta_token_ids_vec(delta_token_ids.begin(), delta_token_ids.end());
      delta.content = this->streamer_->Put(delta_token_ids_vec);
      delta.role = "assistant";

      choice.delta = delta;

      response_map[request_id].push_back(choice);
    }

    picojson::array response_arr;
    for (const auto& [request_id, choices] : response_map) {
      ChatCompletionStreamResponse response;
      response.id = request_id;
      response.choices = choices;
      response.model = "json_ffi";  // TODO: Return model name from engine (or from args)
      response.system_fingerprint = "";
      response_arr.push_back(picojson::value(response.AsJSON()));
    }
    return picojson::value(response_arr).serialize();
  }
};

TVM_REGISTER_GLOBAL("mlc.json_ffi.CreateJSONFFIEngine").set_body_typed([]() {
  return Module(make_object<JSONFFIEngineImpl>());
});

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
