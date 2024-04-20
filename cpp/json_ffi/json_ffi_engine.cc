#include "json_ffi_engine.h"

#include <picojson.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

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
  delta.role = Role::assistant;

  ChatCompletionStreamResponseChoice choice;
  choice.finish_reason = FinishReason::error;
  choice.index = 0;
  choice.delta = delta;

  ChatCompletionStreamResponse response;
  response.id = request_id;
  response.choices = std::vector<ChatCompletionStreamResponseChoice>{choice};
  response.model = "json_ffi";  // TODO: Return model name from engine (or from args)
  response.system_fingerprint = "";

  this->request_stream_callback_(Array<String>{picojson::value(response.ToJSON()).serialize()});
}

bool JSONFFIEngine::AddRequest(std::string request_json_str, std::string request_id) {
  std::optional<ChatCompletionRequest> optional_request =
      ChatCompletionRequest::FromJSON(request_json_str, &err_);
  if (!optional_request.has_value()) {
    return false;
  }
  ChatCompletionRequest request = optional_request.value();
  // Create Request
  // TODO: Check if request_id is present already

  // inputs
  // TODO: Apply conv template
  Array<Data> inputs;
  for (const auto& message : request.messages) {
    if (message.content.has_value()) {
      for (const auto& content : message.content.value()) {
        if (content.find("type") == content.end()) {
          err_ += "Content should have a type field";
          return false;
        }
        std::string type = content.at("type");
        if (type == "text") {
          if (content.find("text") == content.end()) {
            err_ += "Content should have a text field";
            return false;
          }
          std::string text = content.at("text");
          inputs.push_back(TextData(text));
        } else {
          err_ += "Content type not supported";
          return false;
        }
      }
    }
  }

  // generation_cfg
  Optional<GenerationConfig> generation_cfg = GenerationConfig::FromJSON(request_json_str, &err_);
  if (!generation_cfg.defined()) {
    return false;
  }

  Request engine_request(request_id, inputs, generation_cfg.value());
  this->engine_->AddRequest(engine_request);

  return true;
}

bool JSONFFIEngine::Abort(std::string request_id) {
  this->engine_->AbortRequest(request_id);
  return true;
}

std::string JSONFFIEngine::GetLastError() { return err_; }

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
  TVM_MODULE_VTABLE_ENTRY("run_background_loop", &JSONFFIEngineImpl::RunBackgroundLoop);
  TVM_MODULE_VTABLE_ENTRY("run_background_stream_back_loop",
                          &JSONFFIEngineImpl::RunBackgroundStreamBackLoop);
  TVM_MODULE_VTABLE_ENTRY("exit_background_loop", &JSONFFIEngineImpl::ExitBackgroundLoop);
  TVM_MODULE_VTABLE_END();

  void InitBackgroundEngine(EngineConfig engine_config,
                            Optional<PackedFunc> request_stream_callback,
                            Optional<EventTraceRecorder> trace_recorder) {
    this->streamer_ = TextStreamer(Tokenizer::FromPath(engine_config->model));

    CHECK(request_stream_callback.defined())
        << "JSONFFIEngine requires request stream callback function, but it is not given.";
    this->request_stream_callback_ = request_stream_callback.value();

    auto frequest_stream_callback_wrapper = [this](TVMArgs args, TVMRetValue* ret) {
      ICHECK_EQ(args.size(), 1);
      Array<RequestStreamOutput> delta_outputs = args[0];
      Array<String> responses = this->GetResponseFromStreamOutput(delta_outputs);
      this->request_stream_callback_(responses);
    };

    request_stream_callback = PackedFunc(frequest_stream_callback_wrapper);
    this->engine_->InitBackgroundEngine(
        std::move(engine_config), std::move(request_stream_callback), std::move(trace_recorder));
  }

  void Reload(EngineConfig engine_config) { this->engine_->Reload(std::move(engine_config)); }

  void Unload() { this->engine_->Unload(); }

  void Reset() { this->engine_->Reset(); }

  void RunBackgroundLoop() { this->engine_->RunBackgroundLoop(); }

  void RunBackgroundStreamBackLoop() { this->engine_->RunBackgroundStreamBackLoop(); }

  Array<String> GetResponseFromStreamOutput(Array<RequestStreamOutput> delta_outputs) {
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
      delta.content = std::vector<std::unordered_map<std::string, std::string>>();
      delta.content.value().push_back(std::unordered_map<std::string, std::string>{
          {"type", "text"}, {"text", this->streamer_->Put(delta_token_ids_vec)}});

      delta.role = Role::assistant;

      choice.delta = delta;

      response_map[request_id].push_back(choice);
    }

    Array<String> response_arr;
    for (const auto& [request_id, choices] : response_map) {
      ChatCompletionStreamResponse response;
      response.id = request_id;
      response.choices = choices;
      response.model = "json_ffi";  // TODO: Return model name from engine (or from args)
      response.system_fingerprint = "";
      response_arr.push_back(picojson::value(response.ToJSON()).serialize());
    }
    return response_arr;
  }
};

TVM_REGISTER_GLOBAL("mlc.json_ffi.CreateJSONFFIEngine").set_body_typed([]() {
  return Module(make_object<JSONFFIEngineImpl>());
});

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
