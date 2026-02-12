/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file json_ffi/openai_api_protocol.cc
 * \brief The implementation of OpenAI API Protocol in MLC LLM.
 */
#include "openai_api_protocol.h"

#include "../support/json_parser.h"

namespace mlc {
namespace llm {
namespace json_ffi {

Result<ChatFunction> ChatFunction::FromJSON(const tvm::ffi::json::Object& json_obj) {
  using TResult = Result<ChatFunction>;
  ChatFunction chat_func;

  // description
  Result<std::optional<std::string>> description_res =
      json::LookupOptionalWithResultReturn<std::string>(json_obj, "description");
  if (description_res.IsErr()) {
    return TResult::Error(description_res.UnwrapErr());
  }
  chat_func.description = description_res.Unwrap();

  // name
  Result<std::string> name_res = json::LookupWithResultReturn<std::string>(json_obj, "name");
  if (name_res.IsErr()) {
    return TResult::Error(name_res.UnwrapErr());
  }
  chat_func.name = name_res.Unwrap();

  // parameters
  Result<tvm::ffi::json::Object> parameters_obj_res =
      json::LookupWithResultReturn<tvm::ffi::json::Object>(json_obj, "parameters");
  if (parameters_obj_res.IsErr()) {
    return TResult::Error(parameters_obj_res.UnwrapErr());
  }
  tvm::ffi::json::Object parameters_obj = parameters_obj_res.Unwrap();
  chat_func.parameters.reserve(parameters_obj.size());
  for (const auto& [key, value] : parameters_obj) {
    chat_func.parameters[key.cast<tvm::ffi::String>()] = tvm::ffi::json::Stringify(value);
  }

  return TResult::Ok(chat_func);
}

tvm::ffi::json::Object ChatFunction::AsJSON() const {
  tvm::ffi::json::Object obj;
  if (this->description.has_value()) {
    obj.Set("description", this->description.value());
  }
  obj.Set("name", this->name);
  tvm::ffi::json::Object parameters_obj;
  for (const auto& pair : this->parameters) {
    parameters_obj.Set(pair.first, pair.second);
  }
  obj.Set("parameters", parameters_obj);
  return obj;
}

Result<ChatTool> ChatTool::FromJSON(const tvm::ffi::json::Object& json_obj) {
  using TResult = Result<ChatTool>;
  ChatTool chatTool;

  // function
  Result<tvm::ffi::json::Object> function_obj_res =
      json::LookupWithResultReturn<tvm::ffi::json::Object>(json_obj, "function");
  if (function_obj_res.IsErr()) {
    return TResult::Error(function_obj_res.UnwrapErr());
  }
  Result<ChatFunction> function = ChatFunction::FromJSON(function_obj_res.Unwrap());
  if (function.IsErr()) {
    return TResult::Error(function.UnwrapErr());
  }
  chatTool.function = function.Unwrap();

  return TResult::Ok(chatTool);
}

tvm::ffi::json::Object ChatTool::AsJSON() const {
  tvm::ffi::json::Object obj;
  obj.Set("type", "function");
  obj.Set("function", this->function.AsJSON());
  return obj;
}

Result<ChatFunctionCall> ChatFunctionCall::FromJSON(const tvm::ffi::json::Object& json_obj) {
  using TResult = Result<ChatFunctionCall>;
  ChatFunctionCall chat_func_call;

  // name
  Result<std::string> name_res = json::LookupWithResultReturn<std::string>(json_obj, "name");
  if (name_res.IsErr()) {
    return TResult::Error(name_res.UnwrapErr());
  }
  chat_func_call.name = name_res.Unwrap();

  // arguments
  Result<std::optional<tvm::ffi::json::Object>> arguments_obj_res =
      json::LookupOptionalWithResultReturn<tvm::ffi::json::Object>(json_obj, "arguments");
  if (arguments_obj_res.IsErr()) {
    return TResult::Error(arguments_obj_res.UnwrapErr());
  }
  std::optional<tvm::ffi::json::Object> arguments_obj = arguments_obj_res.Unwrap();
  if (arguments_obj.has_value()) {
    std::unordered_map<std::string, std::string> arguments;
    arguments.reserve(arguments_obj.value().size());
    for (const auto& [key, value] : arguments_obj.value()) {
      arguments[key.cast<tvm::ffi::String>()] = tvm::ffi::json::Stringify(value);
    }
    chat_func_call.arguments = std::move(arguments);
  }

  return TResult::Ok(chat_func_call);
}

tvm::ffi::json::Object ChatFunctionCall::AsJSON() const {
  tvm::ffi::json::Object obj;
  tvm::ffi::json::Object arguments_obj;
  if (this->arguments.has_value()) {
    for (const auto& pair : this->arguments.value()) {
      arguments_obj.Set(pair.first, pair.second);
    }
    obj.Set("arguments", arguments_obj);
  }

  obj.Set("name", this->name);
  return obj;
}

Result<ChatToolCall> ChatToolCall::FromJSON(const tvm::ffi::json::Object& json_obj) {
  using TResult = Result<ChatToolCall>;
  ChatToolCall chat_tool_call;

  // function
  Result<tvm::ffi::json::Object> function_obj_res =
      json::LookupWithResultReturn<tvm::ffi::json::Object>(json_obj, "function");
  if (function_obj_res.IsErr()) {
    return TResult::Error(function_obj_res.UnwrapErr());
  }
  Result<ChatFunctionCall> function_res = ChatFunctionCall::FromJSON(function_obj_res.Unwrap());
  if (function_res.IsErr()) {
    return TResult::Error(function_res.UnwrapErr());
  }
  chat_tool_call.function = function_res.Unwrap();

  // overwrite default id
  Result<std::optional<std::string>> id_res =
      json::LookupOptionalWithResultReturn<std::string>(json_obj, "id");
  if (id_res.IsErr()) {
    return TResult::Error(id_res.UnwrapErr());
  }
  std::optional<std::string> id = id_res.UnwrapErr();
  if (id.has_value()) {
    chat_tool_call.id = id.value();
  }

  return TResult::Ok(chat_tool_call);
}

tvm::ffi::json::Object ChatToolCall::AsJSON() const {
  tvm::ffi::json::Object obj;
  obj.Set("id", this->id);
  obj.Set("function", this->function.AsJSON());
  obj.Set("type", "function");
  return obj;
}

Result<ChatCompletionMessage> ChatCompletionMessage::FromJSON(
    const tvm::ffi::json::Object& json_obj) {
  using TResult = Result<ChatCompletionMessage>;
  ChatCompletionMessage message;
  ChatCompletionMessageContent content;

  // content
  if (json_obj.count("content") == 0) {
    return TResult::Error("ValueError: key \"content\" not found in the chat completion.");
  }
  tvm::ffi::json::Value content_val = json_obj.at("content");
  if (content_val.try_cast<std::string>().has_value()) {
    content = content_val.cast<std::string>();
  } else if (content_val == nullptr) {
    // skip
  } else {
    // most complicated case
    std::vector<std::unordered_map<std::string, std::string>> parts;
    Result<tvm::ffi::json::Array> content_arr_res =
        json::LookupWithResultReturn<tvm::ffi::json::Array>(json_obj, "content");
    if (content_arr_res.IsErr()) {
      return TResult::Error(content_arr_res.UnwrapErr());
    }
    tvm::ffi::json::Array content_arr = content_arr_res.Unwrap();
    for (const auto& item : content_arr) {
      if (!item.try_cast<tvm::ffi::json::Object>().has_value()) {
        return TResult::Error("The content of chat completion message is not an object");
      }
      tvm::ffi::json::Object item_obj = item.cast<tvm::ffi::json::Object>();
      std::unordered_map<std::string, std::string> item_map;
      for (const auto& [key, value] : item_obj) {
        item_map[key.cast<tvm::ffi::String>()] = tvm::ffi::json::Stringify(value);
      }
      parts.push_back(std::move(item_map));
    }
    content = parts;
  }
  message.content = content;

  // role
  Result<std::string> role_str_res = json::LookupWithResultReturn<std::string>(json_obj, "role");
  if (role_str_res.IsErr()) {
    return TResult::Error(role_str_res.UnwrapErr());
  }
  std::string role_str = role_str_res.Unwrap();
  if (role_str == "system" || role_str == "user" || role_str == "assistant" || role_str == "tool") {
    message.role = role_str;
  } else {
    return TResult::Error("Invalid role in chat completion message: " + role_str);
  }

  // name
  Result<std::optional<std::string>> name_res =
      json::LookupOptionalWithResultReturn<std::string>(json_obj, "name");
  if (name_res.IsErr()) {
    return TResult::Error(name_res.UnwrapErr());
  }
  message.name = name_res.Unwrap();

  // tool calls
  Result<std::optional<tvm::ffi::json::Array>> tool_calls_arr_res =
      json::LookupOptionalWithResultReturn<tvm::ffi::json::Array>(json_obj, "tool_calls");
  if (tool_calls_arr_res.IsErr()) {
    return TResult::Error(tool_calls_arr_res.UnwrapErr());
  }
  std::optional<tvm::ffi::json::Array> tool_calls_arr = tool_calls_arr_res.Unwrap();
  if (tool_calls_arr.has_value()) {
    std::vector<ChatToolCall> tool_calls;
    tool_calls.reserve(tool_calls_arr.value().size());
    for (const auto& item : tool_calls_arr.value()) {
      if (!item.try_cast<tvm::ffi::json::Object>().has_value()) {
        return TResult::Error("A tool call item in the chat completion message is not an object");
      }
      Result<ChatToolCall> tool_call = ChatToolCall::FromJSON(item.cast<tvm::ffi::json::Object>());
      if (tool_call.IsErr()) {
        return TResult::Error(tool_call.UnwrapErr());
      }
      tool_calls.push_back(tool_call.Unwrap());
    }
    message.tool_calls = tool_calls;
  }

  // tool call id
  Result<std::optional<std::string>> tool_call_id_res =
      json::LookupOptionalWithResultReturn<std::string>(json_obj, "tool_call_id");
  if (tool_call_id_res.IsErr()) {
    return TResult::Error(tool_call_id_res.UnwrapErr());
  }
  message.tool_call_id = tool_call_id_res.Unwrap();

  return TResult::Ok(message);
}

Result<ChatCompletionRequest> ChatCompletionRequest::FromJSON(const std::string& json_str) {
  using TResult = Result<ChatCompletionRequest>;
  Result<tvm::ffi::json::Object> json_obj_res = json::ParseToJSONObjectWithResultReturn(json_str);
  if (json_obj_res.IsErr()) {
    return TResult::Error(json_obj_res.UnwrapErr());
  }
  tvm::ffi::json::Object json_obj = json_obj_res.Unwrap();
  ChatCompletionRequest request;

  // messages
  Result<tvm::ffi::json::Array> messages_arr_res =
      json::LookupWithResultReturn<tvm::ffi::json::Array>(json_obj, "messages");
  if (messages_arr_res.IsErr()) {
    return TResult::Error(messages_arr_res.UnwrapErr());
  }
  std::vector<ChatCompletionMessage> messages;
  tvm::ffi::json::Array messages_arr = messages_arr_res.Unwrap();
  for (const auto& item : messages_arr) {
    if (!item.try_cast<tvm::ffi::json::Object>().has_value()) {
      return TResult::Error("A message in chat completion request is not object");
    }
    tvm::ffi::json::Object item_obj = item.cast<tvm::ffi::json::Object>();
    Result<ChatCompletionMessage> message = ChatCompletionMessage::FromJSON(item_obj);
    if (message.IsErr()) {
      return TResult::Error(message.UnwrapErr());
    }
    messages.push_back(message.Unwrap());
  }
  request.messages = messages;

  // model
  Result<std::optional<std::string>> model_res =
      json::LookupOptionalWithResultReturn<std::string>(json_obj, "model");
  if (model_res.IsErr()) {
    return TResult::Error(model_res.UnwrapErr());
  }
  request.model = model_res.Unwrap();

  // temperature
  Result<std::optional<double>> temperature_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "temperature");
  if (temperature_res.IsErr()) {
    return TResult::Error(temperature_res.UnwrapErr());
  }
  request.temperature = temperature_res.Unwrap();
  // top_p
  Result<std::optional<double>> top_p_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "top_p");
  if (top_p_res.IsErr()) {
    return TResult::Error(top_p_res.UnwrapErr());
  }
  request.top_p = top_p_res.Unwrap();
  // max_tokens
  Result<std::optional<int64_t>> max_tokens_res =
      json::LookupOptionalWithResultReturn<int64_t>(json_obj, "max_tokens");
  if (max_tokens_res.IsErr()) {
    return TResult::Error(max_tokens_res.UnwrapErr());
  }
  request.max_tokens = max_tokens_res.Unwrap();
  // n
  Result<int64_t> n_res = json::LookupOrDefaultWithResultReturn<int64_t>(json_obj, "n", 1);
  if (n_res.IsErr()) {
    return TResult::Error(n_res.UnwrapErr());
  }
  request.n = n_res.Unwrap();
  // frequency_penalty
  Result<std::optional<double>> frequency_penalty_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "frequency_penalty");
  if (frequency_penalty_res.IsErr()) {
    return TResult::Error(frequency_penalty_res.UnwrapErr());
  }
  request.frequency_penalty = frequency_penalty_res.Unwrap();
  // presence_penalty
  Result<std::optional<double>> presence_penalty_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "presence_penalty");
  if (presence_penalty_res.IsErr()) {
    return TResult::Error(presence_penalty_res.UnwrapErr());
  }
  request.presence_penalty = presence_penalty_res.Unwrap();
  // seed
  Result<std::optional<int64_t>> seed_res =
      json::LookupOptionalWithResultReturn<int64_t>(json_obj, "seed");
  if (seed_res.IsErr()) {
    return TResult::Error(seed_res.UnwrapErr());
  }
  request.seed = seed_res.Unwrap();

  // stop strings
  Result<std::optional<tvm::ffi::json::Array>> stop_strs_res =
      json::LookupOptionalWithResultReturn<tvm::ffi::json::Array>(json_obj, "stop");
  if (stop_strs_res.IsErr()) {
    return TResult::Error(stop_strs_res.UnwrapErr());
  }
  std::optional<tvm::ffi::json::Array> stop_strs = stop_strs_res.Unwrap();
  if (stop_strs.has_value()) {
    std::vector<std::string> stop;
    for (const auto& stop_str_value : stop_strs.value()) {
      if (!stop_str_value.try_cast<std::string>().has_value()) {
        return TResult::Error("One given value in field \"stop\" is not a string.");
      }
      stop.push_back(stop_str_value.cast<std::string>());
    }
    request.stop = std::move(stop);
  }

  // tool_choice
  Result<std::string> tool_choice_res =
      json::LookupOrDefaultWithResultReturn<std::string>(json_obj, "tool_choice", "auto");
  if (tool_choice_res.IsErr()) {
    return TResult::Error(tool_choice_res.UnwrapErr());
  }
  request.tool_choice = tool_choice_res.Unwrap();

  // tools
  Result<std::optional<tvm::ffi::json::Array>> tools_arr_res =
      json::LookupOptionalWithResultReturn<tvm::ffi::json::Array>(json_obj, "tools");
  if (tool_choice_res.IsErr()) {
    return TResult::Error(tool_choice_res.UnwrapErr());
  }
  std::optional<tvm::ffi::json::Array> tools_arr = tools_arr_res.Unwrap();
  if (tools_arr.has_value()) {
    std::vector<ChatTool> tools;
    tools.reserve(tools_arr.value().size());
    for (const auto& item : tools_arr.value()) {
      if (!item.try_cast<tvm::ffi::json::Object>().has_value()) {
        return TResult::Error("A tool of the chat completion request is not an object");
      }
      Result<ChatTool> tool = ChatTool::FromJSON(item.cast<tvm::ffi::json::Object>());
      if (tool.IsErr()) {
        return TResult::Error(tool.UnwrapErr());
      }
      tools.push_back(tool.Unwrap());
    }
    request.tools = tools;
  }

  // response format
  std::optional<tvm::ffi::json::Object> response_format_obj =
      json::LookupOptional<tvm::ffi::json::Object>(json_obj, "response_format");
  if (response_format_obj.has_value()) {
    Result<ResponseFormat> response_format_res =
        ResponseFormat::FromJSON(response_format_obj.value());
    if (response_format_res.IsErr()) {
      return TResult::Error(response_format_res.UnwrapErr());
    }
    request.response_format = response_format_res.Unwrap();
  }

  // debug_config
  Result<std::optional<tvm::ffi::json::Object>> debug_config_opt_res =
      json::LookupOptionalWithResultReturn<tvm::ffi::json::Object>(json_obj, "debug_config");
  if (debug_config_opt_res.IsErr()) {
    return TResult::Error(debug_config_opt_res.UnwrapErr());
  }
  auto debug_config_opt = debug_config_opt_res.Unwrap();
  if (debug_config_opt.has_value()) {
    Result<DebugConfig> debug_config_res = DebugConfig::FromJSON(debug_config_opt.value());
    if (debug_config_res.IsErr()) {
      return TResult::Error(debug_config_res.UnwrapErr());
    }
    request.debug_config = debug_config_res.Unwrap();
  }

  // TODO: Other parameters
  return TResult::Ok(request);
}

tvm::ffi::json::Object ChatCompletionMessage::AsJSON() const {
  tvm::ffi::json::Object obj;

  if (this->content.IsText()) {
    obj.Set("content", this->content.Text());
  } else if (this->content.IsParts()) {
    tvm::ffi::json::Array content_arr;
    for (const auto& item : this->content.Parts()) {
      tvm::ffi::json::Object item_obj;
      for (const auto& pair : item) {
        item_obj.Set(pair.first, pair.second);
      }
      content_arr.push_back(item_obj);
    }
    obj.Set("content", content_arr);
  }

  obj.Set("role", this->role);

  if (this->name.has_value()) {
    obj.Set("name", this->name.value());
  }
  if (this->tool_call_id.has_value()) {
    obj.Set("tool_call_id", this->tool_call_id.value());
  }
  if (this->tool_calls.has_value()) {
    tvm::ffi::json::Array tool_calls_arr;
    for (const auto& tool_call : this->tool_calls.value()) {
      tool_calls_arr.push_back(tool_call.AsJSON());
    }
    obj.Set("tool_calls", tool_calls_arr);
  }
  return obj;
}

tvm::ffi::json::Object ChatCompletionResponseChoice::AsJSON() const {
  tvm::ffi::json::Object obj;
  if (!this->finish_reason.has_value()) {
    obj.Set("finish_reason", nullptr);
  } else {
    if (this->finish_reason == FinishReason::stop) {
      obj.Set("finish_reason", "stop");
    } else if (this->finish_reason == FinishReason::length) {
      obj.Set("finish_reason", "length");
    } else if (this->finish_reason == FinishReason::tool_calls) {
      obj.Set("finish_reason", "tool_calls");
    } else if (this->finish_reason == FinishReason::error) {
      obj.Set("finish_reason", "error");
    }
  }
  obj.Set("index", static_cast<int64_t>(this->index));
  obj.Set("message", this->message.AsJSON());
  return obj;
}

tvm::ffi::json::Object ChatCompletionStreamResponseChoice::AsJSON() const {
  tvm::ffi::json::Object obj;
  if (!this->finish_reason.has_value()) {
    obj.Set("finish_reason", nullptr);
  } else {
    if (this->finish_reason.value() == FinishReason::stop) {
      obj.Set("finish_reason", "stop");
    } else if (this->finish_reason.value() == FinishReason::length) {
      obj.Set("finish_reason", "length");
    } else if (this->finish_reason.value() == FinishReason::tool_calls) {
      obj.Set("finish_reason", "tool_calls");
    } else if (this->finish_reason.value() == FinishReason::error) {
      obj.Set("finish_reason", "error");
    }
  }

  obj.Set("index", static_cast<int64_t>(this->index));
  obj.Set("delta", this->delta.AsJSON());
  return obj;
}

tvm::ffi::json::Object ChatCompletionResponse::AsJSON() const {
  tvm::ffi::json::Object obj;
  obj.Set("id", this->id);
  tvm::ffi::json::Array choices_arr;
  for (const auto& choice : this->choices) {
    choices_arr.push_back(choice.AsJSON());
  }
  obj.Set("choices", choices_arr);
  obj.Set("created", static_cast<int64_t>(this->created));
  obj.Set("model", this->model);
  obj.Set("system_fingerprint", this->system_fingerprint);
  obj.Set("object", this->object);
  return obj;
}

tvm::ffi::json::Object ChatCompletionStreamResponse::AsJSON() const {
  tvm::ffi::json::Object obj;
  obj.Set("id", this->id);

  tvm::ffi::json::Array choices_arr;
  for (const auto& choice : this->choices) {
    choices_arr.push_back(choice.AsJSON());
  }
  obj.Set("choices", choices_arr);

  obj.Set("created", static_cast<int64_t>(this->created));
  obj.Set("model", this->model);
  obj.Set("system_fingerprint", this->system_fingerprint);
  obj.Set("object", this->object);
  if (usage.has_value()) {
    obj.Set("usage", usage.value());
  }
  return obj;
}

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
