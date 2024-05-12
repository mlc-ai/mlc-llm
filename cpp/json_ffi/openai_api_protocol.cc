/*!
 *  Copyright (c) 2023 by Contributors
 * \file json_ffi/openai_api_protocol.cc
 * \brief The implementation of OpenAI API Protocol in MLC LLM.
 */
#include "openai_api_protocol.h"

#include "../support/json_parser.h"

namespace mlc {
namespace llm {
namespace json_ffi {

Result<ChatFunction> ChatFunction::FromJSON(const picojson::object& json_obj) {
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
  Result<picojson::object> parameters_obj_res =
      json::LookupWithResultReturn<picojson::object>(json_obj, "parameters");
  if (parameters_obj_res.IsErr()) {
    return TResult::Error(parameters_obj_res.UnwrapErr());
  }
  picojson::object parameters_obj = parameters_obj_res.Unwrap();
  chat_func.parameters.reserve(parameters_obj.size());
  for (const auto& [key, value] : parameters_obj) {
    chat_func.parameters[key] = value.to_str();
  }

  return TResult::Ok(chat_func);
}

picojson::object ChatFunction::AsJSON() const {
  picojson::object obj;
  if (this->description.has_value()) {
    obj["description"] = picojson::value(this->description.value());
  }
  obj["name"] = picojson::value(this->name);
  picojson::object parameters_obj;
  for (const auto& pair : this->parameters) {
    parameters_obj[pair.first] = picojson::value(pair.second);
  }
  obj["parameters"] = picojson::value(parameters_obj);
  return obj;
}

Result<ChatTool> ChatTool::FromJSON(const picojson::object& json_obj) {
  using TResult = Result<ChatTool>;
  ChatTool chatTool;

  // function
  Result<picojson::object> function_obj_res =
      json::LookupWithResultReturn<picojson::object>(json_obj, "function");
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

picojson::object ChatTool::AsJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value("function");
  obj["function"] = picojson::value(this->function.AsJSON());
  return obj;
}

Result<ChatFunctionCall> ChatFunctionCall::FromJSON(const picojson::object& json_obj) {
  using TResult = Result<ChatFunctionCall>;
  ChatFunctionCall chat_func_call;

  // name
  Result<std::string> name_res = json::LookupWithResultReturn<std::string>(json_obj, "name");
  if (name_res.IsErr()) {
    return TResult::Error(name_res.UnwrapErr());
  }
  chat_func_call.name = name_res.Unwrap();

  // arguments
  Result<std::optional<picojson::object>> arguments_obj_res =
      json::LookupOptionalWithResultReturn<picojson::object>(json_obj, "arguments");
  if (arguments_obj_res.IsErr()) {
    return TResult::Error(arguments_obj_res.UnwrapErr());
  }
  std::optional<picojson::object> arguments_obj = arguments_obj_res.Unwrap();
  if (arguments_obj.has_value()) {
    std::unordered_map<std::string, std::string> arguments;
    arguments.reserve(arguments_obj.value().size());
    for (const auto& [key, value] : arguments_obj.value()) {
      arguments[key] = value.to_str();
    }
    chat_func_call.arguments = std::move(arguments);
  }

  return TResult::Ok(chat_func_call);
}

picojson::object ChatFunctionCall::AsJSON() const {
  picojson::object obj;
  picojson::object arguments_obj;
  if (this->arguments.has_value()) {
    for (const auto& pair : this->arguments.value()) {
      arguments_obj[pair.first] = picojson::value(pair.second);
    }
    obj["arguments"] = picojson::value(arguments_obj);
  }

  obj["name"] = picojson::value(this->name);
  return obj;
}

Result<ChatToolCall> ChatToolCall::FromJSON(const picojson::object& json_obj) {
  using TResult = Result<ChatToolCall>;
  ChatToolCall chat_tool_call;

  // function
  Result<picojson::object> function_obj_res =
      json::LookupWithResultReturn<picojson::object>(json_obj, "function");
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

picojson::object ChatToolCall::AsJSON() const {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  obj["function"] = picojson::value(this->function.AsJSON());
  obj["type"] = picojson::value("function");
  return obj;
}

Result<ChatCompletionMessage> ChatCompletionMessage::FromJSON(const picojson::object& json_obj) {
  using TResult = Result<ChatCompletionMessage>;
  ChatCompletionMessage message;

  // content
  Result<picojson::array> content_arr_res =
      json::LookupWithResultReturn<picojson::array>(json_obj, "content");
  if (content_arr_res.IsErr()) {
    return TResult::Error(content_arr_res.UnwrapErr());
  }
  std::vector<std::unordered_map<std::string, std::string>> content;
  for (const auto& item : content_arr_res.Unwrap()) {
    // Todo(mlc-team): allow content item to be a single string.
    if (!item.is<picojson::object>()) {
      return TResult::Error("The content of chat completion message is not an object");
    }
    picojson::object item_obj = item.get<picojson::object>();
    std::unordered_map<std::string, std::string> item_map;
    for (const auto& [key, value] : item_obj) {
      item_map[key] = value.to_str();
    }
    content.push_back(std::move(item_map));
  }
  message.content = content;

  // role
  Result<std::string> role_str_res = json::LookupWithResultReturn<std::string>(json_obj, "role");
  if (role_str_res.IsErr()) {
    return TResult::Error(role_str_res.UnwrapErr());
  }
  std::string role_str = role_str_res.Unwrap();
  if (role_str == "system") {
    message.role = Role::system;
  } else if (role_str == "user") {
    message.role = Role::user;
  } else if (role_str == "assistant") {
    message.role = Role::assistant;
  } else if (role_str == "tool") {
    message.role = Role::tool;
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
  Result<std::optional<picojson::array>> tool_calls_arr_res =
      json::LookupOptionalWithResultReturn<picojson::array>(json_obj, "tool_calls");
  if (tool_calls_arr_res.IsErr()) {
    return TResult::Error(tool_calls_arr_res.UnwrapErr());
  }
  std::optional<picojson::array> tool_calls_arr = tool_calls_arr_res.Unwrap();
  if (tool_calls_arr.has_value()) {
    std::vector<ChatToolCall> tool_calls;
    tool_calls.reserve(tool_calls_arr.value().size());
    for (const auto& item : tool_calls_arr.value()) {
      if (!item.is<picojson::object>()) {
        return TResult::Error("A tool call item in the chat completion message is not an object");
      }
      Result<ChatToolCall> tool_call = ChatToolCall::FromJSON(item.get<picojson::object>());
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
  Result<picojson::object> json_obj_res = json::ParseToJSONObjectWithResultReturn(json_str);
  if (json_obj_res.IsErr()) {
    return TResult::Error(json_obj_res.UnwrapErr());
  }
  picojson::object json_obj = json_obj_res.Unwrap();
  ChatCompletionRequest request;

  // messages
  Result<picojson::array> messages_arr_res =
      json::LookupWithResultReturn<picojson::array>(json_obj, "messages");
  if (messages_arr_res.IsErr()) {
    return TResult::Error(messages_arr_res.UnwrapErr());
  }
  std::vector<ChatCompletionMessage> messages;
  for (const auto& item : messages_arr_res.Unwrap()) {
    if (!item.is<picojson::object>()) {
      return TResult::Error("A message in chat completion request is not object");
    }
    picojson::object item_obj = item.get<picojson::object>();
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

  // max_tokens
  Result<std::optional<int64_t>> max_tokens_res =
      json::LookupOptionalWithResultReturn<int64_t>(json_obj, "max_tokens");
  if (max_tokens_res.IsErr()) {
    return TResult::Error(max_tokens_res.UnwrapErr());
  }
  request.max_tokens = max_tokens_res.Unwrap();

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

  // tool_choice
  Result<std::string> tool_choice_res =
      json::LookupOrDefaultWithResultReturn<std::string>(json_obj, "tool_choice", "auto");
  if (tool_choice_res.IsErr()) {
    return TResult::Error(tool_choice_res.UnwrapErr());
  }
  request.tool_choice = tool_choice_res.Unwrap();

  // tools
  Result<std::optional<picojson::array>> tools_arr_res =
      json::LookupOptionalWithResultReturn<picojson::array>(json_obj, "tools");
  if (tool_choice_res.IsErr()) {
    return TResult::Error(tool_choice_res.UnwrapErr());
  }
  std::optional<picojson::array> tools_arr = tools_arr_res.Unwrap();
  if (tools_arr.has_value()) {
    std::vector<ChatTool> tools;
    tools.reserve(tools_arr.value().size());
    for (const auto& item : tools_arr.value()) {
      if (!item.is<picojson::object>()) {
        return TResult::Error("A tool of the chat completion request is not an object");
      }
      Result<ChatTool> tool = ChatTool::FromJSON(item.get<picojson::object>());
      if (tool.IsErr()) {
        return TResult::Error(tool.UnwrapErr());
      }
      tools.push_back(tool.Unwrap());
    }
    request.tools = tools;
  }

  // TODO: Other parameters

  return TResult::Ok(request);
}

picojson::object ChatCompletionMessage::AsJSON() const {
  picojson::object obj;
  picojson::array content_arr;
  for (const auto& item : this->content.value()) {
    picojson::object item_obj;
    for (const auto& pair : item) {
      item_obj[pair.first] = picojson::value(pair.second);
    }
    content_arr.push_back(picojson::value(item_obj));
  }
  obj["content"] = picojson::value(content_arr);
  if (this->role == Role::system) {
    obj["role"] = picojson::value("system");
  } else if (this->role == Role::user) {
    obj["role"] = picojson::value("user");
  } else if (this->role == Role::assistant) {
    obj["role"] = picojson::value("assistant");
  } else if (this->role == Role::tool) {
    obj["role"] = picojson::value("tool");
  }
  if (this->name.has_value()) {
    obj["name"] = picojson::value(this->name.value());
  }
  if (this->tool_call_id.has_value()) {
    obj["tool_call_id"] = picojson::value(this->tool_call_id.value());
  }
  if (this->tool_calls.has_value()) {
    picojson::array tool_calls_arr;
    for (const auto& tool_call : this->tool_calls.value()) {
      tool_calls_arr.push_back(picojson::value(tool_call.AsJSON()));
    }
    obj["tool_calls"] = picojson::value(tool_calls_arr);
  }
  return obj;
}

Result<Conversation> ChatCompletionRequest::CheckFunctionCalling(Conversation conv_template) {
  using TResult = Result<Conversation>;
  if (!tools.has_value() || (tool_choice.has_value() && tool_choice.value() == "none")) {
    conv_template.use_function_calling = false;
    return TResult::Ok(conv_template);
  }
  std::vector<ChatTool> tools_ = tools.value();
  std::string tool_choice_ = tool_choice.value();

  // TODO: support with tool choice as dict
  for (const auto& tool : tools_) {
    if (tool.function.name == tool_choice_) {
      conv_template.use_function_calling = true;
      picojson::value function_str(tool.function.AsJSON());
      conv_template.function_string = function_str.serialize();
      return TResult::Ok(conv_template);
    }
  }

  if (tool_choice_ != "auto") {
    return TResult::Error("Invalid tool_choice value in the request: " + tool_choice_);
  }

  picojson::array function_list;
  for (const auto& tool : tools_) {
    function_list.push_back(picojson::value(tool.function.AsJSON()));
  }

  conv_template.use_function_calling = true;
  picojson::value function_list_json(function_list);
  conv_template.function_string = function_list_json.serialize();
  return TResult::Ok(conv_template);
};

picojson::object ChatCompletionResponseChoice::AsJSON() const {
  picojson::object obj;
  if (!this->finish_reason.has_value()) {
    obj["finish_reason"] = picojson::value();
  } else {
    if (this->finish_reason == FinishReason::stop) {
      obj["finish_reason"] = picojson::value("stop");
    } else if (this->finish_reason == FinishReason::length) {
      obj["finish_reason"] = picojson::value("length");
    } else if (this->finish_reason == FinishReason::tool_calls) {
      obj["finish_reason"] = picojson::value("tool_calls");
    } else if (this->finish_reason == FinishReason::error) {
      obj["finish_reason"] = picojson::value("error");
    }
  }
  obj["index"] = picojson::value((int64_t)this->index);
  obj["message"] = picojson::value(this->message.AsJSON());
  return obj;
}

picojson::object ChatCompletionStreamResponseChoice::AsJSON() const {
  picojson::object obj;
  if (!this->finish_reason.has_value()) {
    obj["finish_reason"] = picojson::value();
  } else {
    if (this->finish_reason.value() == FinishReason::stop) {
      obj["finish_reason"] = picojson::value("stop");
    } else if (this->finish_reason.value() == FinishReason::length) {
      obj["finish_reason"] = picojson::value("length");
    } else if (this->finish_reason.value() == FinishReason::tool_calls) {
      obj["finish_reason"] = picojson::value("tool_calls");
    } else if (this->finish_reason.value() == FinishReason::error) {
      obj["finish_reason"] = picojson::value("error");
    }
  }

  obj["index"] = picojson::value((int64_t)this->index);
  obj["delta"] = picojson::value(this->delta.AsJSON());
  return obj;
}

picojson::object ChatCompletionResponse::AsJSON() const {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  picojson::array choices_arr;
  for (const auto& choice : this->choices) {
    choices_arr.push_back(picojson::value(choice.AsJSON()));
  }
  obj["choices"] = picojson::value(choices_arr);
  obj["created"] = picojson::value((int64_t)this->created);
  obj["model"] = picojson::value(this->model);
  obj["system_fingerprint"] = picojson::value(this->system_fingerprint);
  obj["object"] = picojson::value(this->object);
  return obj;
}

picojson::object ChatCompletionStreamResponse::AsJSON() const {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  picojson::array choices_arr;
  for (const auto& choice : this->choices) {
    choices_arr.push_back(picojson::value(choice.AsJSON()));
  }
  obj["choices"] = picojson::value(choices_arr);
  obj["created"] = picojson::value((int64_t)this->created);
  obj["model"] = picojson::value(this->model);
  obj["system_fingerprint"] = picojson::value(this->system_fingerprint);
  obj["object"] = picojson::value(this->object);
  return obj;
}

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
