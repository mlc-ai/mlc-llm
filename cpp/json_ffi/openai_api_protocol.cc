/*!
 *  Copyright (c) 2023 by Contributors
 * \file json_ffi/openai_api_protocol.cc
 * \brief The implementation of OpenAI API Protocol in MLC LLM.
 */
#include "openai_api_protocol.h"

#include "../metadata/json_parser.h"

namespace mlc {
namespace llm {
namespace json_ffi {

std::string generate_uuid_string(size_t length) {
    auto randchar = []() -> char {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
}

std::optional<ChatFunction> ChatFunction::FromJSON(const picojson::object& json_obj, std::string* err) {
  ChatFunction chatFunc;

  // description (optional)
  std::string description;
  if (json::ParseJSONField(json_obj, "description", description, err, false)) {
    chatFunc.description = description;
  }
  
  // name
  std::string name;
  if (!json::ParseJSONField(json_obj, "name", name, err, true)) {
    return std::nullopt;
  }
  chatFunc.name = name;

  // parameters
  picojson::object parameters_obj;
  if (!json::ParseJSONField(json_obj, "parameters", parameters_obj, err, true)) {
    return std::nullopt;
  }
  std::unordered_map<std::string, std::string> parameters;
  for (picojson::value::object::const_iterator i = parameters_obj.begin(); i != parameters_obj.end(); ++i) {
    parameters[i->first] = i->second.to_str();
  }
  chatFunc.parameters = parameters;

  return chatFunc;
}


picojson::object ChatFunction::ToJSON() const {
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


std::optional<ChatTool> ChatTool::FromJSON(const picojson::object& json_obj, std::string* err) {
  ChatTool chatTool;

  // function
  picojson::object function_obj;
  if (!json::ParseJSONField(json_obj, "function", function_obj, err, true)) {
    return std::nullopt;
  }

  std::optional<ChatFunction> function = ChatFunction::FromJSON(function_obj, err);
  if (!function.has_value()){
    return std::nullopt;
  }
  chatTool.function = function.value();

  return chatTool;
}

picojson::object ChatTool::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value("function");
  obj["function"] = picojson::value(this->function.ToJSON());
  return obj;
}

std::optional<ChatFunctionCall> ChatFunctionCall::FromJSON(const picojson::object& json_obj, std::string* err) {
  ChatFunctionCall chatFuncCall;
  
  // name
  std::string name;
  if (!json::ParseJSONField(json_obj, "name", name, err, true)) {
    return std::nullopt;
  }
  chatFuncCall.name = name;

  // arguments
  picojson::object arguments_obj;
  if (json::ParseJSONField(json_obj, "arguments", arguments_obj, err, false)) {
    std::unordered_map<std::string, std::string> arguments;
    for (picojson::value::object::const_iterator i = arguments_obj.begin(); i != arguments_obj.end(); ++i) {
      arguments[i->first] = i->second.to_str();
    }
    chatFuncCall.arguments = arguments;
  }

  return chatFuncCall;
}

picojson::object ChatFunctionCall::ToJSON() const {
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

std::optional<ChatToolCall> ChatToolCall::FromJSON(const picojson::object& json_obj, std::string* err) {
  ChatToolCall chatToolCall;

  // function
  picojson::object function_obj;
  if (!json::ParseJSONField(json_obj, "function", function_obj, err, true)) {
    return std::nullopt;
  }

  std::optional<ChatFunctionCall> function = ChatFunctionCall::FromJSON(function_obj, err);
  if (!function.has_value()){
    return std::nullopt;
  };
  chatToolCall.function = function.value();

  // overwrite default id
  std::string id;
  if (!json::ParseJSONField(json_obj, "id", id, err, false)) {
    return std::nullopt;
  }
  chatToolCall.id = id;

  return chatToolCall;
}

picojson::object ChatToolCall::ToJSON() const {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  obj["function"] = picojson::value(this->function.ToJSON());
  obj["type"] = picojson::value("function");
  return obj;
}

std::optional<ChatCompletionMessage> ChatCompletionMessage::FromJSON(const picojson::object& json_obj,
                                                                     std::string* err) {
  ChatCompletionMessage message;

  // content
  picojson::array content_arr;
  if (!json::ParseJSONField(json_obj, "content", content_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<std::unordered_map<std::string, std::string> > content;
  for (const auto& item : content_arr) {
    if (!item.is<picojson::object>()) {
      *err += "Content item is not an object";
      return std::nullopt;
    }
    std::unordered_map<std::string, std::string> item_map;
    picojson::object item_obj = item.get<picojson::object>();
    for (picojson::value::object::const_iterator i = item_obj.begin(); i != item_obj.end(); ++i) {
      item_map[i->first] = i->second.to_str();
    }
    content.push_back(item_map);
  }
  message.content = content;

  // role
  std::string role_str;
  if (!json::ParseJSONField(json_obj, "role", role_str, err, true)) {
    return std::nullopt;
  }
  if (role_str == "system") {
    message.role = Role::system;
  } else if (role_str == "user") {
    message.role = Role::user;
  } else if (role_str == "assistant") {
    message.role = Role::assistant;
  } else if (role_str == "tool") {
    message.role = Role::tool;
  } else {
    *err += "Invalid role";
    return std::nullopt;
  }

  // name
  std::string name;
  if (json::ParseJSONField(json_obj, "name", name, err, false)) {
    message.name = name;
  }

  // tool calls
  picojson::array tool_calls_arr;
  if (json::ParseJSONField(json_obj, "tool_calls", tool_calls_arr, err, false)) {
    std::vector<ChatToolCall> tool_calls;
    for (const auto& item : tool_calls_arr) {
      if (!item.is<picojson::object>()) {
        *err += "Chat Tool Call item is not an object";
        return std::nullopt;
      }
      picojson::object item_obj = item.get<picojson::object>();
      std::optional<ChatToolCall> tool_call = ChatToolCall::FromJSON(item_obj, err);
      if (!tool_call.has_value()){
        return std::nullopt;
      };
      tool_calls.push_back(tool_call.value());
    }
    message.tool_calls = tool_calls; 
  }

  // tool call id
  std::string tool_call_id;
  if (json::ParseJSONField(json_obj, "tool_call_id", tool_call_id, err, false)) {
    message.tool_call_id = tool_call_id;
  }

  return message;
}

std::optional<ChatCompletionRequest> ChatCompletionRequest::FromJSON(
    const picojson::object& json_obj, std::string* err) {
  ChatCompletionRequest request;

  // messages
  picojson::array messages_arr;
  if (!json::ParseJSONField(json_obj, "messages", messages_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<ChatCompletionMessage> messages;
  for (const auto& item : messages_arr) {
    picojson::object item_obj = item.get<picojson::object>();
    std::optional<ChatCompletionMessage> message = ChatCompletionMessage::FromJSON(item_obj, err);
    if (!message.has_value()) {
      return std::nullopt;
    }
    messages.push_back(message.value());
  }
  request.messages = messages;

  // model
  std::string model;
  if (!json::ParseJSONField(json_obj, "model", model, err, true)) {
    return std::nullopt;
  }
  request.model = model;

  // frequency_penalty
  double frequency_penalty;
  if (json::ParseJSONField(json_obj, "frequency_penalty", frequency_penalty, err, false)) {
    request.frequency_penalty = frequency_penalty;
  }

  // presence_penalty
  double presence_penalty;
  if (json::ParseJSONField(json_obj, "presence_penalty", presence_penalty, err, false)) {
    request.presence_penalty = presence_penalty;
  }

  // tool_choice
  std::string tool_choice = "auto";
  request.tool_choice = tool_choice;
  if (json::ParseJSONField(json_obj, "tool_choice", tool_choice, err, false)) {
    request.tool_choice = tool_choice;
  }

  // tools
  picojson::array tools_arr;
  if (json::ParseJSONField(json_obj, "tools", tools_arr, err, false)) {
    std::vector<ChatTool> tools;
    for (const auto& item : tools_arr) {
      if (!item.is<picojson::object>()) {
        *err += "Chat Tool item is not an object";
        return std::nullopt;
      }
      picojson::object item_obj = item.get<picojson::object>();
      std::optional<ChatTool> tool = ChatTool::FromJSON(item_obj, err);
      if (!tool.has_value()){
        return std::nullopt;
      };
      tools.push_back(tool.value());
    }
    request.tools = tools;
  }

  // TODO: Other parameters

  return request;
}

std::optional<ChatCompletionRequest> ChatCompletionRequest::FromJSON(const std::string& json_str,
                                                                     std::string* err) {
  std::optional<picojson::object> json_obj = json::LoadJSONFromString(json_str, err);
  if (!json_obj.has_value()) {
    return std::nullopt;
  }
  return ChatCompletionRequest::FromJSON(json_obj.value(), err);
}

picojson::object ChatCompletionMessage::ToJSON() const {
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
      tool_calls_arr.push_back(picojson::value(tool_call.ToJSON()));
    }
    obj["tool_calls"] = picojson::value(tool_calls_arr);
  }
  return obj;
}


bool ChatCompletionRequest::check_function_calling(Conversation& conv_template, std::string* err) {
  if (!tools.has_value() || (tool_choice.has_value() && tool_choice.value() == "none")) {
      conv_template.use_function_calling = false;
      return true;
  }
  std::vector<ChatTool> tools_ = tools.value();
  std::string tool_choice_ = tool_choice.value();

  // TODO: support with tool choice as dict
  for (const auto& tool : tools_) {
    if (tool.function.name == tool_choice_) {
        conv_template.use_function_calling = true;
        picojson::value function_str(tool.function.ToJSON());
        conv_template.function_string = function_str.serialize();
        return true;
    }
  }

  if (tool_choice_ != "auto") {
    *err += "Invalid tool_choice value: " + tool_choice_;
    return false;
  }

  picojson::array function_list;
  for (const auto& tool : tools_) {
    function_list.push_back(picojson::value(tool.function.ToJSON()));
  }

  conv_template.use_function_calling = true;
  picojson::value function_list_json(function_list);
  conv_template.function_string = function_list_json.serialize();
  return true;
};


picojson::object ChatCompletionResponseChoice::ToJSON() const {
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
  obj["message"] = picojson::value(this->message.ToJSON());
  return obj;
}

picojson::object ChatCompletionStreamResponseChoice::ToJSON() const {
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
  obj["delta"] = picojson::value(this->delta.ToJSON());
  return obj;
}

picojson::object ChatCompletionResponse::ToJSON() const {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  picojson::array choices_arr;
  for (const auto& choice : this->choices) {
    choices_arr.push_back(picojson::value(choice.ToJSON()));
  }
  obj["choices"] = picojson::value(choices_arr);
  obj["created"] = picojson::value((int64_t)this->created);
  obj["model"] = picojson::value(this->model);
  obj["system_fingerprint"] = picojson::value(this->system_fingerprint);
  obj["object"] = picojson::value(this->object);
  return obj;
}

picojson::object ChatCompletionStreamResponse::ToJSON() const {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  picojson::array choices_arr;
  for (const auto& choice : this->choices) {
    choices_arr.push_back(picojson::value(choice.ToJSON()));
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