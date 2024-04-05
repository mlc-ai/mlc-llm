#include "openai_api_protocol.h"

picojson::value LoadJsonFromString(const std::string& json_str, std::string& err) {
  picojson::value json;
  err = picojson::parse(json, json_str);
  return json;
}

template <typename T>
bool ParseJsonField(picojson::object& json_obj, const std::string& field, T& value,
                    std::string& err, bool required) {
  // T can be int, double, bool, string, picojson::array
  if (json_obj.count(field)) {
    if (!json_obj[field].is<T>()) {
      err += "Field " + field + " is not of type " + typeid(T).name() + "\n";
      return false;
    }
    value = json_obj[field].get<T>();
  } else {
    if (required) {
      err += "Field " + field + " is required\n";
      return false;
    }
  }
  return true;
}

std::optional<ChatCompletionMessage> ChatCompletionMessage::FromJSON(const picojson::value& json,
                                                                     std::string& err) {
  if (!json.is<picojson::object>()) {
    err += "Input is not a valid JSON object";
    return std::nullopt;
  }
  picojson::object json_obj = json.get<picojson::object>();

  ChatCompletionMessage message;

  // content
  picojson::array content_arr;
  if (!ParseJsonField(json_obj, "content", content_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<std::map<std::string, std::string> > content;
  for (const auto& item : content_arr) {
    if (!item.is<picojson::object>()) {
      err += "Content item is not an object";
      return std::nullopt;
    }
    std::map<std::string, std::string> item_map;
    picojson::object item_obj = item.get<picojson::object>();
    for (picojson::value::object::const_iterator i = item_obj.begin(); i != item_obj.end(); ++i) {
      item_map[i->first] = i->second.to_str();
    }
    content.push_back(item_map);
  }
  message.content = content;

  // role
  std::string role_str;
  if (!ParseJsonField(json_obj, "role", role_str, err, true)) {
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
    err += "Invalid role";
    return std::nullopt;
  }

  // name
  std::string name;
  if (ParseJsonField(json_obj, "name", name, err, false)) {
    message.name = name;
  }

  // TODO: tool_calls and tool_call_id

  return message;
}

std::optional<ChatCompletionRequest> ChatCompletionRequest::FromJSON(const picojson::value& json,
                                                                     std::string& err) {
  if (!json.is<picojson::object>()) {
    err += "Input is not a valid JSON object";
    return std::nullopt;
  }
  picojson::object json_obj = json.get<picojson::object>();

  ChatCompletionRequest request;

  // messages
  picojson::array messages_arr;
  if (!ParseJsonField(json_obj, "messages", messages_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<ChatCompletionMessage> messages;
  for (const auto& item : messages_arr) {
    std::optional<ChatCompletionMessage> message = ChatCompletionMessage::FromJSON(item, err);
    if (!message.has_value()) {
      return std::nullopt;
    }
    messages.push_back(message.value());
  }
  request.messages = messages;

  // model
  std::string model;
  if (!ParseJsonField(json_obj, "model", model, err, true)) {
    return std::nullopt;
  }
  request.model = model;

  // frequency_penalty
  double frequency_penalty;
  if (ParseJsonField(json_obj, "frequency_penalty", frequency_penalty, err, false)) {
    request.frequency_penalty = frequency_penalty;
  }

  // presence_penalty
  double presence_penalty;
  if (ParseJsonField(json_obj, "presence_penalty", presence_penalty, err, false)) {
    request.presence_penalty = presence_penalty;
  }

  // TODO: Other parameters

  return request;
}

std::optional<ChatCompletionRequest> ChatCompletionRequest::FromJSON(const std::string& json_str,
                                                                     std::string& err) {
  picojson::value json = LoadJsonFromString(json_str, err);
  return ChatCompletionRequest::FromJSON(json, err);
}

picojson::object ChatCompletionMessage::ToJSON() {
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
  if (name.has_value()) {
    obj["name"] = picojson::value(name.value());
  }
  return obj;
}

picojson::object ChatCompletionResponseChoice::ToJSON() {
  picojson::object obj;
  if (!this->finish_reason.has_value()) {
    obj["finish_reason"] = picojson::value("null");
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

picojson::object ChatCompletionStreamResponseChoice::ToJSON() {
  picojson::object obj;
  if (!this->finish_reason.has_value()) {
    obj["finish_reason"] = picojson::value("null");
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

picojson::object ChatCompletionResponse::ToJSON() {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  picojson::array choices_arr;
  for (auto& choice : this->choices) {
    choices_arr.push_back(picojson::value(choice.ToJSON()));
  }
  obj["choices"] = picojson::value(choices_arr);
  obj["created"] = picojson::value((int64_t)this->created);
  obj["model"] = picojson::value(this->model);
  obj["system_fingerprint"] = picojson::value(this->system_fingerprint);
  obj["object"] = picojson::value(this->object);
  return obj;
}

picojson::object ChatCompletionStreamResponse::ToJSON() {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  picojson::array choices_arr;
  for (auto& choice : this->choices) {
    choices_arr.push_back(picojson::value(choice.ToJSON()));
  }
  obj["choices"] = picojson::value(choices_arr);
  obj["created"] = picojson::value((int64_t)this->created);
  obj["model"] = picojson::value(this->model);
  obj["system_fingerprint"] = picojson::value(this->system_fingerprint);
  obj["object"] = picojson::value(this->object);
  return obj;
}
