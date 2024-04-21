#include "conv_template.h"

#include "../metadata/json_parser.h"

namespace mlc {
namespace llm {
namespace json_ffi {

using namespace mlc::llm;

std::map<MessagePlaceholders, std::string> PLACEHOLDERS = {
    {MessagePlaceholders::SYSTEM, "{system_message}"},
    {MessagePlaceholders::USER, "{user_message}"},
    {MessagePlaceholders::ASSISTANT, "{assistant_message}"},
    {MessagePlaceholders::TOOL, "{tool_message}"},
    {MessagePlaceholders::FUNCTION, "{function_string}"}};

MessagePlaceholders messagePlaceholderFromString(const std::string& role) {
  static const std::unordered_map<std::string, MessagePlaceholders> enum_map = {
      {"system", MessagePlaceholders::SYSTEM},       {"user", MessagePlaceholders::USER},
      {"assistant", MessagePlaceholders::ASSISTANT}, {"tool", MessagePlaceholders::TOOL},
      {"function", MessagePlaceholders::FUNCTION},
  };

  return enum_map.at(role);
}

Conversation::Conversation()
    : role_templates({{"user", PLACEHOLDERS[MessagePlaceholders::USER]},
                      {"assistant", PLACEHOLDERS[MessagePlaceholders::ASSISTANT]},
                      {"tool", PLACEHOLDERS[MessagePlaceholders::TOOL]}}) {}

std::vector<std::string> Conversation::checkMessageSeps(std::vector<std::string>& seps) {
  if (seps.size() == 0 || seps.size() > 2) {
    throw std::invalid_argument("seps should have size 1 or 2.");
  }
  return seps;
}

std::optional<std::vector<Data>> Conversation::asPrompt(std::string* err) {
  // Get the system message
  std::string system_msg = system_template;
  size_t pos = system_msg.find(PLACEHOLDERS[MessagePlaceholders::SYSTEM]);
  if (pos != std::string::npos) {
    system_msg.replace(pos, PLACEHOLDERS[MessagePlaceholders::SYSTEM].length(),
                       this->system_message);
  }

  // Get the message strings
  std::vector<Data> message_list;
  std::vector<std::string> separators = seps;
  if (separators.size() == 1) {
    separators.push_back(separators[0]);
  }

  if (!system_msg.empty()) {
    system_msg += separators[0];
    message_list.push_back(TextData(system_message));
  }

  for (int i = 0; i < messages.size(); i++) {
    std::string role = messages[i].first;
    std::optional<std::vector<std::unordered_map<std::string, std::string>>> content =
        messages[i].second;
    if (roles.find(role) == roles.end()) {
      *err += "\nRole " + role + " is not supported. ";
      return std::nullopt;
    }

    std::string separator = separators[role == "assistant"];  // check assistant role

    // If content is empty, add the role and separator
    // assistant's turn to generate text
    if (!content.has_value()) {
      message_list.push_back(TextData(roles[role] + role_empty_sep));
      continue;
    }

    std::string message = "";
    std::string role_prefix = "";
    // Do not append role prefix if this is the first message and there
    // is already a system message
    if (add_role_after_system_message || system_msg.empty() || i != 0) {
      role_prefix = roles[role] + role_content_sep;
    }

    message += role_prefix;

    for (auto& item : content.value()) {
      if (item.find("type") == item.end()) {
        *err += "Content item should have a type field";
        return std::nullopt;
      }
      if (item["type"] == "text") {
        if (item.find("text") == item.end()) {
          *err += "Content item should have a text field";
          return std::nullopt;
        }
        // replace placeholder[ROLE] with input message from role
        std::string role_text = role_templates[role];
        std::string placeholder = PLACEHOLDERS[messagePlaceholderFromString(role)];
        size_t pos = role_text.find(placeholder);
        if (pos != std::string::npos) {
          role_text.replace(pos, placeholder.length(), item["text"]);
        }
        if (use_function_calling.has_value() && use_function_calling.value()) {
          // replace placeholder[FUNCTION] with function_string
          // this assumes function calling is used for a single request scenario only
          if (!function_string.has_value()) {
            *err += "Function string is required for function calling";
            return std::nullopt;
          }
          pos = role_text.find(PLACEHOLDERS[MessagePlaceholders::FUNCTION]);
          if (pos != std::string::npos) {
            role_text.replace(pos, PLACEHOLDERS[MessagePlaceholders::FUNCTION].length(),
                              function_string.value());
          }
        }
        message += role_text;
      } else {
        *err += "Unsupported content type: " + item["type"];
        return std::nullopt;
      }
    }

    message += separator;
    message_list.push_back(TextData(message));
  }

  std::vector<Data> prompt = message_list;

  return prompt;
}

std::optional<Conversation> Conversation::FromJSON(const picojson::object& json, std::string* err) {
  Conversation conv;

  // name
  std::string name;
  if (json::ParseJSONField(json, "name", name, err, false)) {
    conv.name = name;
  }

  std::string system_template;
  if (!json::ParseJSONField(json, "system_template", system_template, err, true)) {
    return std::nullopt;
  }
  conv.system_template = system_template;

  std::string system_message;
  if (!json::ParseJSONField(json, "system_message", system_message, err, true)) {
    return std::nullopt;
  }
  conv.system_message = system_message;

  picojson::array system_prefix_token_ids_arr;
  if (json::ParseJSONField(json, "system_prefix_token_ids", system_prefix_token_ids_arr, err,
                           false)) {
    std::vector<int> system_prefix_token_ids;
    for (const auto& token_id : system_prefix_token_ids_arr) {
      if (!token_id.is<int64_t>()) {
        *err += "system_prefix_token_ids should be an array of integers.";
        return std::nullopt;
      }
      system_prefix_token_ids.push_back(token_id.get<int64_t>());
    }
    conv.system_prefix_token_ids = system_prefix_token_ids;
  }

  bool add_role_after_system_message;
  if (!json::ParseJSONField(json, "add_role_after_system_message", add_role_after_system_message,
                            err, true)) {
    return std::nullopt;
  }
  conv.add_role_after_system_message = add_role_after_system_message;

  picojson::object roles_object;
  if (!json::ParseJSONField(json, "roles", roles_object, err, true)) {
    return std::nullopt;
  }
  std::unordered_map<std::string, std::string> roles;
  for (const auto& role : roles_object) {
    if (!role.second.is<std::string>()) {
      *err += "roles should be a map of string to string.";
      return std::nullopt;
    }
    roles[role.first] = role.second.get<std::string>();
  }
  conv.roles = roles;

  picojson::object role_templates_object;
  if (json::ParseJSONField(json, "role_templates", role_templates_object, err, false)) {
    for (const auto& role : role_templates_object) {
      if (!role.second.is<std::string>()) {
        *err += "role_templates should be a map of string to string.";
        return std::nullopt;
      }
      conv.role_templates[role.first] = role.second.get<std::string>();
    }
  }

  picojson::array messages_arr;
  if (!json::ParseJSONField(json, "messages", messages_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<std::pair<std::string,
                        std::optional<std::vector<std::unordered_map<std::string, std::string>>>>>
      messages;
  for (const auto& message : messages_arr) {
    if (!message.is<picojson::object>()) {
      *err += "messages should be an array of objects.";
      return std::nullopt;
    }
    picojson::object message_obj = message.get<picojson::object>();
    std::string role;
    if (!json::ParseJSONField(message_obj, "role", role, err, true)) {
      *err += "role field is required in messages.";
      return std::nullopt;
    }
    picojson::array content_arr;
    std::vector<std::unordered_map<std::string, std::string>> content;
    if (json::ParseJSONField(message_obj, "content", content_arr, err, false)) {
      for (const auto& item : content_arr) {
        if (!item.is<picojson::object>()) {
          *err += "Content item is not an object";
          return std::nullopt;
        }
        std::unordered_map<std::string, std::string> item_map;
        picojson::object item_obj = item.get<picojson::object>();
        for (picojson::value::object::const_iterator i = item_obj.begin(); i != item_obj.end();
             ++i) {
          item_map[i->first] = i->second.to_str();
        }
        content.push_back(item_map);
      }
    }
    messages.push_back({role, content});
  }
  conv.messages = messages;

  picojson::array seps_arr;
  if (!json::ParseJSONField(json, "seps", seps_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<std::string> seps;
  for (const auto& sep : seps_arr) {
    if (!sep.is<std::string>()) {
      *err += "seps should be an array of strings.";
      return std::nullopt;
    }
    seps.push_back(sep.get<std::string>());
  }
  conv.seps = seps;

  std::string role_content_sep;
  if (!json::ParseJSONField(json, "role_content_sep", role_content_sep, err, true)) {
    return std::nullopt;
  }
  conv.role_content_sep = role_content_sep;

  std::string role_empty_sep;
  if (!json::ParseJSONField(json, "role_empty_sep", role_empty_sep, err, true)) {
    return std::nullopt;
  }
  conv.role_empty_sep = role_empty_sep;

  picojson::array stop_str_arr;
  if (!json::ParseJSONField(json, "stop_str", stop_str_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<std::string> stop_str;
  for (const auto& stop : stop_str_arr) {
    if (!stop.is<std::string>()) {
      *err += "stop_str should be an array of strings.";
      return std::nullopt;
    }
    stop_str.push_back(stop.get<std::string>());
  }
  conv.stop_str = stop_str;

  picojson::array stop_token_ids_arr;
  if (!json::ParseJSONField(json, "stop_token_ids", stop_token_ids_arr, err, true)) {
    return std::nullopt;
  }
  std::vector<int> stop_token_ids;
  for (const auto& stop : stop_token_ids_arr) {
    if (!stop.is<int64_t>()) {
      *err += "stop_token_ids should be an array of integers.";
      return std::nullopt;
    }
    stop_token_ids.push_back(stop.get<int64_t>());
  }
  conv.stop_token_ids = stop_token_ids;

  std::string function_string;
  if (!json::ParseJSONField(json, "function_string", function_string, err, false)) {
    conv.function_string = function_string;
  }

  bool use_function_calling;
  if (json::ParseJSONField(json, "use_function_calling", use_function_calling, err, false)) {
    conv.use_function_calling = use_function_calling;
  }

  return conv;
}

std::optional<Conversation> Conversation::FromJSON(const std::string& json_str, std::string* err) {
  std::optional<picojson::object> json_obj = json::LoadJSONFromString(json_str, err);
  if (!json_obj.has_value()) {
    return std::nullopt;
  }
  return Conversation::FromJSON(json_obj.value(), err);
}
}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
