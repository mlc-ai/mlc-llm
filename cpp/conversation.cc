

#include "conversation.h"

#include <string>
#include <unordered_map>

namespace mlc {
namespace llm {

void Conversation::LoadJSONOverride(const picojson::value& config_json, bool partial_update) {
  std::string err_templ = " in conversion template json file.";
  picojson::object config = config_json.get<picojson::object>();
  if (config.count("name")) {
    CHECK(config["name"].is<std::string>()) << "Invalid name" << err_templ;
    this->name = config["name"].get<std::string>();
  } else {
    CHECK(partial_update) << "Key \"name\" not found.";
  }
  if (config.count("system")) {
    CHECK(config["system"].is<std::string>()) << "Invalid system" << err_templ;
    this->system = config["system"].get<std::string>();
  } else {
    CHECK(partial_update) << "Key \"system\" not found.";
  }
  if (config.count("roles")) {
    CHECK(config["roles"].is<picojson::array>()) << "Invalid roles" << err_templ;
    picojson::array roles_arr = config["roles"].get<picojson::array>();
    std::vector<std::string> roles;
    for (const picojson::value& v : roles_arr) {
      CHECK(v.is<std::string>()) << "Invalid roles" << err_templ;
      roles.push_back(v.get<std::string>());
    }
    this->roles = roles;
  } else {
    CHECK(partial_update) << "Key \"roles\" not found.";
  }
  if (config.count("messages")) {
    CHECK(config["messages"].is<picojson::array>()) << "Invalid messages" << err_templ;
    std::vector<std::vector<std::string>> messages;
    picojson::array msgs_arr = config["messages"].get<picojson::array>();
    for (const picojson::value& msgs_i : msgs_arr) {
      CHECK(msgs_i.is<picojson::array>()) << "Invalid messages" << err_templ;
      picojson::array msgs_i_arr = msgs_i.get<picojson::array>();
      std::vector<std::string> messages_i;
      for (const picojson::value& msg_v : msgs_i_arr) {
        CHECK(msg_v.is<std::string>()) << "Invalid messages" << err_templ;
        messages_i.push_back(msg_v.get<std::string>());
      }
      messages.push_back(messages_i);
    }
    this->messages = messages;
  } else {
    CHECK(partial_update) << "Key \"messages\" not found.";
  }
  if (config.count("offset")) {
    CHECK(config["offset"].is<int64_t>()) << "Invalid offset" << err_templ;
    this->offset = config["offset"].get<int64_t>();
  } else {
    this->offset = this->messages.size();
  }
  if (config.count("separator_style")) {
    CHECK(config["separator_style"].is<int64_t>()) << "Invalid separator style" << err_templ;
    this->separator_style = SeparatorStyle(config["separator_style"].get<int64_t>());
  } else {
    CHECK(partial_update) << "Key \"separator_style\" not found.";
  }
  if (config.count("seps")) {
    std::vector<std::string> seps;
    CHECK(config["seps"].is<picojson::array>()) << "Invalid seps" << err_templ;
    picojson::array seps_arr = config["seps"].get<picojson::array>();
    for (const picojson::value& sep : seps_arr) {
      CHECK(sep.is<std::string>()) << "Invalid seps" << err_templ;
      seps.push_back(sep.get<std::string>());
    }
    this->seps = seps;
  } else {
    CHECK(partial_update) << "Key \"seps\" not found.";
  }
  if (config.count("role_msg_sep")) {
    CHECK(config["role_msg_sep"].is<std::string>()) << "Invalid role_msg_sep" << err_templ;
    this->role_msg_sep = config["role_msg_sep"].get<std::string>();
  } else {
    CHECK(partial_update) << "Key \"role_msg_sep\" not found.";
  }
  if (config.count("role_empty_sep")) {
    CHECK(config["role_empty_sep"].is<std::string>()) << "Invalid role_empty_sep" << err_templ;
    this->role_empty_sep = config["role_empty_sep"].get<std::string>();
  } else {
    CHECK(partial_update) << "Key \"role_empty_sep\" not found.";
  }
  if (config.count("stop_str")) {
    CHECK(config["stop_str"].is<std::string>()) << "Invalid stop_str" << err_templ;
    this->stop_str = config["stop_str"].get<std::string>();
  } else {
    CHECK(partial_update) << "Key \"stop_str\" not found.";
  }
  if (config.count("stop_tokens")) {
    CHECK(config["stop_tokens"].is<picojson::array>()) << "Invalid stop_tokens" << err_templ;
    picojson::array stop_tokens_arr = config["stop_tokens"].get<picojson::array>();
    std::vector<int32_t> stop_tokens;
    for (const picojson::value& stop_token : stop_tokens_arr) {
      CHECK(stop_token.is<int64_t>()) << "Invalid stop_tokens" << err_templ;
      stop_tokens.push_back(stop_token.get<int64_t>());
    }
    this->stop_tokens = stop_tokens;
  } else {
    CHECK(partial_update) << "Key \"stop_tokens\" not found.";
  }
  if (config.count("prefix_tokens")) {
    CHECK(config["prefix_tokens"].is<picojson::array>()) << "Invalid prefix_tokens" << err_templ;
    picojson::array prefix_tokens_arr = config["prefix_tokens"].get<picojson::array>();
    std::vector<int32_t> prefix_tokens;
    for (const picojson::value& prefix_token : prefix_tokens_arr) {
      CHECK(prefix_token.is<int64_t>()) << "Invalid prefix_tokens" << err_templ;
      prefix_tokens.push_back(prefix_token.get<int64_t>());
    }
    this->prefix_tokens = prefix_tokens;
  } else {
    CHECK(partial_update) << "Key \"prefix_tokens\" not found.";
  }
  if (config.count("add_bos")) {
    CHECK(config["add_bos"].is<bool>()) << "Invalid add_bos" << err_templ;
    this->add_bos = config["add_bos"].get<bool>();
  } else {
    CHECK(partial_update) << "Key \"add_bos\" not found.";
  }
}

void Conversation::LoadJSONOverride(const std::string& config_str, bool partial_update) {
  picojson::value config_json;
  std::string err = picojson::parse(config_json, config_str);
  if (!err.empty()) {
    LOG(FATAL) << err;
    return;
  }
  LoadJSONOverride(config_json, partial_update);
}

picojson::value Conversation::SerializeToJSON() const {
  picojson::object config;
  config["name"] = picojson::value(this->name);
  config["system"] = picojson::value(this->system);
  picojson::array roles_arr;
  for (const std::string& role_str : this->roles) {
    roles_arr.push_back(picojson::value(role_str));
  }
  config["roles"] = picojson::value(roles_arr);
  picojson::array msgs_arr;
  for (const std::vector<std::string>& msgs_i : this->messages) {
    picojson::array msgs_i_arr;
    for (const std::string& msg_str : msgs_i) {
      msgs_i_arr.push_back(picojson::value(msg_str));
    }
    msgs_arr.push_back(picojson::value(msgs_i_arr));
  }
  config["messages"] = picojson::value(msgs_arr);
  config["offset"] = picojson::value((int64_t)this->offset);
  config["separator_style"] = picojson::value((int64_t)this->separator_style);
  picojson::array seps_arr;
  for (const std::string& sep_str : this->seps) {
    seps_arr.push_back(picojson::value(sep_str));
  }
  config["seps"] = picojson::value(seps_arr);
  config["role_msg_sep"] = picojson::value(role_msg_sep);
  config["role_empty_sep"] = picojson::value(role_empty_sep);
  config["stop_str"] = picojson::value(this->stop_str);
  picojson::array stop_tokens_arr;
  for (const int32_t& stop_token_str : this->stop_tokens) {
    stop_tokens_arr.push_back(picojson::value((int64_t)stop_token_str));
  }
  config["stop_tokens"] = picojson::value(stop_tokens_arr);
  picojson::array prefix_tokens_arr;
  for (const int32_t& prefix_token_str : this->prefix_tokens) {
    prefix_tokens_arr.push_back(picojson::value((int64_t)prefix_token_str));
  }
  config["prefix_tokens"] = picojson::value(prefix_tokens_arr);
  config["add_bos"] = picojson::value(this->add_bos);
  return picojson::value(config);
}

std::string Conversation::GetConfigJSON() const { return SerializeToJSON().serialize(true); }

}  // namespace llm
}  // namespace mlc
