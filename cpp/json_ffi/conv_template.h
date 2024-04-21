#ifndef MLC_LLM_JSON_FFI_OPENAI_API_CONV_TEMPLATE_H
#define MLC_LLM_JSON_FFI_OPENAI_API_CONV_TEMPLATE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>
#include <typeinfo>

#include "../serve/data.h"
#include "picojson.h"

using namespace mlc::llm::serve;

enum class MessagePlaceholders {
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL,
    FUNCTION
};

MessagePlaceholders message_placeholder_from_string(const std::string& role);

namespace mlc{
namespace llm{
namespace json_ffi{
struct Conversation {
    std::optional<std::string> name = std::nullopt;
    std::string system_template;
    std::string system_message;
    std::optional<std::vector<int>> system_prefix_token_ids = std::nullopt;
    bool add_role_after_system_message = true;
    std::unordered_map<std::string, std::string> roles;
    std::unordered_map<std::string, std::string> role_templates;
    std::vector<std::pair<std::string, std::optional<std::vector<std::unordered_map<std::string, std::string>>>>> messages;
    std::vector<std::string> seps;
    std::string role_content_sep;
    std::string role_empty_sep;
    std::vector<std::string> stop_str;
    std::vector<int> stop_token_ids;
    std::optional<std::string> function_string = std::nullopt;
    std::optional<bool> use_function_calling = false;

    Conversation();

    static std::vector<std::string> check_message_seps(std::vector<std::string> seps);

    std::optional<std::vector<Data>> as_prompt(std::string* err);

    static std::optional<Conversation> FromJSON(const picojson::object& json, std::string* err);
    static std::optional<Conversation> FromJSON(const std::string& json_str, std::string* err);
};

}
}
}

#endif /* MLC_LLM_JSON_FFI_OPENAI_API_CONV_TEMPLATE_H */