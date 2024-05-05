#ifndef MLC_LLM_JSON_FFI_CONV_TEMPLATE_H
#define MLC_LLM_JSON_FFI_CONV_TEMPLATE_H

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <typeinfo>
#include <variant>
#include <vector>

#include "../serve/data.h"
#include "../support/result.h"
#include "picojson.h"

using namespace mlc::llm::serve;

namespace mlc {
namespace llm {
namespace json_ffi {

/****************** Conversation template ******************/

enum class MessagePlaceholders { SYSTEM, USER, ASSISTANT, TOOL, FUNCTION };

MessagePlaceholders MessagePlaceholderFromString(const std::string& role);

class Message {
 public:
  std::string role;
  std::optional<std::vector<std::unordered_map<std::string, std::string>>> content = std::nullopt;
};

/**
 * @brief A struct that specifies the convention template of conversation
 * and contains the conversation history.
 */
struct Conversation {
  // Optional name of the template.
  std::optional<std::string> name = std::nullopt;

  // The system prompt template, it optionally contains the system
  // message placeholder, and the placeholder will be replaced with
  // the system message below.
  std::string system_template;

  // The content of the system prompt (without the template format).
  std::string system_message;

  // The system token ids to be prepended at the beginning of tokenized
  // generated prompt.
  std::optional<std::vector<int>> system_prefix_token_ids = std::nullopt;

  // Whether or not to append user role and separator after the system message.
  // This is mainly for [INST] [/INST] style prompt format
  bool add_role_after_system_message = true;

  // The conversation roles
  std::unordered_map<std::string, std::string> roles;

  // The roles prompt template, it optionally contains the defaults
  // message placeholders and will be replaced by actual content
  std::unordered_map<std::string, std::string> role_templates;

  // The conversation history messages.
  // Each message is a pair of strings, denoting "(role, content)".
  // The content can be None.
  std::vector<Message> messages;

  // The separators between messages when concatenating into a single prompt.
  // List size should be either 1 or 2.
  // - When size is 1, the separator will be used between adjacent messages.
  // - When size is 2, seps[0] is used after user message, and
  //   seps[1] is used after assistant message.
  std::vector<std::string> seps;

  // The separator between the role and the content in a message.
  std::string role_content_sep;

  // The separator between the role and empty contents.
  std::string role_empty_sep;

  // The stop criteria
  std::vector<std::string> stop_str;
  std::vector<int> stop_token_ids;

  // Function call fields
  // whether using function calling or not, helps check for output message format in API call
  std::optional<std::string> function_string = std::nullopt;
  bool use_function_calling = false;

  Conversation();

  /*! \brief Create the list of prompts from the messages based on the conversation template. */
  Result<std::vector<Data>> AsPrompt();

  /*! \brief Create a Conversation instance from the given JSON object. */
  static Result<Conversation> FromJSON(const picojson::object& json);
  /*! \brief Parse and create a Conversation instance from the given JSON string. */
  static Result<Conversation> FromJSON(const std::string& json_str);
};

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_CONV_TEMPLATE_H
