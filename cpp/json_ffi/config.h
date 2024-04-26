#ifndef MLC_LLM_JSON_FFI_CONFIG_H
#define MLC_LLM_JSON_FFI_CONFIG_H

#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <typeinfo>
#include <variant>
#include <vector>

#include "../serve/data.h"
#include "picojson.h"

using namespace mlc::llm::serve;

namespace mlc {
namespace llm {
namespace json_ffi {

/****************** Model-defined generation config ******************/

class ModelDefinedGenerationConfigNode : public Object {
 public:
  double temperature;
  double top_p;
  double frequency_penalty;
  double presence_penalty;

  static constexpr const char* _type_key = "mlc.json_ffi.ModelDefinedGenerationConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(ModelDefinedGenerationConfigNode, Object);
};

class ModelDefinedGenerationConfig : public ObjectRef {
 public:
  explicit ModelDefinedGenerationConfig(double temperature, double top_p, double frequency_penalty,
                                        double presence_penalty);

  TVM_DEFINE_OBJECT_REF_METHODS(ModelDefinedGenerationConfig, ObjectRef,
                                ModelDefinedGenerationConfigNode);
};

/****************** Conversation template ******************/

enum class MessagePlaceholders { SYSTEM, USER, ASSISTANT, TOOL, FUNCTION };

MessagePlaceholders messagePlaceholderFromString(const std::string& role);

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
  std::optional<bool> use_function_calling = false;

  Conversation();

  /**
   * @brief Checks the size of the separators vector.
   * This function checks if the size of the separators vector is either 1 or 2.
   * If the size is not 1 or 2, it throws an invalid_argument exception.
   */
  static std::vector<std::string> CheckMessageSeps(std::vector<std::string>& seps);

  /*!
   * \brief Create the list of prompts from the messages based on the conversation template.
   * When creation fails, errors are dumped to the input error string, and nullopt is returned.
   */
  std::optional<std::vector<Data>> AsPrompt(std::string* err);

  /*!
   * \brief Create a Conversation instance from the given JSON object.
   * When creation fails, errors are dumped to the input error string, and nullopt is returned.
   */
  static std::optional<Conversation> FromJSON(const picojson::object& json, std::string* err);

  /*!
   * \brief Parse and create a Conversation instance from the given JSON string.
   * When creation fails, errors are dumped to the input error string, and nullopt is returned.
   */
  static std::optional<Conversation> FromJSON(const std::string& json_str, std::string* err);
};

/****************** JSON FFI engine config ******************/

class JSONFFIEngineConfigNode : public Object {
 public:
  String conv_template;
  Map<String, ModelDefinedGenerationConfig> model_generation_cfgs;

  static constexpr const char* _type_key = "mlc.json_ffi.JSONFFIEngineConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(JSONFFIEngineConfigNode, Object);
};

class JSONFFIEngineConfig : public ObjectRef {
 public:
  explicit JSONFFIEngineConfig(String conv_template,
                               Map<String, ModelDefinedGenerationConfig> model_generation_cfgs);

  TVM_DEFINE_OBJECT_REF_METHODS(JSONFFIEngineConfig, ObjectRef, JSONFFIEngineConfigNode);
};

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif /* MLC_LLM_JSON_FFI_CONV_TEMPLATE_H */
