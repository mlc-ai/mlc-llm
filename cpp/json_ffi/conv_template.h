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
#include "openai_api_protocol.h"
#include "picojson.h"

using namespace mlc::llm::serve;

namespace mlc {
namespace llm {
namespace json_ffi {

/****************** Model vision config ******************/

/*! \brief Defines the Vision config of the model (if present) */
class ModelVisionConfig {
 public:
  int hidden_size;
  int image_size;
  int intermediate_size;
  int num_attention_heads;
  int num_hidden_layers;
  int patch_size;
  int projection_dim;
  int vocab_size;
  std::string dtype;
  int num_channels;
  double layer_norm_eps;

  static ModelVisionConfig FromJSON(const picojson::object& json_obj);
};

/****************** Model config ******************/

/*! \brief Defines the config of the model.
Populated from "model_config" field in mlc-chat-config.json */
class ModelConfig {
 public:
  int vocab_size;
  int context_window_size;
  int sliding_window_size;
  int prefill_chunk_size;
  int tensor_parallel_shards;
  int pipeline_parallel_stages;
  int max_batch_size;
  std::optional<ModelVisionConfig> vision_config = std::nullopt;

  static ModelConfig FromJSON(const picojson::object& json_obj);
};

/****************** Conversation template ******************/

enum class MessagePlaceholders { SYSTEM, USER, ASSISTANT, TOOL, FUNCTION };

MessagePlaceholders MessagePlaceholderFromString(const std::string& role);

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
  std::vector<ChatCompletionMessage> messages;

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

  Conversation();

  /*!
   * \brief Get the system text(with the prompt template) given the system prompt message
   * \param system_msg The system prompt message.
   * \return The created system text.
   */
  std::string GetSystemText(const std::string& system_msg) const;

  /*!
   * \brief replace the content from role by the correct role text in template
   * \param role The input role
   * \param content The input content from the role
   * \param fn_call_str The function calling string if any.
   * \return The created text.
   */
  std::string GetRoleText(const std::string& role, const std::string& content,
                          const std::optional<std::string>& fn_call_str) const;

  /*! \brief Create a Conversation instance from the given JSON object. */
  static Result<Conversation> FromJSON(const picojson::object& json);
  /*! \brief Parse and create a Conversation instance from the given JSON string. */
  static Result<Conversation> FromJSON(const std::string& json_str);
};

/*! \brief Create the list of prompts from the messages based on the conversation template. */
Result<std::vector<Data>> CreatePrompt(const Conversation& conv,
                                       const ChatCompletionRequest& request,
                                       const ModelConfig& config, DLDevice device);

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_CONV_TEMPLATE_H
