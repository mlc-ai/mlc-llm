/*!
 *  Copyright (c) 2023 by Contributors
 * \file json_ffi/openai_api_protocol.h
 * \brief The header of OpenAI API Protocol in MLC LLM.
 */
#ifndef MLC_LLM_JSON_FFI_OPENAI_API_PROTOCOL_H
#define MLC_LLM_JSON_FFI_OPENAI_API_PROTOCOL_H

#include <ctime>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "picojson.h"

namespace mlc {
namespace llm {
namespace json_ffi {

enum class Role { system, user, assistant, tool };
enum class Type { text, json_object, function };
enum class FinishReason { stop, length, tool_calls, error };

// TODO: Implement the following class
class ChatFunction {
 public:
  std::optional<std::string> description = std::nullopt;
  std::string name;
  std::unordered_map<std::string, std::string>
      parameters;  // Assuming parameters are string key-value pairs

  static std::optional<ChatFunction> FromJSON(const picojson::value& json, std::string* err);
};

// TODO: Implement the following class
class ChatTool {
 public:
  Type type = Type::function;
  ChatFunction function;

  static std::optional<ChatTool> FromJSON(const picojson::value& json, std::string* err);
};

// TODO: Implement the following class
class ChatFunctionCall {
 public:
  std::string name;
  std::optional<std::unordered_map<std::string, std::string>> arguments =
      std::nullopt;  // Assuming arguments are string key-value pairs
};

// TODO: Implement the following class
class ChatToolCall {
 public:
  std::string id;  // TODO: python code initializes this to an random string
  Type type = Type::function;
  ChatFunctionCall function;
};

class ChatCompletionMessage {
 public:
  std::optional<std::vector<std::unordered_map<std::string, std::string>>> content =
      std::nullopt;  // Assuming content is a list of string key-value pairs
  Role role;
  std::optional<std::string> name = std::nullopt;
  std::optional<std::vector<ChatToolCall>> tool_calls = std::nullopt;  // TODO: Implement this
  std::optional<std::string> tool_call_id = std::nullopt;              // TODO: Implement this

  static std::optional<ChatCompletionMessage> FromJSON(const picojson::value& json,
                                                       std::string* err);
  picojson::object ToJSON();
};

class RequestResponseFormat {
 public:
  Type type = Type::text;
  std::optional<std::string> json_schema = std::nullopt;
};

class ChatCompletionRequest {
 public:
  std::vector<ChatCompletionMessage> messages;
  std::string model;
  double frequency_penalty = 0.0;
  double presence_penalty = 0.0;
  bool logprobs = false;
  int top_logprobs = 0;
  std::optional<std::unordered_map<int, double>> logit_bias = std::nullopt;
  std::optional<int> max_tokens = std::nullopt;
  int n = 1;
  std::optional<int> seed = std::nullopt;
  std::optional<std::vector<std::string>> stop = std::nullopt;
  bool stream = false;
  double temperature = 1.0;
  double top_p = 1.0;
  std::optional<std::vector<ChatTool>> tools = std::nullopt;
  std::optional<std::string> tool_choice = std::nullopt;
  std::optional<std::string> user = std::nullopt;
  bool ignore_eos = false;
  //   RequestResponseFormat response_format; //TODO: implement this

  /*!
   * \brief Create a ChatCompletionRequest instance from the given JSON object.
   * When creation fails, errors are dumped to the input error string, and nullopt is returned.
   */
  static std::optional<ChatCompletionRequest> FromJSON(const picojson::object& json_obj,
                                                       std::string* err);
  /*!
   * \brief Parse and create a ChatCompletionRequest instance from the given JSON string.
   * When creation fails, errors are dumped to the input error string, and nullopt is returned.
   */
  static std::optional<ChatCompletionRequest> FromJSON(const std::string& json_str,
                                                       std::string* err);

  // TODO: check_penalty_range, check_logit_bias, check_logprobs
};

class ChatCompletionResponseChoice {
 public:
  std::optional<FinishReason> finish_reason;
  int index = 0;
  ChatCompletionMessage message;
  // TODO: logprobs

  picojson::object ToJSON();
};

class ChatCompletionStreamResponseChoice {
 public:
  std::optional<FinishReason> finish_reason;
  int index = 0;
  ChatCompletionMessage delta;
  // TODO: logprobs

  picojson::object ToJSON();
};

class ChatCompletionResponse {
 public:
  std::string id;
  std::vector<ChatCompletionResponseChoice> choices;
  int created = static_cast<int>(std::time(nullptr));
  std::string model;
  std::string system_fingerprint;
  std::string object = "chat.completion";
  // TODO: usage_info

  picojson::object ToJSON();
};

class ChatCompletionStreamResponse {
 public:
  std::string id;
  std::vector<ChatCompletionStreamResponseChoice> choices;
  int created = static_cast<int>(std::time(nullptr));
  std::string model;
  std::string system_fingerprint;
  std::string object = "chat.completion.chunk";

  picojson::object ToJSON();
};

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_OPENAI_API_PROTOCOL_H
