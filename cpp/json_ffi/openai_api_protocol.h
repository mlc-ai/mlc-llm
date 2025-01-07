/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file json_ffi/openai_api_protocol.h
 * \brief The header of OpenAI API Protocol in MLC LLM.
 */
#ifndef MLC_LLM_JSON_FFI_OPENAI_API_PROTOCOL_H
#define MLC_LLM_JSON_FFI_OPENAI_API_PROTOCOL_H

#include <ctime>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../serve/config.h"
#include "../support/result.h"
#include "picojson.h"

namespace mlc {
namespace llm {
namespace json_ffi {

using serve::DebugConfig;
using serve::ResponseFormat;

enum class Type { text, json_object, function };
enum class FinishReason { stop, length, tool_calls, error };

inline std::string GenerateUUID(size_t length) {
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

class ChatFunction {
 public:
  std::optional<std::string> description = std::nullopt;
  std::string name;
  // Todo: change to std::vector<std::pair<std::string, std::string>>?
  std::unordered_map<std::string, std::string>
      parameters;  // Assuming parameters are string key-value pairs

  static Result<ChatFunction> FromJSON(const picojson::object& json);
  picojson::object AsJSON() const;
};

class ChatTool {
 public:
  Type type = Type::function;
  ChatFunction function;

  static Result<ChatTool> FromJSON(const picojson::object& json);
  picojson::object AsJSON() const;
};

class ChatFunctionCall {
 public:
  std::string name;
  std::optional<std::unordered_map<std::string, std::string>> arguments =
      std::nullopt;  // Assuming arguments are string key-value pairs

  static Result<ChatFunctionCall> FromJSON(const picojson::object& json);
  picojson::object AsJSON() const;
};

class ChatToolCall {
 public:
  std::string id = "call_" + GenerateUUID(8);
  Type type = Type::function;
  ChatFunctionCall function;

  static Result<ChatToolCall> FromJSON(const picojson::object& json);
  picojson::object AsJSON() const;
};

class ChatCompletionMessageContent {
 public:
  ChatCompletionMessageContent() = default;

  ChatCompletionMessageContent(std::nullopt_t) {}  // NOLINT(*)

  ChatCompletionMessageContent(std::string text) : text_(text) {}  // NOLINT(*)

  ChatCompletionMessageContent(
      std::vector<std::unordered_map<std::string, std::string>> parts)  // NOLINT(*)
      : parts_(parts) {}

  bool IsNull() const { return !IsText() && !IsParts(); }

  bool IsText() const { return text_.operator bool(); }

  bool IsParts() const { return parts_.operator bool(); }

  const std::string& Text() const { return text_.value(); }

  const std::vector<std::unordered_map<std::string, std::string>>& Parts() const {
    return parts_.value();
  }

 private:
  /*! \brief used to store text content */
  std::optional<std::string> text_;
  std::optional<std::vector<std::unordered_map<std::string, std::string>>> parts_;
};

class ChatCompletionMessage {
 public:
  ChatCompletionMessageContent content =
      std::nullopt;  // Assuming content is a list of string key-value pairs
  std::string role;
  std::optional<std::string> name = std::nullopt;
  std::optional<std::vector<ChatToolCall>> tool_calls = std::nullopt;
  std::optional<std::string> tool_call_id = std::nullopt;

  static Result<ChatCompletionMessage> FromJSON(const picojson::object& json);
  picojson::object AsJSON() const;
};

class ChatCompletionRequest {
 public:
  std::vector<ChatCompletionMessage> messages;
  std::optional<std::string> model = std::nullopt;
  std::optional<double> frequency_penalty = std::nullopt;
  std::optional<double> presence_penalty = std::nullopt;
  bool logprobs = false;
  int top_logprobs = 0;
  std::optional<std::vector<std::pair<int, float>>> logit_bias = std::nullopt;
  std::optional<int> max_tokens = std::nullopt;
  int n = 1;
  std::optional<int> seed = std::nullopt;
  std::optional<std::vector<std::string>> stop = std::nullopt;
  bool stream = false;
  std::optional<double> temperature = std::nullopt;
  std::optional<double> top_p = std::nullopt;
  std::optional<std::vector<ChatTool>> tools = std::nullopt;
  std::optional<std::string> tool_choice = std::nullopt;
  std::optional<std::string> user = std::nullopt;
  bool ignore_eos = false;
  std::optional<ResponseFormat> response_format = std::nullopt;
  std::optional<DebugConfig> debug_config = std::nullopt;

  /*! \brief Parse and create a ChatCompletionRequest instance from the given JSON string. */
  static Result<ChatCompletionRequest> FromJSON(const std::string& json_str);

  // TODO: check_penalty_range, check_logit_bias, check_logprobs
};

class ChatCompletionResponseChoice {
 public:
  std::optional<FinishReason> finish_reason;
  int index = 0;
  ChatCompletionMessage message;
  // TODO: logprobs

  picojson::object AsJSON() const;
};

class ChatCompletionStreamResponseChoice {
 public:
  std::optional<FinishReason> finish_reason;
  int index = 0;
  ChatCompletionMessage delta;
  // TODO: logprobs

  picojson::object AsJSON() const;
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

  picojson::object AsJSON() const;
};

class ChatCompletionStreamResponse {
 public:
  std::string id;
  std::vector<ChatCompletionStreamResponseChoice> choices;
  int created = static_cast<int>(std::time(nullptr));
  std::string model;
  std::string system_fingerprint;
  std::string object = "chat.completion.chunk";
  std::optional<picojson::value> usage;

  picojson::object AsJSON() const;
};

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_OPENAI_API_PROTOCOL_H
