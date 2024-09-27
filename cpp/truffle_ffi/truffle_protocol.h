/*!
 *  Copyright (c) 2023 by Contributors
 * \file json_ffi/openai_api_protocol.h
 * \brief The header of OpenAI API Protocol in MLC LLM.
 */
#ifndef MLC_LLM_TRUFFLE_FFI_OPENAI_API_PROTOCOL_H
#define MLC_LLM_TRUFFLE_FFI_OPENAI_API_PROTOCOL_H

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
namespace truffle_ffi {

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




class TruffleRequest {
  public:
   std::string id;
   std::string context;
   std::optional<int> max_tokens = std::nullopt;
   std::optional<std::vector<std::string>> stop = std::nullopt;
  
  
  bool stream = true;



  std::optional<double> temperature = std::nullopt;
  std::optional<double> top_p = std::nullopt;
  std::optional<double> frequency_penalty = std::nullopt;
  std::optional<double> presence_penalty = std::nullopt;

    static Result<TruffleRequest> FromJSON(const std::string& json_str);
};


class TruffleResponse{
  public:
   std::string id;
   std::string content;
   std::optional<FinishReason> finish_reason;
   std::optional<picojson::value> usage;
   picojson::object AsJSON() const;
};





}  // namespace 
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_OPENAI_API_PROTOCOL_H
