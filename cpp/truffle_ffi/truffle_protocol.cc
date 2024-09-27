/*!
 *  Copyright (c) 2023 by Contributors
 * \file json_ffi/openai_api_protocol.cc
 * \brief The implementation of OpenAI API Protocol in MLC LLM.
 */
#include "truffle_protocol.h"

#include "../support/json_parser.h"

namespace mlc {
namespace llm {
namespace truffle_ffi {


Result<TruffleRequest> TruffleRequest::FromJSON(const std::string& json_str) {
  using TResult = Result<TruffleRequest>;
  Result<picojson::object> json_obj_res = json::ParseToJSONObjectWithResultReturn(json_str);
  if (json_obj_res.IsErr()) {
    return TResult::Error(json_obj_res.UnwrapErr());
  }
  picojson::object json_obj = json_obj_res.Unwrap();
  TruffleRequest request;

  Result<std::string> context_str =
      json::LookupWithResultReturn<std::string>(json_obj, "context");
  if (context_str.IsErr()) {
    return TResult::Error(context_str.UnwrapErr());
  }
 
  request.context = context_str.Unwrap();
  

  // temperature
  Result<std::optional<double>> temperature_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "temperature");
  if (temperature_res.IsErr()) {
    return TResult::Error(temperature_res.UnwrapErr());
  }
  request.temperature = temperature_res.Unwrap();
  // top_p
  Result<std::optional<double>> top_p_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "top_p");
  if (top_p_res.IsErr()) {
    return TResult::Error(top_p_res.UnwrapErr());
  }
  request.top_p = top_p_res.Unwrap();
  // max_tokens
  Result<std::optional<int64_t>> max_tokens_res =
      json::LookupOptionalWithResultReturn<int64_t>(json_obj, "max_tokens");
  if (max_tokens_res.IsErr()) {
    return TResult::Error(max_tokens_res.UnwrapErr());
  }
  request.max_tokens = max_tokens_res.Unwrap();

  // frequency_penalty
  Result<std::optional<double>> frequency_penalty_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "frequency_penalty");
  if (frequency_penalty_res.IsErr()) {
    return TResult::Error(frequency_penalty_res.UnwrapErr());
  }
  request.frequency_penalty = frequency_penalty_res.Unwrap();
  // presence_penalty
  Result<std::optional<double>> presence_penalty_res =
      json::LookupOptionalWithResultReturn<double>(json_obj, "presence_penalty");
  if (presence_penalty_res.IsErr()) {
    return TResult::Error(presence_penalty_res.UnwrapErr());
  }
  request.presence_penalty = presence_penalty_res.Unwrap();
  

  // stop strings
  Result<std::optional<picojson::array>> stop_strs_res =
      json::LookupOptionalWithResultReturn<picojson::array>(json_obj, "stop");
  if (stop_strs_res.IsErr()) {
    return TResult::Error(stop_strs_res.UnwrapErr());
  }
  std::optional<picojson::array> stop_strs = stop_strs_res.Unwrap();
  if (stop_strs.has_value()) {
    std::vector<std::string> stop;
    for (picojson::value stop_str_value : stop_strs.value()) {
      if (!stop_str_value.is<std::string>()) {
        return TResult::Error("One given value in field \"stop\" is not a string.");
      }
      stop.push_back(stop_str_value.get<std::string>());
    }
    request.stop = std::move(stop);
  }

  

  // TODO: Other parameters
  return TResult::Ok(request);
}

picojson::object TruffleResponse::AsJSON() const {
  picojson::object obj;
  obj["id"] = picojson::value(this->id);
  obj["content"] = picojson::value(this->content);


  if (!this->finish_reason.has_value()) {
    obj["finish_reason"] = picojson::value();
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
  if (usage.has_value()) {
    obj["usage"] = usage.value();
  }
  return obj;
}



}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
