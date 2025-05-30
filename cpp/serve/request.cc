/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/request.cc
 */

#include "request.h"

#include <tvm/ffi/function.h>

#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** Request ******************/

TVM_REGISTER_OBJECT_TYPE(RequestNode);

Request::Request(String id, Array<Data> inputs, GenerationConfig generation_cfg) {
  if (generation_cfg->debug_config.special_request == SpecialRequestKind::kNone) {
    CHECK(!inputs.empty()) << "No input data is given.";
  }
  // Compute the total input length, or fall back to "-1" which means
  // unknown due to the existence of untokenized data.
  int prompt_tokens = 0;
  for (Data input : inputs) {
    if (const auto* token_data = input.as<TokenDataNode>()) {
      prompt_tokens += token_data->token_ids.size();
    } else if (const auto* image_data = input.as<ImageDataNode>()) {
      prompt_tokens += image_data->GetLength();
    } else {
      prompt_tokens = -1;
      break;
    }
  }

  ObjectPtr<RequestNode> n = make_object<RequestNode>();
  n->id = std::move(id);
  n->inputs = std::move(inputs);
  n->prompt_tokens = prompt_tokens;
  n->generation_cfg = std::move(generation_cfg);
  data_ = std::move(n);
}

Request Request::FromUntokenized(const Request& request, const Tokenizer& tokenizer) {
  bool has_untokenized_input = false;
  Array<Data> inputs;
  inputs.reserve(request->inputs.size());
  // Tokenize all text inputs.
  for (Data input : request->inputs) {
    if (const auto* text_data = input.as<TextDataNode>()) {
      has_untokenized_input = true;
      std::vector<int> token_ids = tokenizer->Encode(text_data->text);
      inputs.push_back(TokenData(token_ids));
    } else {
      inputs.push_back(input);
    }
  }

  // If there is no untokenized input, we don't need to create a new request.
  if (!has_untokenized_input) {
    ICHECK_NE(request->prompt_tokens, -1);
    return request;
  } else {
    return Request(request->id, std::move(inputs), request->generation_cfg);
  }
}

TVM_FFI_REGISTER_GLOBAL("mlc.serve.RequestGetInputs").set_body_typed([](Request request) {
  return request->inputs;
});

TVM_FFI_REGISTER_GLOBAL("mlc.serve.RequestGetGenerationConfigJSON")
    .set_body_typed([](Request request) {
      return picojson::value(request->generation_cfg->AsJSON()).serialize();
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
