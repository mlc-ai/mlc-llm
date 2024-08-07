/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/data.cc
 */
#include "data.h"

#include <tvm/runtime/registry.h>

#include "model.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** Data ******************/

TVM_REGISTER_OBJECT_TYPE(DataNode);

/****************** TextData ******************/

TVM_REGISTER_OBJECT_TYPE(TextDataNode);

TextData::TextData(String text) {
  ObjectPtr<TextDataNode> n = make_object<TextDataNode>();
  n->text = std::move(text);
  data_ = std::move(n);
}

int TextDataNode::GetLength() const {
  LOG(FATAL) << "\"GetLength\" for TextData is not supported. "
                "Please tokenize the text and construct a TokenData object.";
}

ObjectRef TextDataNode::GetEmbedding(Model model, ObjectRef* dst, int offset) const {
  LOG(FATAL) << "\"GetEmbedding\" for TextData is not supported. "
                "Please tokenize the text and construct a TokenData object.";
}

TVM_REGISTER_GLOBAL("mlc.serve.TextData").set_body_typed([](String text) {
  return TextData(std::move(text));
});

TVM_REGISTER_GLOBAL("mlc.serve.TextDataGetTextString").set_body_typed([](TextData data) {
  return data->text;
});

/****************** TokenData ******************/

TVM_REGISTER_OBJECT_TYPE(TokenDataNode);

TokenData::TokenData(IntTuple token_ids) {
  ObjectPtr<TokenDataNode> n = make_object<TokenDataNode>();
  n->token_ids = std::move(token_ids);
  data_ = std::move(n);
}

TokenData::TokenData(std::vector<int32_t> token_ids) {
  ObjectPtr<TokenDataNode> n = make_object<TokenDataNode>();
  n->token_ids = IntTuple(token_ids.begin(), token_ids.end());
  data_ = std::move(n);
}

int TokenDataNode::GetLength() const { return token_ids.size(); }

ObjectRef TokenDataNode::GetEmbedding(Model model, ObjectRef* dst, int offset) const {
  return model->TokenEmbed(token_ids, dst, offset);
}

TVM_REGISTER_GLOBAL("mlc.serve.TokenData").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<int32_t> token_ids;
  token_ids.reserve(args.size());
  for (int i = 0; i < args.size(); i++) {
    token_ids.push_back(args[i]);
  }
  *rv = TokenData(std::move(token_ids));
});

TVM_REGISTER_GLOBAL("mlc.serve.TokenDataGetTokenIds").set_body_typed([](TokenData data) {
  return data->token_ids;
});

/****************** ImageData ******************/

TVM_REGISTER_OBJECT_TYPE(ImageDataNode);

ImageData::ImageData(NDArray image, int embed_size) {
  ObjectPtr<ImageDataNode> n = make_object<ImageDataNode>();
  n->image = std::move(image);
  n->embed_size = embed_size;
  data_ = std::move(n);
}

int ImageDataNode::GetLength() const { return embed_size; }

ObjectRef ImageDataNode::GetEmbedding(Model model, ObjectRef* dst, int offset) const {
  return model->ImageEmbed(image, dst, offset);
}

TVM_REGISTER_GLOBAL("mlc.serve.ImageData").set_body_typed([](NDArray image, int embed_size) {
  return ImageData(std::move(image), embed_size);
});

TVM_REGISTER_GLOBAL("mlc.serve.ImageDataGetImage").set_body_typed([](ImageData data) {
  return data->image;
});

/****************** SampleResult ******************/

/*! \brief Convert a single token with probability to JSON string. */
inline void TokenToLogProbJSON(const Tokenizer& tokenizer, const TokenProbPair& token_prob,
                               std::ostringstream* os) {
  const std::string& token = tokenizer->PostProcessedTokenTable()[token_prob.first];

  (*os) << "\"token\": \"";
  for (char ch : token) {
    if (ch >= 33 && ch <= 126) {
      // The character is in ASCII visible range.
      // Handle escape characters in JSON.
      if (ch == '"') {
        (*os) << "\\\"";
      } else if (ch == '\\') {
        (*os) << "\\\\";
      } else {
        (*os) << ch;
      }
    }
  }
  (*os) << "\", ";
  (*os) << "\"logprob\": " << std::log(std::max(token_prob.second, 1e-10f)) << ", ";
  (*os) << "\"bytes\": [";
  int token_len = token.size();
  for (int pos = 0; pos < token_len; ++pos) {
    (*os) << static_cast<int>(static_cast<unsigned char>(token[pos]));
    if (pos != token_len - 1) {
      (*os) << ", ";
    }
  }
  (*os) << "]";
}

int32_t SampleResult::GetTokenId() const { return this->sampled_token_id.first; }

std::string SampleResult::GetLogProbJSON(const Tokenizer& tokenizer, bool logprob) const {
  ICHECK(top_prob_tokens.empty() || logprob);
  if (!logprob) {
    // Logprob is not needed.
    return "";
  }

  std::ostringstream os;
  os << "{";
  // - Convert the sampled token to JSON.
  TokenToLogProbJSON(tokenizer, sampled_token_id, &os);
  // - Convert the tokens with top probabilities.
  os << ", \"top_logprobs\": [";
  int num_top = top_prob_tokens.size();
  for (int i = 0; i < num_top; ++i) {
    os << "{";
    TokenToLogProbJSON(tokenizer, top_prob_tokens[i], &os);
    os << "}";
    if (i != num_top - 1) {
      os << ", ";
    }
  }
  os << "]}";
  return os.str();
}

/****************** RequestStreamOutput ******************/

TVM_REGISTER_OBJECT_TYPE(RequestStreamOutputObj);

RequestStreamOutput::RequestStreamOutput(
    String request_id, std::vector<std::vector<int64_t>> group_delta_token_ids,
    std::optional<std::vector<std::vector<String>>> group_delta_logprob_json_strs,
    std::vector<Optional<String>> group_finish_reason,
    std::vector<String> group_extra_prefix_string) {
  ObjectPtr<RequestStreamOutputObj> n = make_object<RequestStreamOutputObj>();
  n->request_id = std::move(request_id);
  n->group_delta_token_ids = std::move(group_delta_token_ids);
  n->group_delta_logprob_json_strs = std::move(group_delta_logprob_json_strs);
  n->group_finish_reason = std::move(group_finish_reason);
  n->group_extra_prefix_string = std::move(group_extra_prefix_string);
  data_ = std::move(n);
}

RequestStreamOutput RequestStreamOutput::Usage(String request_id,
                                               String request_final_usage_json_str) {
  ObjectPtr<RequestStreamOutputObj> n = make_object<RequestStreamOutputObj>();
  n->request_id = std::move(request_id);
  n->request_final_usage_json_str = std::move(request_final_usage_json_str);
  return RequestStreamOutput(n);
}

TVM_REGISTER_GLOBAL("mlc.serve.RequestStreamOutputUnpack")
    .set_body_typed([](RequestStreamOutput output) {
      CHECK(!output->unpacked) << "One RequestStreamOutput can be unpacked for at most once.";
      std::vector<IntTuple> group_delta_token_ids;
      std::vector<Array<String>> group_delta_logprob_json_strs;
      group_delta_token_ids.reserve(output->group_delta_token_ids.size());
      if (output->group_delta_logprob_json_strs.has_value()) {
        group_delta_logprob_json_strs.reserve(output->group_delta_token_ids.size());
      }
      for (int i = 0; i < static_cast<int>(output->group_delta_token_ids.size()); ++i) {
        group_delta_token_ids.push_back(output->group_delta_token_ids[i]);
        if (output->group_delta_logprob_json_strs.has_value()) {
          group_delta_logprob_json_strs.push_back(output->group_delta_logprob_json_strs.value()[i]);
        }
      }
      Array<ObjectRef> ret = {output->request_id,
                              Array<IntTuple>(std::move(group_delta_token_ids)),
                              output->group_delta_logprob_json_strs.has_value()
                                  ? Array<Array<String>>(std::move(group_delta_logprob_json_strs))
                                  : Optional<Array<Array<String>>>(),
                              Array<Optional<String>>(output->group_finish_reason),
                              output->request_final_usage_json_str,
                              Array<String>(output->group_extra_prefix_string)};
      output->unpacked = true;
      return ret;
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
