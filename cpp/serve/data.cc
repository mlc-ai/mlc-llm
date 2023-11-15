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

NDArray TextDataNode::GetEmbedding(Model model) const {
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

NDArray TokenDataNode::GetEmbedding(Model model) const { return model->TokenEmbed(token_ids); }

TVM_REGISTER_GLOBAL("mlc.serve.TokenData").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<int32_t> token_ids;
  token_ids.reserve(args.size());
  for (int i = 0; i < args.size(); i++) {
    token_ids.push_back(args[i]);
  }
  *rv = TokenData(std::move(token_ids));
});

TVM_REGISTER_GLOBAL("mlc.serve.TokenDataGetLength").set_body_typed([](TokenData data) {
  return static_cast<int64_t>(data->token_ids.size());
});

TVM_REGISTER_GLOBAL("mlc.serve.TokenDataGetElem").set_body_typed([](TokenData data, int idx) {
  ICHECK_LT(idx, static_cast<int>(data->token_ids.size()));
  return data->token_ids[idx];
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
