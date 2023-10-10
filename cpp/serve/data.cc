/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/data.cc
 */
#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(DataNode);

TVM_REGISTER_OBJECT_TYPE(TextDataNode);

TextData::TextData(String text) {
  ObjectPtr<TextDataNode> n = make_object<TextDataNode>();
  n->text = std::move(text);
  data_ = std::move(n);
}

TVM_REGISTER_OBJECT_TYPE(TokenDataNode);

TokenData::TokenData(ShapeTuple token_ids) {
  ObjectPtr<TokenDataNode> n = make_object<TokenDataNode>();
  n->token_ids = std::move(token_ids);
  data_ = std::move(n);
}

TokenData::TokenData(std::vector<int32_t> token_ids) {
  ObjectPtr<TokenDataNode> n = make_object<TokenDataNode>();
  n->token_ids = ShapeTuple(token_ids.begin(), token_ids.end());
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
