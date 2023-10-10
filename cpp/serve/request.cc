/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/request.cc
 */

#include "request.h"

#include <tvm/runtime/packed_func.h>

#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** Request ******************/

TVM_REGISTER_OBJECT_TYPE(RequestNode);

Request::Request(Array<Data> inputs, String generation_cfg_json, PackedFunc fcallback) {
  ObjectPtr<RequestNode> n = make_object<RequestNode>();
  n->inputs = std::move(inputs);
  n->generation_cfg = GenerationConfig(generation_cfg_json);
  n->fcallback = std::move(fcallback);
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
