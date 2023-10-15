/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/request.cc
 */

#include "request.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** Request ******************/

TVM_REGISTER_OBJECT_TYPE(RequestNode);

Request::Request(Array<Data> inputs, GenerationConfig generation_cfg, PackedFunc fcallback) {
  ObjectPtr<RequestNode> n = make_object<RequestNode>();
  n->inputs = std::move(inputs);
  n->generation_cfg = std::move(generation_cfg);
  n->fcallback = std::move(fcallback);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("mlc.serve.Request")
    .set_body_typed([](Array<Data> inputs, String generation_cfg_json, PackedFunc fcallback) {
      return Request(std::move(inputs), GenerationConfig(std::move(generation_cfg_json)),
                     std::move(fcallback));
    });

TVM_REGISTER_GLOBAL("mlc.serve.RequestGetInputs").set_body_typed([](Request request) {
  return request->inputs;
});

TVM_REGISTER_GLOBAL("mlc.serve.RequestGetGenerationConfigJSON").set_body_typed([](Request request) {
  return request->generation_cfg->AsJSONString();
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
