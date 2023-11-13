/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/request_state.cc
 */

#include "request_state.h"

#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** RequestModelState ******************/

TVM_REGISTER_OBJECT_TYPE(RequestModelStateNode);

RequestModelState::RequestModelState(int model_id, Array<Data> inputs) {
  ObjectPtr<RequestModelStateNode> n = make_object<RequestModelStateNode>();
  n->model_id = model_id;
  n->request_id = -1;
  n->inputs = std::move(inputs);
  data_ = std::move(n);
}

TVM_REGISTER_OBJECT_TYPE(RequestStateNode);

RequestState::RequestState(int num_models, Array<Data> inputs, int raw_input_length) {
  ObjectPtr<RequestStateNode> n = make_object<RequestStateNode>();
  Array<RequestModelState> mstates;
  mstates.reserve(num_models);
  for (int i = 0; i < num_models; ++i) {
    mstates.push_back(RequestModelState(i, inputs));
  }
  n->mstates = std::move(mstates);
  n->raw_input_length = raw_input_length;
  n->tadd = std::chrono::high_resolution_clock::now();
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
