/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_state.cc
 */
#include "engine_state.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(EngineStateObj);

EngineState::EngineState() { data_ = make_object<EngineStateObj>(); }

void EngineStateObj::Reset() {
  running_queue.clear();
  waiting_queue.clear();
  request_states.clear();
  id_manager.Reset();
  metrics.Reset();
  if (prefix_cache.defined()) {
    prefix_cache->Reset();
  }
}

RequestState EngineStateObj::GetRequestState(Request request) {
  auto it = request_states.find(request->id);
  ICHECK(it != request_states.end());
  return it->second;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
