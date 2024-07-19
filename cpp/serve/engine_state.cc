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

const std::vector<RequestStateEntry>& EngineStateObj::GetRunningRequestStateEntries() {
  if (running_rsentries_changed) {
    cached_running_rsentries_.clear();
    for (const Request& request : running_queue) {
      for (const RequestStateEntry& rsentry : GetRequestState(request)->entries) {
        // One request entry is considered as running for decode if it is a leaf and has
        // finished all input prefill.
        if (rsentry->status == RequestStateStatus::kAlive && rsentry->child_indices.empty() &&
            rsentry->mstates[0]->inputs.empty()) {
          cached_running_rsentries_.push_back(rsentry);
        }
      }
    }
    running_rsentries_changed = false;
  }
  return cached_running_rsentries_;
  //
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
