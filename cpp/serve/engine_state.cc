/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_state.cc
 */
#include "engine_state.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_FFI_STATIC_INIT_BLOCK() { EngineStateObj::RegisterReflection(); }

EngineState::EngineState() { data_ = tvm::ffi::make_object<EngineStateObj>(); }

void EngineStateObj::Reset() {
  running_queue.clear();
  waiting_queue.clear();
  request_states.clear();
  id_manager.Reset();
  metrics.Reset();
  if (prefix_cache.defined()) {
    prefix_cache->Reset();
  }
  running_rsentries_changed = true;
  postproc_workspace = ActionPostProcessWorkspace();
}

RequestState EngineStateObj::GetRequestState(Request request) {
  TVM_FFI_ICHECK(request->rstate != nullptr) << "The state of the request has not been defined.";
  return GetRef<RequestState>(static_cast<RequestStateNode*>(request->rstate));
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
