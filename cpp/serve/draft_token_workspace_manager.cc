/*!
 * Copyright (c) 2024 by Contributors
 * \file serve/draft_token_workspace_manager.cc
 */

#include "draft_token_workspace_manager.h"

#include "model.h"

namespace mlc {
namespace llm {
namespace serve {

DraftTokenWorkspaceManagerObj::DraftTokenWorkspaceManagerObj(int max_num_tokens, int vocab_size,
                                                             int hidden_size,
                                                             DLDataType hidden_states_dtype,
                                                             DLDevice device,
                                                             const FunctionTable& ft)
    : max_num_tokens_(max_num_tokens),
      vocab_size_(vocab_size),
      hidden_size_(hidden_size),
      hidden_states_dtype_(hidden_states_dtype),
      device_(device),
      ft_(ft) {
  free_slots_.resize(max_num_tokens);
  std::iota(free_slots_.begin(), free_slots_.end(), 0);
}

void DraftTokenWorkspaceManagerObj::AllocSlots(int num_slots, std::vector<int>* result) {
  ICHECK_LE(num_slots, free_slots_.size());
  result->assign(free_slots_.rbegin(), free_slots_.rbegin() + num_slots);
  std::vector<int> allocated(free_slots_.begin(), free_slots_.begin() + num_slots);
  free_slots_.resize(free_slots_.size() - num_slots);
}

void DraftTokenWorkspaceManagerObj::FreeSlots(const std::vector<int>& slots) {
  std::copy(slots.begin(), slots.end(), std::back_inserter(free_slots_));
}

void DraftTokenWorkspaceManagerObj::AllocWorkspace(ModelWorkspace* workspace,
                                                   bool require_hidden_states) {
  workspace->draft_probs =
      NDArray::Empty({max_num_tokens_, vocab_size_}, DataType::Float(32), device_);
  workspace->draft_probs_storage =
      NDArray::Empty({max_num_tokens_, vocab_size_}, DataType::Float(32), device_);
  if (require_hidden_states) {
    workspace->draft_hidden_states_storage =
        NDArray::Empty({max_num_tokens_, hidden_size_}, hidden_states_dtype_, device_);
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
