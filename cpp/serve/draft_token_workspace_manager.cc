/*!
 * Copyright (c) 2023-2025 by Contributors
 * \file serve/draft_token_workspace_manager.cc
 */

#include "draft_token_workspace_manager.h"

#include "model.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_FFI_STATIC_INIT_BLOCK() { DraftTokenWorkspaceManagerObj::RegisterReflection(); }

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
  TVM_FFI_ICHECK_LE(num_slots, free_slots_.size());
  result->assign(free_slots_.rbegin(), free_slots_.rbegin() + num_slots);
  free_slots_.resize(free_slots_.size() - num_slots);
  for (int slot : (*result)) {
    ref_count_[slot] = 1;
  }
}

void DraftTokenWorkspaceManagerObj::AllocSlots(int num_slots,
                                               const std::vector<int>& initial_ref_count,
                                               std::vector<int>* result) {
  TVM_FFI_ICHECK_LE(num_slots, free_slots_.size());
  TVM_FFI_ICHECK_EQ(num_slots, initial_ref_count.size());
  result->assign(free_slots_.rbegin(), free_slots_.rbegin() + num_slots);
  free_slots_.resize(free_slots_.size() - num_slots);
  for (int i = 0; i < num_slots; ++i) {
    int slot = (*result)[i];
    TVM_FFI_ICHECK(initial_ref_count[i] > 0);
    ref_count_[slot] = initial_ref_count[i];
  }
}

void DraftTokenWorkspaceManagerObj::FreeSlots(const std::vector<int>& slots) {
  for (int slot : slots) {
    if (--ref_count_.at(slot) == 0) {
      free_slots_.push_back(slot);
      ref_count_.erase(slot);
    }
  }
}

void DraftTokenWorkspaceManagerObj::AllocWorkspace(ModelWorkspace* workspace,
                                                   bool require_hidden_states) {
  workspace->draft_probs =
      Tensor::Empty({max_num_tokens_, vocab_size_}, DataType::Float(32), device_);
  workspace->draft_probs_storage =
      Tensor::Empty({max_num_tokens_, vocab_size_}, DataType::Float(32), device_);
  if (require_hidden_states) {
    workspace->draft_hidden_states_storage = ft_.Empty(
        {max_num_tokens_, hidden_size_}, hidden_states_dtype_, device_, /*worker0_only=*/false);
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
