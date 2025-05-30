/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/draft_token_workspace_manager.h
 */

#ifndef MLC_LLM_SERVE_DRAFT_TOKEN_WORKSPACE_MANAGER_H_
#define MLC_LLM_SERVE_DRAFT_TOKEN_WORKSPACE_MANAGER_H_
#include <tvm/runtime/device_api.h>

#include <numeric>
#include <optional>
#include <vector>

#include "data.h"
#include "function_table.h"
namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

struct ModelWorkspace;

/*!
 * \brief Managing the workspace for draft token generation.
 *
 * The workspace is used to store the associated states for each draft token, including the
 * probability distribution of the draft token, the hidden states, etc. The workspace manager
 * maintains a pool of slots for the draft tokens to store the states.
 */
class DraftTokenWorkspaceManagerObj : public Object {
 public:
  /*!
   * \brief Constructor
   * \param max_num_tokens The maximum number of draft tokens that can be stored in the workspace.
   * \param vocab_size The size of the vocabulary.
   * \param hidden_size The size of the hidden states.
   * \param hidden_states_dtype The data type of the hidden states.
   * \param device The device running the model.
   * \param ft The function table.
   */
  DraftTokenWorkspaceManagerObj(int max_num_tokens, int vocab_size, int hidden_size,
                                DLDataType hidden_states_dtype, DLDevice device,
                                const FunctionTable& ft);

  /*!
   * \brief Allocate the workspace for draft tokens and update `ModelWorkspace` data structure.
   * \param workspace The object to stored the allocated draft token workspace.
   * \param require_hidden_states Whether to allocate workspace for the hidden states.
   */
  void AllocWorkspace(ModelWorkspace* workspace, bool require_hidden_states);

  /*!
   * \brief Allocate slots for the draft tokens.
   * \param num_slots The number of slots to allocate.
   * \param result The vector to store the allocated slots.
   */
  void AllocSlots(int num_slots, std::vector<int>* result);

  /*!
   * \brief Allocate slots for the draft tokens.
   * \param num_slots The number of slots to allocate.
   * \param initial_ref_count The initial reference count for each slot.
   * \param result The vector to store the allocated slots.
   */
  void AllocSlots(int num_slots, const std::vector<int>& initial_ref_count,
                  std::vector<int>* result);

  /*!
   * \brief Free the slots.
   * \param slots The slots to free.
   */
  void FreeSlots(const std::vector<int>& slots);

  static constexpr const char* _type_key = "mlc.serve.DraftTokenWorkspaceManager";

 private:
  std::vector<int> free_slots_;
  int max_num_tokens_;
  int vocab_size_;
  int hidden_size_;
  DataType hidden_states_dtype_;
  DLDevice device_;
  const FunctionTable& ft_;
  std::unordered_map<int, int> ref_count_;
};

class DraftTokenWorkspaceManager : public ObjectRef {
 public:
  DraftTokenWorkspaceManager(int max_num_tokens, int vocab_size, int hidden_size,
                             DLDataType hidden_states_dtype, DLDevice device,
                             const FunctionTable& ft) {
    data_ = tvm::ffi::make_object<DraftTokenWorkspaceManagerObj>(
        max_num_tokens, vocab_size, hidden_size, hidden_states_dtype, device, ft);
  }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(DraftTokenWorkspaceManager, ObjectRef,
                                        DraftTokenWorkspaceManagerObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_DRAFT_TOKEN_WORKSPACE_MANAGER_H_
