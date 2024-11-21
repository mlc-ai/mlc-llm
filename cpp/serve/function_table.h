/*!
 *  Copyright (c) 2023-2024 by Contributors
 * \file serve/function_table.h
 * \brief The header for function table in serving for distributed inference.
 */

#ifndef MLC_LLM_SERVE_FUNCTION_TABLE_H_
#define MLC_LLM_SERVE_FUNCTION_TABLE_H_

#include <picojson.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <string>

#include "../metadata/model.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

//--------------------------------------------------------
// The function table under batching settings.
// The implementation is mostly the same as the one for
// single-sequence distributed inference in llm_chat.cc.
// The only difference is that the function table for
// batching uses a different set of packed functions.
//
// Here we choose to have the duplicate code instead of
// reusing the existing function table. This is mainly
// for the independent development of batching/serving
// and make the codebase manageable.
// We will eventually merge two implementation into one
// after the batching development becomes stable.
//--------------------------------------------------------
struct FunctionTable {
  static PackedFunc SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name);

  void Init(String reload_lib_path, Device device, picojson::object model_config,
            Optional<Session> session, int num_shards, int num_stages);

  ObjectRef LoadParams(const std::string& model_path, Device device);

  void _InitFunctions();

  ObjectRef Empty(ShapeTuple shape, DataType dtype, Device device, bool worker0_only) const;

  /*!
   * \brief Copy a host array to the worker or local gpu.
   * \param host_array The host array to be copied.
   * \param buffer_cache_key The key to the buffer cache.
   * \param max_reserved_shape The maximum shape to be reserved in the buffer cache.
   * \param local_only Whether to copy the array to the local gpu only. If true, the use_disco
   *                  flag will be ignored. This can be useful for functions that run only on the
   *                  local gpu when disco is enabled.
   * \return The array on the worker or local gpu.
   */
  ObjectRef CopyToWorker0(const NDArray& host_array, String buffer_cache_key,
                          ShapeTuple max_reserved_shape, bool local_only = false);

  void DebugCallFuncOnAllAllWorker(const String& func_name) const;

  bool use_disco = false;
  Device local_gpu_device;
  Session sess{nullptr};
  DRef disco_mod{nullptr};
  Map<String, ObjectRef> cached_buffers{nullptr};
  tvm::runtime::Module local_vm{nullptr};
  picojson::object model_config;

  TypedPackedFunc<PackedFunc(const std::string&)> mod_get_func;
  TypedPackedFunc<PackedFunc(const std::string&)> get_global_func;

  ModelMetadata model_metadata_;

  PackedFunc embed_func_;
  PackedFunc image_embed_func_;
  PackedFunc single_batch_prefill_func_;
  PackedFunc single_batch_decode_func_;
  PackedFunc prefill_func_;
  PackedFunc decode_func_;
  PackedFunc verify_func_;
  PackedFunc single_batch_prefill_to_last_hidden_func_;
  PackedFunc single_batch_decode_to_last_hidden_func_;
  PackedFunc prefill_to_last_hidden_func_;
  PackedFunc decode_to_last_hidden_func_;
  PackedFunc verify_to_last_hidden_func_;
  PackedFunc fuse_embed_hidden_func_;
  PackedFunc get_logits_func_;
  PackedFunc batch_get_logits_func_;
  PackedFunc batch_select_last_hidden_func_;
  PackedFunc softmax_func_;
  PackedFunc apply_logit_bias_func_;
  PackedFunc apply_penalty_func_;
  PackedFunc apply_bitmask_func_;
  PackedFunc alloc_embedding_tensor_func_;
  PackedFunc create_kv_cache_func_;
  PackedFunc reset_kv_cache_func_;
  bool support_backtracking_kv_;
  PackedFunc kv_cache_add_sequence_func_;
  PackedFunc kv_cache_fork_sequence_func_;
  PackedFunc kv_cache_enable_sliding_window_for_seq_;
  PackedFunc kv_cache_remove_sequence_func_;
  PackedFunc kv_cache_begin_forward_func_;
  PackedFunc kv_cache_end_forward_func_;
  PackedFunc kv_cache_popn_func_;
  PackedFunc kv_cache_commit_accepted_token_tree_nodes_func_;
  PackedFunc kv_cache_get_num_available_pages_func_;
  PackedFunc kv_cache_get_total_sequence_length_func_;
  PackedFunc gpu_multinomial_from_uniform_func_;
  PackedFunc gpu_argsort_probs_func_;
  PackedFunc gpu_sample_with_top_p_func_;
  PackedFunc gpu_sampler_take_probs_func_;
  PackedFunc gpu_verify_draft_tokens_func_;
  PackedFunc gpu_renormalize_by_top_p_func_;
  PackedFunc nd_view_func_;
  PackedFunc nd_get_shape_func_;
  PackedFunc nd_copy_embedding_to_offset_func_;
  PackedFunc tuple_getitem_func_;
  PackedFunc last_group_send_to_worker_0_;
  // Auxiliary functions for speculative decoding.
  PackedFunc gather_probs_func_;
  PackedFunc scatter_probs_func_;
  PackedFunc gather_hidden_states_func_;
  PackedFunc scatter_hidden_states_func_;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_FUNCTION_TABLE_H_
