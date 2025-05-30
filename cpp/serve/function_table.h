/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/function_table.h
 * \brief The header for function table in serving for distributed inference.
 */

#ifndef MLC_LLM_SERVE_FUNCTION_TABLE_H_
#define MLC_LLM_SERVE_FUNCTION_TABLE_H_

#include <picojson.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include <string>

#include "../metadata/model.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;
using tvm::ffi::Function;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::TypedFunction;

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
  static Function SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name);

  void Init(String reload_lib_path, Device device, picojson::object model_config,
            Optional<Session> session, int num_shards, int num_stages);

  ObjectRef LoadParams(const std::string& model_path, Device device);

  void _InitFunctions();

  ObjectRef Empty(Shape shape, DataType dtype, Device device, bool worker0_only) const;

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
                          Shape max_reserved_shape, bool local_only = false);

  void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) const;

  bool use_disco = false;
  Device local_gpu_device;
  Session sess{nullptr};
  DRef disco_mod{nullptr};
  Map<String, ObjectRef> cached_buffers{nullptr};
  tvm::runtime::Module local_vm{nullptr};
  picojson::object model_config;

  TypedFunction<Function(const std::string&)> mod_get_func;
  TypedFunction<Function(const std::string&)> get_global_func;

  ModelMetadata model_metadata_;

  Function embed_func_;
  Function image_embed_func_;
  Function single_batch_prefill_func_;
  Function single_batch_decode_func_;
  Function single_batch_extend_func_;
  Function prefill_func_;
  Function decode_func_;
  Function extend_func_;
  Function verify_func_;
  Function single_batch_prefill_to_last_hidden_func_;
  Function single_batch_decode_to_last_hidden_func_;
  Function prefill_to_last_hidden_func_;
  Function decode_to_last_hidden_func_;
  Function verify_to_last_hidden_func_;
  Function fuse_embed_hidden_func_;
  Function get_logits_func_;
  Function batch_get_logits_func_;
  Function batch_select_last_hidden_func_;
  Function softmax_func_;
  Function apply_logit_bias_func_;
  Function apply_penalty_func_;
  Function apply_bitmask_func_;
  Function alloc_embedding_tensor_func_;
  Function cuda_graph_alloc_init_func_;
  Function create_kv_cache_func_;
  Function reset_kv_cache_func_;
  bool support_backtracking_kv_;
  Function kv_cache_add_sequence_func_;
  Function kv_cache_fork_sequence_func_;
  Function kv_cache_enable_sliding_window_for_seq_;
  Function kv_cache_remove_sequence_func_;
  Function kv_cache_begin_forward_func_;
  Function kv_cache_end_forward_func_;
  Function kv_cache_disagg_prepare_recv_func_;
  Function kv_cache_disagg_mark_send_func_;
  Function kv_cache_popn_func_;
  Function kv_cache_commit_accepted_token_tree_nodes_func_;
  Function kv_cache_get_num_available_pages_func_;
  Function kv_cache_get_total_sequence_length_func_;
  Function gpu_multinomial_from_uniform_func_;
  Function gpu_argsort_probs_func_;
  Function gpu_sample_with_top_p_func_;
  Function gpu_sampler_take_probs_func_;
  Function gpu_verify_draft_tokens_func_;
  Function gpu_renormalize_by_top_p_func_;
  Function nd_view_func_;
  Function nd_get_shape_func_;
  Function nd_copy_embedding_to_offset_func_;
  Function tuple_getitem_func_;
  Function last_group_send_to_worker_0_;
  // Auxiliary functions for speculative decoding.
  Function gather_probs_func_;
  Function scatter_probs_func_;
  Function gather_hidden_states_func_;
  Function scatter_hidden_states_func_;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_FUNCTION_TABLE_H_
