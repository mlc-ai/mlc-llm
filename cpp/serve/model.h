/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/model.h
 * \brief The header for runtime module of LLM functions (prefill/decode/etc.)
 */

#ifndef MLC_LLM_SERVE_MODEL_H_
#define MLC_LLM_SERVE_MODEL_H_

#include <picojson.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/ndarray.h>

#include "../base.h"
#include "../support/result.h"
#include "config.h"
#include "draft_token_workspace_manager.h"
#include "event_trace_recorder.h"
#include "function_table.h"
#include "logit_processor.h"
#include "sampler/sampler.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

// Declare the sampler class for `Model::CreateSampler`.
class Sampler;

/*!
 * \brief The workspace tensors that may be shared across different
 * calls to Model. For example, the prefill action use the `embeddings`
 * workspace for the concatenated embeddings of different sequences.
 * The workspace tensor is created by Model but owned by engine.
 */
struct ModelWorkspace {
  /*!
   * \brief The embedding tensor. It can be either an NDArray when tensor
   * model parallelism is not enabled, or a DRef when using tensor model parallelism.
   */
  ObjectRef embeddings{nullptr};
  /*!
   * \brief The hidden_states tensor for the current batch. It can be either an NDArray when tensor
   * model parallelism is not enabled, or a DRef when using tensor model parallelism.
   */
  ObjectRef hidden_states{nullptr};

  /*!
   * \brief The draft token probabilities tensor for the current batch.
   */
  NDArray draft_probs{nullptr};

  /*!
   * \brief The hidden_states tensor storing the hidden_states of draft tokens of all requests.
   */
  ObjectRef draft_hidden_states_storage{nullptr};

  /*!
   * \brief The draft token probabilities tensor storing the probabilities of draft tokens of all
   * requests.
   */
  NDArray draft_probs_storage{nullptr};
};

/*!
 * \brief The model module for LLM functions.
 * It runs an LLM, and has an internal KV cache that maintains
 * the history KV values of all processed tokens.
 *
 * It contains the following functions:
 *
 * Model related:
 * - "token_embed": take token ids as input and return the embeddings,
 * - "batch_prefill": take embedding of a single sequence
 * as input, forward the embedding through LLM and return the logits,
 * - "decode": take the embeddings of the last-committed token of an
 * entire batch as input, forward through LLM and return the logits
 * for all sequences in the batch,
 * - "softmax_with_temperature": take logits and temperatures, return
 * probabilities.
 *
 * KV cache related:
 * - "create_kv_cache": create the KV cache for this module,
 * - "add_new_sequence": add (declare) a new sequence in the KV cache,
 * - "remove_sequence": remove a sequence from KV cache.
 *
 * ... and some other auxiliary functions.
 */
class ModelObj : public Object {
 public:
  /*********************** Model Computation  ***********************/

  /*!
   * \brief Compute embeddings for the input token ids.
   * When the input destination pointer is defined, it in-place writes the
   * embedding into the input destination array at the given offset.
   * Otherwise, the embeddings will be directly returned back.
   * \param token_ids The token ids to compute embedding for.
   * \param dst The destination array of the embedding lookup.
   * \param offset The token offset where the computed embeddings will be written
   * into the destination array.
   * \return The updated destination embedding array or the computed embeddings.
   * \note When `dst` is undefined, we require `offset` to be 0.
   */
  virtual ObjectRef TokenEmbed(IntTuple batch_token_ids, ObjectRef* dst = nullptr,
                               int offset = 0) = 0;

  /*!
   * \brief Compute embeddings for the input image.
   * \param image The image to compute embedding for.
   * \return The computed embeddings.
   */
  virtual ObjectRef ImageEmbed(const NDArray& image, ObjectRef* dst = nullptr, int offset = 0) = 0;

  /*!
   * \brief Fuse the embeddings and hidden_states.
   * \param embeddings The embedding of the input to be prefilled.
   * \param previous_hidden_states The hidden_states from previous base model.
   * \param batch_size Batch size.
   * \param seq_len Sequence length.
   * \return The fused hidden_states.
   */
  virtual ObjectRef FuseEmbedHidden(const ObjectRef& embeddings,
                                    const ObjectRef& previous_hidden_states, int batch_size,
                                    int seq_len) = 0;

  /*!
   * \brief Return if the model has lm_head so that we can get logits.
   */
  virtual bool CanGetLogits() = 0;

  /*!
   * \brief Compute logits for last hidden_states.
   * \param last_hidden_states The last hidden_states to compute logits for.
   * \return The computed logits.
   */
  virtual NDArray GetLogits(const ObjectRef& last_hidden_states) = 0;

  virtual Array<NDArray> GetMultiStepLogits(const ObjectRef& last_hidden_states) = 0;

  /*!
   * \brief Batch prefill function. Embedding in, logits out.
   * The embedding order of sequences in `embedding_arr` follows
   * the order of `seq_ids`.
   * \param embeddings The embedding of the input to be prefilled.
   * \param seq_id The id of the sequence in the KV cache.
   * \param lengths The length of each sequence to prefill.
   * \return The logits for the next token.
   */
  virtual NDArray BatchPrefill(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                               const std::vector<int>& lengths) = 0;

  /*!
   * \brief Batch prefill function. Input hidden_states are computed from
   * input embeddings and previous hidden_states, output last hidden_states.
   * \param hidden_states The hidden_states of the input to be prefilled.
   * \param seq_id The id of the sequence in the KV cache.
   * \param lengths The length of each sequence to prefill.
   * \return The hidden_states for the next token.
   */
  virtual ObjectRef BatchPrefillToLastHidden(const ObjectRef& hidden_states,
                                             const std::vector<int64_t>& seq_ids,
                                             const std::vector<int>& lengths) = 0;

  /*!
   * \brief Batch decode function. Embedding in, logits out.
   * The embedding order of sequences in `embeddings` follows
   * the order of `seq_ids`.
   * \param embeddings The embedding of last generated token in the entire batch.
   * \param seq_id The id of the sequence in the KV cache.
   * \return The logits for the next token for each sequence in the batch.
   */
  virtual NDArray BatchDecode(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids) = 0;

  virtual NDArray BatchTreeDecode(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                                  const std::vector<int>& lengths,
                                  const std::vector<int64_t>& token_tree_parent_ptr) = 0;

  /*!
   * \brief Batch decode function. Input hidden_states are computed from
   * input embeddings and previous hidden_states, output last hidden_states.
   * \param hidden_states The hidden_states of last generated token in the entire batch.
   * \param seq_id The id of the sequence in the KV cache.
   * \return The hidden_states for the next token for each sequence in the batch.
   */
  virtual ObjectRef BatchDecodeToLastHidden(const ObjectRef& hidden_states,
                                            const std::vector<int64_t>& seq_ids) = 0;

  /*!
   * \brief Batch verify function. Embedding in, logits out.
   * \param embeddings The embedding of the input to be verified.
   * \param seq_id The id of the sequence in the KV cache.
   * \param lengths The length of each sequence to verify.
   * \param token_tree_parent_ptr The parent pointers of the token tree.
   * It's size is the sum of "lengths". It contains a batch of independent trees,
   * one for each sequence. Parent being "-1" means the node is a root.
   * \return The logits for the draft token for each sequence in the batch.
   * \note The function runs for **every** sequence in the batch.
   * That is to say, it does not accept "running a verify step for a subset
   * of the full batch".
   */
  virtual NDArray BatchVerify(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                              const std::vector<int>& lengths,
                              const std::vector<int64_t>& token_tree_parent_ptr) = 0;

  /*!
   * \brief Batch verify function. Input hidden_states are computed from
   * input embeddings and previous hidden_states, output last hidden_states.
   * \param hidden_states The hidden_states of the input to be verified.
   * \param seq_id The id of the sequence in the KV cache.
   * \param lengths The length of each sequence to verify.
   * \param token_tree_parent_ptr The parent pointers of the token tree.
   * It's size is the sum of "lengths". It contains a batch of independent trees,
   * one for each sequence. Parent being "-1" means the node is a root.
   * \return The hidden_states for the draft token for each sequence in the batch.
   * \note The function runs for **every** sequence in the batch.
   * That is to say, it does not accept "running a verify step for a subset
   * of the full batch".
   */
  virtual ObjectRef BatchVerifyToLastHidden(const ObjectRef& hidden_states,
                                            const std::vector<int64_t>& seq_ids,
                                            const std::vector<int>& lengths,
                                            const std::vector<int64_t>& token_tree_parent_ptr) = 0;

  /*********************** KV Cache Management  ***********************/

  /*!
   * \brief Create the KV cache inside the model with regard to the input config.
   * \param page_size The number of consecutive tokens handled in each page in paged KV cache.
   * \param max_num_sequence The maximum number of sequences that are allowed to be
   * processed by the KV cache at any time.
   * \param max_total_sequence_length The maximum length allowed for a single sequence
   * in the engine.
   * \param prefill_chunk_size The maximum total number of tokens whose KV data
   * are allowed to exist in the KV cache at any time.
   * \param max_history_size The maximum history size for RNN state to roll back.
   * The KV cache does not need this.
   */
  virtual void CreateKVCache(int page_size, int max_num_sequence, int64_t max_total_sequence_length,
                             int64_t prefill_chunk_size, int max_history_size) = 0;

  /*! \brief Add a new sequence with the given sequence id to the KV cache. */
  virtual void AddNewSequence(int64_t seq_id) = 0;

  /*! \brief Fork a sequence from a given parent sequence. */
  virtual void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos = -1) = 0;

  /*! \brief Remove the given sequence from the KV cache in the model. */
  virtual void RemoveSequence(int64_t seq_id) = 0;

  /*! \brief Pop out N pages from KV cache. */
  virtual void PopNFromKVCache(int64_t seq_id, int num_tokens) = 0;

  /*!
   * \brief Commit the accepted token tree nodes to KV cache.
   * The unaccepted token tree node will be removed from KV cache.
   * This is usually used in the verification stage of speculative decoding.
   */
  virtual void CommitAcceptedTokenTreeNodesToKVCache(
      const std::vector<int64_t>& seq_ids, const std::vector<int64_t>& accepted_leaf_indices) = 0;

  /*!
   * \brief Enabling sliding window for the given sequence.
   * It is a no-op if the model does not support sliding window.
   * \note Given this operation is tied with the underlying KV cache,
   * we add the function in Model interface to expose this for Engine.
   * This may be optimized with decoupling KV cache and Model in the future.
   */
  virtual void EnableSlidingWindowForSeq(int64_t seq_id) = 0;

  /*! \brief Prepare for the disaggregation KV data receive for the specified sequence and length.*/
  virtual IntTuple DisaggPrepareKVRecv(int64_t seq_id, int length) = 0;

  /*! \brief Prepare for the disaggregation KV data send for the specified sequence and length.*/
  virtual void DisaggMarkKVSend(int64_t seq_id, int begin_pos,
                                IntTuple compressed_kv_append_metadata, int dst_group_offset) = 0;

  /************** Raw Info Query **************/

  /*! \brief Return the metadata JSON object of the model. */
  virtual ModelMetadata GetMetadata() const = 0;

  /*! \brief Get the number of available pages in KV cache. */
  virtual int GetNumAvailablePages() const = 0;

  /*! \brief Get the current total sequence length in the KV cache. */
  virtual int GetCurrentTotalSequenceLength() const = 0;

  /*********************** Utilities  ***********************/

  /*! \brief Load the model's weight parameters, which is not loaded at construction time. */
  virtual void LoadParams() = 0;

  /*!
   * \brief Set the maximum number of sequences to be processed for the model,
   * which is not initialized at construction time.
   */
  virtual void SetMaxNumSequence(int max_num_sequence) = 0;

  /*!
   * \brief Set the prefill chunk size for the model,
   * which is not initialized at construction time.
   */
  virtual void SetPrefillChunkSize(int prefill_chunk_size) = 0;

  /*! \brief Create a logit processor from this model. */
  virtual LogitProcessor CreateLogitProcessor(int max_num_token,
                                              Optional<EventTraceRecorder> trace_recorder) = 0;

  /*! \brief Create a sampler from this model. */
  virtual Sampler CreateSampler(int max_num_sample, int num_models,
                                Optional<EventTraceRecorder> trace_recorder) = 0;

  /*!
   * \brief Estimate number of CPU units required to drive the model
   * executing during TP.
   * \note This normally equals to the number of TP shards (or 0 if
   * the model does not use TP) and can be used to hint runtime to
   * avoid overuse cores in other places.
   */
  virtual int EstimateHostCPURequirement() const = 0;

  /*! \brief Get the sliding window size of the model. "-1" means sliding window is not enabled. */
  virtual int GetSlidingWindowSize() const = 0;

  /*! \brief Get the attention sink size of the model. */
  virtual int GetAttentionSinkSize() const = 0;

  /*! \brief Allocate an embedding tensor with the prefill chunk size. */
  virtual ObjectRef AllocEmbeddingTensor() = 0;

  /*! \brief Allocate an hidden_states tensor with the prefill chunk size. */
  virtual ObjectRef AllocHiddenStatesTensor() = 0;

  /*! \brief Reset the model KV cache and other metrics. */
  virtual void Reset() = 0;

  /*********************** Utilities for speculative decoding. ***********************/

  virtual DraftTokenWorkspaceManager CreateDraftTokenWorkspaceManager(int max_num_token) = 0;

  /*! \brief Gather the hidden_states of the given indices and in-place update the dst tensor. */
  virtual ObjectRef GatherHiddenStates(const ObjectRef& input, const std::vector<int>& indices,
                                       ObjectRef* dst) = 0;

  /*! \brief Scatter the hidden_states of the given indices to the dst tensor. */
  virtual void ScatterHiddenStates(const ObjectRef& input, const std::vector<int>& indices,
                                   ObjectRef* dst) = 0;

  /*! \brief Gather the draft token probabilities of the given indices and in-place update the dst
   * tensor. */
  virtual NDArray GatherDraftProbs(const NDArray& input, const std::vector<int>& indices,
                                   NDArray* dst) = 0;

  /*! \brief Scatter the draft token probabilities of the given indices to the dst tensor. */
  virtual void ScatterDraftProbs(const NDArray& input, const std::vector<int>& indices,
                                 NDArray* dst) = 0;

  /************** Debug/Profile **************/

  /*! \brief Call the given global function on all workers. Only for debug purpose. */
  virtual void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) = 0;

  static constexpr const char* _type_key = "mlc.serve.Model";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(ModelObj, Object);
};

class Model : public ObjectRef {
 public:
  /*!
   * \brief Create the runtime module for LLM functions.
   * \param reload_lib_path The model library path.
   * \param model_path The path to the model weight parameters.
   * \param model_config The model config json object.
   * \param device The device to run the model on.
   * \param session The session to run the model on.
   * \param num_shards The number of tensor parallel shards of the model.
   * \param num_stages The number of pipeline parallel stages of the model.
   * \param trace_enabled A boolean indicating whether tracing is enabled.
   * \return The created runtime module.
   */
  static Model Create(String reload_lib_path, String model_path,
                      const picojson::object& model_config, DLDevice device,
                      const Optional<Session>& session, int num_shards, int num_stages,
                      bool trace_enabled);

  /*!
   * Load the model config from the given model path.
   * \param model_path The path to the model weight parameters.
   * \return The model config json object.
   */
  static Result<picojson::object> LoadModelConfig(const String& model_path);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Model, ObjectRef, ModelObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_MODEL_H_
