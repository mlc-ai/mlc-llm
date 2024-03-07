/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/model.h
 * \brief The header for runtime module of LLM functions (prefill/decode/etc.)
 */

#ifndef MLC_LLM_SERVE_MODEL_H_
#define MLC_LLM_SERVE_MODEL_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>

#include "../base.h"
#include "config.h"
#include "event_trace_recorder.h"
#include "function_table.h"
#include "logit_processor.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

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
   * \brief Batch decode function. Embedding in, logits out.
   * The embedding order of sequences in `embeddings` follows
   * the order of `seq_ids`.
   * \param embeddings The embedding of last generated token in the entire batch.
   * \param seq_id The id of the sequence in the KV cache.
   * \return The logits for the next token for each sequence in the batch.
   */
  virtual NDArray BatchDecode(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids) = 0;

  /*!
   * \brief Batch verify function. Embedding in, logits out.
   * \param embeddings The embedding of the input to be verified.
   * \param seq_id The id of the sequence in the KV cache.
   * \param lengths The length of each sequence to verify.
   * \return The logits for the draft token for each sequence in the batch.
   * \note The function runs for **every** sequence in the batch.
   * That is to say, it does not accept "running a verify step for a subset
   * of the full batch".
   */
  virtual NDArray BatchVerify(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                              const std::vector<int>& lengths) = 0;

  /*********************** KV Cache Management  ***********************/

  /*!
   * \brief Create the KV cache inside the model with regard to the input config.
   * \param kv_cache_config The configuration of KV cache.
   */
  virtual void CreateKVCache(KVCacheConfig kv_cache_config) = 0;

  /*! \brief Add a new sequence with the given sequence id to the KV cache. */
  virtual void AddNewSequence(int64_t seq_id) = 0;

  /*! \brief Fork a sequence from a given parent sequence. */
  virtual void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id) = 0;

  /*! \brief Remove the given sequence from the KV cache in the model. */
  virtual void RemoveSequence(int64_t seq_id) = 0;

  /*! \brief Get the number of available pages in KV cache. */
  virtual int GetNumAvailablePages() const = 0;

  /*! \brief Pop out N pages from KV cache. */
  virtual void PopNFromKVCache(int seq_id, int num_tokens) = 0;

  /*********************** Utilities  ***********************/

  /*! \brief Create a logit processor from this model. */
  virtual LogitProcessor CreateLogitProcessor(int max_num_token,
                                              Optional<EventTraceRecorder> trace_recorder) = 0;

  /*!
   * \brief Estimate number of CPU units required to drive the model
   * executing during TP.
   * \note This normally equals to the number of TP shards (or 0 if
   * the model does not use TP) and can be used to hint runtime to
   * avoid overuse cores in other places.
   */
  virtual int EstimateHostCPURequirement() const = 0;

  /*! \brief Get the max window size of the model. */
  virtual int GetMaxWindowSize() const = 0;

  /*! \brief Allocate an embedding tensor with the prefill chunk size. */
  virtual ObjectRef AllocEmbeddingTensor() = 0;

  /*! \brief Reset the model KV cache and other statistics. */
  virtual void Reset() = 0;

  static constexpr const char* _type_key = "mlc.serve.Model";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(ModelObj, Object);
};

class Model : public ObjectRef {
 public:
  /*!
   * \brief Create the runtime module for LLM functions.
   * \param reload_lib The model library. It might be a path to the binary
   * file or an executable module that is pre-loaded.
   * \param model_path The path to the model weight parameters.
   * \param device The device to run the model on.
   * \param max_num_sequence The maximum number of sequences to be processed
   * \return The created runtime module.
   */
  TVM_DLL static Model Create(TVMArgValue reload_lib, String model_path, DLDevice device,
                              int max_num_sequence);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Model, ObjectRef, ModelObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_MODEL_H_
