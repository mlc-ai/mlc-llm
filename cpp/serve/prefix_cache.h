/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/prefix_cache.h
 */
#ifndef MLC_LLM_SERVE_PREFIX_CACHE_H_
#define MLC_LLM_SERVE_PREFIX_CACHE_H_
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/object.h>

#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "model.h"
#include "radix_tree.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The signature of callback removing function.
 */
using PrefixCacheRemoveCallback = std::function<void(int64_t)>;

/*!
 * \brief The matched result from prefix cache. This result describes how to pre-process the new
 * sequence, to leverage the existing data in KVCache by reusing past sequences or forking from
 * other sequences.
 */
class PrefixCacheMatchedResult {
 public:
  /*!
   * \brief The matched and prefilled prefix offset.
   */
  size_t prefilled_offset = 0;
  /*!
   * \brief The sequence ID to fork from.
   */
  int64_t forked_seq_id = -1;
  /*!
   * \brief The finished sequence ID to reuse.
   */
  int64_t reused_seq_id = -1;
  /*!
   * \brief The number of tailing tokens to be popped from the reused sequence.
   */
  size_t reused_seq_pop_last_tokens = 0;
};

class PrefixCacheObj : public Object {
 public:
  /*!
   * \brief Insert a new tokenized sequence into Prefix Cache.
   * \param seq_id The sequence ID.
   * \param tokens The tokens of tokenized sequence.
   * \param sliding_window_size The sliding window size for the sequence, -1 as sliding window
   * disabled.
   * \param attention_sink_size The attention sink size for the sequence, 0 by default.
   * \return The matched result.
   */
  virtual PrefixCacheMatchedResult InsertSequence(int64_t seq_id, std::vector<int32_t> tokens,
                                                  int sliding_window_size = -1,
                                                  int attention_sink_size = 0) = 0;

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * This extension might be cached and lazily committed later.
   * \param seq_id The sequence to be extended.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if the given sequence id is not valid or active.
   */
  virtual void ExtendSequence(int64_t seq_id, const std::vector<int32_t>& tokens) = 0;

  /*! \brief Commit the cached sequence extension from "ExtendSequence". */
  virtual void CommitSequenceExtention() = 0;

  /*!
   * \brief Roll back a sequence by number of tokens.
   * \param seq_id The sequence ID for index.
   * \param num_tokens The number of tokens to be rolled back.
   * \throw Error if the given sequence id is not valid or active.
   */
  virtual void RollBackSequence(int64_t seq_id, size_t num_tokens) = 0;

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   * memory is sufficient and the number of sequence in prefix cache belows the maximum number of
   * sequence. And it will be reused again in the future request.
   * \param seq_id The sequence to be recycled.
   * \param lazy The flag if the sequence should be removed lazily or intermediary.
   * \throw Error if the given sequence id is not valid.
   */
  virtual void RecycleSequence(int64_t seq_id, bool lazy = true) = 0;

  /*!
   * \brief Try to remove recycling sequence to free up memory. It will remove the oldest recycling
   sequence.
   * \return The flag if there is a sequence removed. In other word, return true when memory is
   freed successfully.
   * \throw Error if the given sequence id is not valid.
   */
  virtual bool TryFreeMemory() = 0;

  /*!
   * \brief Check if a sequence exists.
   * \param seq_id The sequence ID for index.
   * \return The sequence existence.
   * \throw Error if sequence ID is not valid.
   */
  virtual bool HasSequence(int64_t seq_id) = 0;

  /*!
   * \brief Reset the prefix cache to initial status.
   */
  virtual void Reset() = 0;

  /*! \brief Return the prefix cache mode. */
  virtual PrefixCacheMode Mode() = 0;

  static constexpr const char* _type_key = "mlc.serve.PrefixCache";
  TVM_DECLARE_BASE_OBJECT_INFO(PrefixCacheObj, Object);
};

TVM_REGISTER_OBJECT_TYPE(PrefixCacheObj);

class PrefixCache : public ObjectRef {
 public:
  /*!
   * \brief Initialization of prefix cache.
   * \param max_recycling_seqs The maximum number of recycling sequences in prefix cache.
   * \param remove_callback The optional callback function to call when removing a sequence.
   */
  static PrefixCache CreateRadixPrefixCache(size_t max_recycling_seqs,
                                            PrefixCacheRemoveCallback remove_callback = nullptr);
  /*!
   * \brief Initialization of no prefix cache.
   */
  static PrefixCache CreateNoPrefixCache();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PrefixCache, ObjectRef, PrefixCacheObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_PREFIX_CACHE_H_
