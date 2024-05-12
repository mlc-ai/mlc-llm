/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/prefix_cache.h
 */
#ifndef MLC_LLM_SERVE_PREFIX_CACHE_H_
#define MLC_LLM_SERVE_PREFIX_CACHE_H_
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

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
 * \brief The matched result from prefix cache.
 */
struct PrefixCacheMatchedResult {
  /*!
   * \brief The new sequence ID.
   * If it is the same as original, do nothing. Otherwise, it is different, which means reusing an
   * exsisting recycling sequence ID.
   */
  int64_t new_seq_id = -1;
  /*!
   * \brief The parent sequence ID to fork in KVCache. The default value if -1, which means no
   * forking operation needed.
   */
  int64_t parent_seq_id = -1;
  /*!
   * \brief The matched prefix offset, which should be skipped when prefilling.
   */
  size_t matched_offset = 0;
  /*!
   * \brief The number of tailing tokens to be popped in KVCache. Used when stripping the reused
   * recycling sequence to the matched offset.
   */
  size_t pop_last_tokens = 0;
};

class PrefixCacheObj : public Object {
 public:
  /*!
   * \brief Insert a new tokenized sequence into Prefix Cache.
   * \param tokens The tokens of tokenized sequence.
   * \return The matched result.
   */
  virtual PrefixCacheMatchedResult InsertSequence(int64_t seq_id, IntTuple tokens) = 0;

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * \param seq_id The sequence to be extneded.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if the given sequence id is not valid or active.
   */
  virtual void ExtendSequence(int64_t seq_id, IntTuple tokens) = 0;

  /*!
   * \brief Roll back a sequence by number of tokens.
   * \param seq_id The sequence ID for index.
   * \param num_tokens The number of tokens to be rolled back.
   * \throw Error if the given sequence id is not valid or active.
   */
  virtual void RollBackSequence(int64_t seq_id, size_t num_tokens) = 0;

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   memory is sufficient. And it will be reused again in the future request.
   * \param seq_id The sequence to be recycled.
   * \param callback The callback function to be invoked when removing the sequence.
   * \param lazy The flag if the sequence should be removed lazily or intermediary.
   * \throw Error if the given sequence id is not valid.
   */
  virtual void RecycleSequence(int64_t seq_id, PackedFunc callback, bool lazy = true) = 0;

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
  void Reset(){};

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mlc.serve.PrefixCache";
  TVM_DECLARE_BASE_OBJECT_INFO(PrefixCacheObj, Object)
};

TVM_REGISTER_OBJECT_TYPE(PrefixCacheObj);

class PrefixCache : public ObjectRef {
 public:
  /*!
   * \brief Initialization of paged radix tree.
   * \param num_pages The number of radix tree pages.
   * \param page_size The page size of each radix tree page.
   * \param num_seqs The maximum number of sequence ID.
   * \param sliding_window_size The sliding window size, -1 for disabled sliding window.
   * \param attention_sink_size The attention sink position for sliding window.
   */
  static PrefixCache Init(size_t num_pages, size_t page_size, size_t num_seqs,
                          int sliding_window_size, int attention_sink_size);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PrefixCache, ObjectRef, PrefixCacheObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_PREFIX_CACHE_H_
