/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/prefix_cache.cc
 */
#include "prefix_cache.h"

#include <tvm/runtime/registry.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

class PrefixCacheImpl : public PrefixCacheObj {
 public:
  /*!
   * \brief Contructor of paged radix tree.
   * \param num_pages The number of radix tree pages.
   * \param page_size The page size of each radix tree page.
   * \param num_seqs The maximum number of sequence ID.
   * \param sliding_window_size The sliding window size, -1 for disabled sliding window.
   * \param attention_sink_size The attention sink position for sliding window.
   */
  explicit PrefixCacheImpl(size_t num_pages, size_t page_size, size_t num_seqs,
                           int sliding_window_size, int attention_sink_size)
      : radix_tree(PagedRadixTree(num_pages, page_size, num_seqs)), max_recycling_seqs(num_seqs) {
    latest_visit.clear();
    recycle_callbacks.clear();
    recycling_seqs.clear();
    lru_counter = 0;
    if (sliding_window_size > 0) {
      sliding_window = true;
      max_fork_offset = attention_sink_size;
    } else {
      sliding_window = false;
      max_fork_offset = -1;
    }
  }

  /*!
   * \brief Insert a new tokenized sequence into Prefix Cache.
   * \param tokens The tokens of tokenized sequence.
   * \return The matched result.
   */
  MatchedResult InsertSequence(int64_t seq_id, IntTuple tokens) {
    auto [matched_offset, matched_seqs] = radix_tree->MatchPrefix(tokens);
    if (!matched_offset) {
      // No prefix matched
      radix_tree->AddSequence(seq_id);
      ++lru_counter;
      latest_visit[seq_id] = lru_counter;
      return MatchedResult{seq_id, -1, 0, 0};
    }

    CHECK(!matched_seqs.empty());

    size_t shortest_recycling_seq_length = 0;
    int64_t shortest_recycling_seq_id = -1;

    for (int64_t matched_seq_id : matched_seqs) {
      if (recycle_callbacks.find(matched_seq_id) != recycle_callbacks.end()) {
        size_t matched_seq_length = radix_tree->GetSequenceLength(matched_seq_id);
        if (shortest_recycling_seq_id == -1 || matched_seq_length < shortest_recycling_seq_length) {
          shortest_recycling_seq_id = matched_seq_id;
          shortest_recycling_seq_length = matched_seq_length;
        }
      }
    }
    // If the sequence is fully matched, roll back the last token to activate prefill.
    if (matched_offset == tokens.size()) --matched_offset;

    // For multiple candidates, we greedily reuse the shortest recycling sequence, so that the loss
    // or roll back trailing tokens will be minimum.
    if (shortest_recycling_seq_id != -1 &&
        (!sliding_window ||
         (matched_offset <= max_fork_offset || matched_offset == shortest_recycling_seq_length))) {
      // Reuse recycling sequence
      recycle_callbacks.erase(shortest_recycling_seq_id);
      CHECK(latest_visit.find(shortest_recycling_seq_id) != latest_visit.end());
      CHECK(recycling_seqs.erase(
          {latest_visit[shortest_recycling_seq_id], shortest_recycling_seq_id}));
      ++lru_counter;
      latest_visit[shortest_recycling_seq_id] = lru_counter;
      if (shortest_recycling_seq_length > matched_offset) {
        // Recycling sequence is longer than new sequence
        radix_tree->RollBackSequence(shortest_recycling_seq_id,
                                     shortest_recycling_seq_length - matched_offset);
      }
      return MatchedResult{shortest_recycling_seq_id, -1, matched_offset,
                           shortest_recycling_seq_length - matched_offset};
    }

    // If there is no recycling sequences in matched candidates, we can only fork from the active
    // sequences.
    if (sliding_window) {
      // If sliding window enabled, the sequence can be forked before attention sink position.
      matched_offset = std::min(matched_offset, max_fork_offset);
      if (!matched_offset) {
        radix_tree->AddSequence(seq_id);
        ++lru_counter;
        latest_visit[seq_id] = lru_counter;
        return MatchedResult{seq_id, -1, 0, 0};
      }
    }
    // Fork active sequence
    int64_t matched_seq_id = *matched_seqs.begin();
    radix_tree->ForkSequence(seq_id, matched_seq_id, matched_offset);
    ++lru_counter;
    latest_visit[seq_id] = lru_counter;
    return MatchedResult{seq_id, matched_seq_id, matched_offset, 0};
  }

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * \param seq_id The sequence to be extneded.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if the given sequence id is not valid or active.
   */
  void ExtendSequence(int64_t seq_id, IntTuple tokens) {
    ++lru_counter;
    radix_tree->ExtendSequence(seq_id, tokens);
    latest_visit[seq_id] = lru_counter;
  }

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   memory is sufficient. And it will be reused again in the future request.
   * \param seq_id The sequence to be recycled.
   * \param callback The callback function to be invoked when removing the sequence.
   * \param lazy The flag if the sequence should be removed lazily or intermediary.
   * \throw Error if the given sequence id is not valid.
   */
  void RecycleSequence(int64_t seq_id, PackedFunc callback, bool lazy = true) {
    CHECK(latest_visit.find(seq_id) != latest_visit.end());
    size_t timestamp = latest_visit[seq_id];
    CHECK(recycling_seqs.find({timestamp, seq_id}) == recycling_seqs.end());
    CHECK(recycle_callbacks.find(seq_id) == recycle_callbacks.end());
    if (lazy) {
      // Remove the sequence lazily.
      if (recycle_callbacks.size() == max_recycling_seqs) {
        // If the number of recycling sequence has reached the maximum values, pop the oldest one.
        TryFreeMemory();
      }
      recycle_callbacks[seq_id] = callback;
      recycling_seqs.emplace(latest_visit[seq_id], seq_id);
    } else {
      // Remove the sequence intermediately.
      radix_tree->RemoveSequence(seq_id);
      callback();
    }
  }

  /*!
   * \brief Try to remove recycling sequence to free up memory. It will remove the oldest recycling
   sequence.
   * \return The flag if there is a sequence removed. In other word, return true when memory is
   freed successfully.
   * \throw Error if the given sequence id is not valid.
   */
  bool TryFreeMemory() {
    if (recycling_seqs.empty()) {
      // There is no recycling sequence. No memory can be freed.
      return false;
    }
    auto it = recycling_seqs.begin();
    int64_t seq_id = it->second;
    CHECK(recycle_callbacks.find(seq_id) != recycle_callbacks.end());
    CHECK(latest_visit.find(seq_id) != latest_visit.end());
    radix_tree->RemoveSequence(seq_id);
    recycle_callbacks[seq_id]();

    recycle_callbacks.erase(seq_id);
    recycling_seqs.erase(it);
    latest_visit.erase(seq_id);
    return true;
  }

  /*!
   * \brief Check if a sequence exists.
   * \param seq_id The sequence ID for index.
   * \return The sequence existence.
   * \throw Error if sequence ID is not valid.
   */
  bool HasSequence(int64_t seq_id) { return radix_tree->HasSequence(seq_id); }

  /*!
   * \brief Reset the prefix cache to initial status.
   */
  void Reset() {
    radix_tree->Reset();
    latest_visit.clear();
    recycling_seqs.clear();
    recycle_callbacks.clear();
    lru_counter = 0;
  }

 private:
  /*!
   * \brief The core data structure radix tree.
   */
  PagedRadixTree radix_tree;
  /*!
   * \brief The map from sequence to LRU time stamps.
   */
  std::unordered_map<int64_t, size_t> latest_visit;
  /*!
   * \brief The rset of the pair of LRU time stamps and recycling sequence ID. Used to get the
   * oldest recycling sequence.
   */
  std::set<std::pair<size_t, int64_t>> recycling_seqs;
  /*!
   * \brief The recycle callback functions to invoke when removing the sequence.
   * e.g. it can be removing sequence in the KVCache of each model, and recycling the seuqence ID
   * back to ID manager.
   */
  std::unordered_map<int64_t, PackedFunc> recycle_callbacks;
  /*!
   * \brief The flag whether to enable sliding windos.
   */
  bool sliding_window = false;
  /*!
   * \brief The maximum forking offset, enabled with sliding window, and set by attention sink
   * position.
   */
  size_t max_fork_offset = 0;
  /*!
   * \brief The maximum number of recycling sequence.
   */
  int max_recycling_seqs = -1;
  /*!
   * \brief The LRU counter.
   */
  size_t lru_counter = 0;
};

TVM_REGISTER_OBJECT_TYPE(PrefixCacheImpl);

PrefixCache PrefixCache::Init(size_t num_pages, size_t page_size, size_t num_seqs,
                              int sliding_window_size, int attention_sink_size) {
  ObjectPtr<PrefixCacheImpl> n = make_object<PrefixCacheImpl>(
      num_pages, page_size, num_seqs, sliding_window_size, attention_sink_size);
  return PrefixCache(std::move(n));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
