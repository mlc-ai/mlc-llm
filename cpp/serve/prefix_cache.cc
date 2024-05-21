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

/*!
 * \brief The implementation of prefix cache.
 */
class PrefixCacheImpl : public PrefixCacheObj {
 public:
  /*!
   * \brief Contructor of paged radix tree.
   * \param max_num_seqs The maximum number of sequence ID.
   * \param sliding_window_size The sliding window size, -1 for disabled sliding window.
   * \param attention_sink_size The attention sink position for sliding window.
   */
  explicit PrefixCacheImpl(size_t max_num_seqs,
                           std::optional<TypedPackedFunc<void(int64_t)>> remove_callback)
      : radix_tree(PagedRadixTree::Create()),
        max_num_seqs(max_num_seqs),
        remove_callback(remove_callback) {
    recycling_seq_lrus.clear();
    reversed_recycling_seq_lrus.clear();
    seq_states.clear();
    seq_sliding_window_infos.clear();
    lru_counter = 0;
  }

  /*!
   * \brief Insert a new tokenized sequence into Prefix Cache.
   * \param tokens The tokens of tokenized sequence.
   * \return The matched result.
   */
  PrefixCacheMatchedResult InsertSequence(int64_t seq_id, IntTuple tokens, int sliding_window_size,
                                          int attention_sink_size) {
    if (seq_states.size() == max_num_seqs) {
      // If prefix cache has reached maximum number of sequences, try to pop one recycling sequence.
      CHECK(TryFreeMemory())
          << "PrefixCache has reached the maximum number of sequences, and no recycling sequence "
             "to be popped for new sequence. Please set larger value for maximum number of "
             "sequences, or reduce the number of running sequence, to align with maximum number of "
             "sequence in PrefixCache.";
      CHECK_EQ(seq_states.size(), max_num_seqs - 1);
    }
    CHECK_NE(sliding_window_size, 0);
    CHECK_GE(attention_sink_size, 0);
    CHECK(seq_states.find(seq_id) == seq_states.end());
    CHECK(seq_sliding_window_infos.find(seq_id) == seq_sliding_window_infos.end());
    std::pair<int, size_t> sliding_window_info{sliding_window_size, attention_sink_size};
    IntTuple popped_tokens = IntTuple(std::vector<int64_t>(tokens.begin(), tokens.end() - 1));
    auto [matched_offset, matched_seqs] = radix_tree->MatchPrefix(popped_tokens);
    // No prefix matched, directly adding new sequence.
    if (!matched_offset) {
      radix_tree->AddSequence(seq_id);
      seq_states.emplace(seq_id, SequenceState::kActive);
      seq_sliding_window_infos.emplace(seq_id, sliding_window_info);
      return PrefixCacheMatchedResult{seq_id, -1, 0, 0};
    }

    CHECK(!matched_seqs.empty());

    // The reusage of recycling sequences logic is different between with/without sliding window
    // enabled.
    if (sliding_window_size != -1) {
      // If sliding window enabled, the reusage of recycling sequences should be limitted to exactly
      // matched. And no rolling back is allowed due to the sliding window.
      for (int64_t matched_seq_id : matched_seqs) {
        if (seq_states.at(matched_seq_id) == SequenceState::kRecycling &&
            seq_sliding_window_infos.at(matched_seq_id) == sliding_window_info) {
          size_t matched_seq_length = radix_tree->GetSequenceLength(matched_seq_id);
          if (matched_seq_length == matched_offset) {
            ReuseRecyclingSequence(matched_seq_id);
            return PrefixCacheMatchedResult{matched_seq_id, -1, matched_offset, 0};
          }
        }
      }
      // If no sequence reused, we fallback to forking matched sequence. Due to the sliding window,
      // we have to align the matched offset to attention sink size, to avoid forking beyond
      // attention sink size.
      matched_offset = std::min(matched_offset, static_cast<size_t>(attention_sink_size));
    } else {
      // If sliding window is not enabled, we can greedily reuse the shortest recycling sequence
      // without sliding window, so that the loss or roll back of trailing tokens will be minimum.
      size_t shortest_recycling_seq_length = 0;
      int64_t shortest_recycling_seq_id = -1;

      for (int64_t matched_seq_id : matched_seqs) {
        if (seq_states.at(matched_seq_id) == SequenceState::kRecycling &&
            seq_sliding_window_infos.at(matched_seq_id) == sliding_window_info) {
          size_t matched_seq_length = radix_tree->GetSequenceLength(matched_seq_id);
          if (shortest_recycling_seq_id == -1 ||
              matched_seq_length < shortest_recycling_seq_length) {
            shortest_recycling_seq_id = matched_seq_id;
            shortest_recycling_seq_length = matched_seq_length;
          }
        }
      }
      if (shortest_recycling_seq_id != -1) {
        ReuseRecyclingSequence(shortest_recycling_seq_id);
        if (shortest_recycling_seq_length > matched_offset) {
          // Recycling sequence is longer than new sequence, rolling back the redundant trailing
          // tokens, to match the new sequence.
          radix_tree->RollBackSequence(shortest_recycling_seq_id,
                                       shortest_recycling_seq_length - matched_offset);
        }
        return PrefixCacheMatchedResult{shortest_recycling_seq_id, -1, matched_offset,
                                        shortest_recycling_seq_length - matched_offset};
      }
    }
    // No reusage of recycling sequence, fallback to forking matched sequence. However, due to some
    // sequence enabled with sliding window, we can fork them within the first attention sink size.
    // So we fork from the sequence whose fork-able offset is longest.
    size_t longest_forking_offset = 0;
    int64_t longest_forking_seq_id = -1;
    for (int64_t matched_seq_id : matched_seqs) {
      auto [matched_seq_sliding_window_size, matched_seq_attention_sink_size] =
          seq_sliding_window_infos.at(matched_seq_id);
      if (matched_seq_sliding_window_size == -1) {
        // If the matched is not enabled with sliding window, we can fork within matched offset
        // tokens arbitrarily.
        if (matched_offset > longest_forking_offset) {
          longest_forking_offset = matched_offset;
          longest_forking_seq_id = matched_seq_id;
        }
      } else {
        // If the matched is enabled with sliding window, we can fork within effective matched
        // offset tokens, which is the minimum between matched offset and its attention sink size.
        size_t effective_matched_offset = std::min(matched_offset, matched_seq_attention_sink_size);
        if (effective_matched_offset > longest_forking_offset) {
          longest_forking_offset = effective_matched_offset;
          longest_forking_seq_id = matched_seq_id;
        }
      }
    }
    if (longest_forking_offset > 0) {
      radix_tree->ForkSequence(seq_id, longest_forking_seq_id, longest_forking_offset);
      seq_states.emplace(seq_id, SequenceState::kActive);
      seq_sliding_window_infos.emplace(seq_id, sliding_window_info);
      return PrefixCacheMatchedResult{seq_id, longest_forking_seq_id, longest_forking_offset, 0};
    }
    // No forking from matched sequence, fallback to adding new sequence.
    radix_tree->AddSequence(seq_id);
    seq_states.emplace(seq_id, SequenceState::kActive);
    seq_sliding_window_infos.emplace(seq_id, sliding_window_info);
    return PrefixCacheMatchedResult{seq_id, -1, 0, 0};
  }

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * \param seq_id The sequence to be extneded.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if the given sequence id is not valid or active.
   */
  void ExtendSequence(int64_t seq_id, IntTuple tokens) {
    CHECK(seq_states.at(seq_id) == SequenceState::kActive);
    radix_tree->ExtendSequence(seq_id, tokens);
  }

  /*!
   * \brief Roll back a sequence by number of tokens.
   * \param seq_id The sequence ID for index.
   * \param num_tokens The number of tokens to be rolled back.
   * \throw Error if the given sequence id is not valid or active.
   */
  void RollBackSequence(int64_t seq_id, size_t num_tokens) {
    CHECK(seq_states.at(seq_id) == SequenceState::kActive);
    radix_tree->RollBackSequence(seq_id, num_tokens);
  }

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   memory is sufficient. And it will be reused again in the future request.
   * \param seq_id The sequence to be recycled.
   * \param lazy The flag if the sequence should be removed lazily or intermediary.
   * \throw Error if the given sequence id is not valid.
   */
  void RecycleSequence(int64_t seq_id, bool lazy = true) {
    CHECK(seq_states.at(seq_id) == SequenceState::kActive);
    CHECK(recycling_seq_lrus.find(seq_id) == recycling_seq_lrus.end());
    if (lazy) {
      // Remove the sequence lazily.
      seq_states.at(seq_id) = SequenceState::kRecycling;
      ++lru_counter;
      recycling_seq_lrus.emplace(seq_id, lru_counter);
      reversed_recycling_seq_lrus.emplace(lru_counter, seq_id);
    } else {
      // Remove the sequence intermediately.
      radix_tree->RemoveSequence(seq_id);
      if (remove_callback.has_value()) {
        remove_callback.value()(seq_id);
      }
      CHECK(seq_states.erase(seq_id));
      CHECK(seq_sliding_window_infos.erase(seq_id));
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
    if (reversed_recycling_seq_lrus.empty()) {
      // There is no recycling sequence. No memory can be freed.
      return false;
    }
    auto [lru, seq_id] = *reversed_recycling_seq_lrus.begin();
    CHECK(seq_states.at(seq_id) == SequenceState::kRecycling);
    CHECK_EQ(recycling_seq_lrus.at(seq_id), lru);
    radix_tree->RemoveSequence(seq_id);
    if (remove_callback.has_value()) {
      remove_callback.value()(seq_id);
    }
    CHECK(seq_states.erase(seq_id));
    CHECK(recycling_seq_lrus.erase(seq_id));
    CHECK(reversed_recycling_seq_lrus.erase(lru));
    CHECK(seq_sliding_window_infos.erase(seq_id));
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
    recycling_seq_lrus.clear();
    reversed_recycling_seq_lrus.clear();
    seq_states.clear();
    seq_sliding_window_infos.clear();
    lru_counter = 0;
  }

 private:
  void ReuseRecyclingSequence(int64_t seq_id) {
    CHECK(seq_states.at(seq_id) == SequenceState::kRecycling);
    size_t lru = recycling_seq_lrus.at(seq_id);
    CHECK_EQ(reversed_recycling_seq_lrus.at(lru), seq_id);
    seq_states.at(seq_id) = SequenceState::kActive;
    CHECK(recycling_seq_lrus.erase(seq_id));
    CHECK(reversed_recycling_seq_lrus.erase(lru));
  }

  /*!
   * \brief The sequence states.
   */
  enum class SequenceState : int {
    /*!
     * \brief The state of active sequence. In this state, the sequence can be forked only. When
     * recycling a sequence, it will transfer to kRecycling.
     */
    kActive = 0,
    /*!
     * \brief The state of recycling sequence. In this state, the sequence can be forked or be
     * reused. And it will transfer to kActive only when reused.
     */
    kRecycling = 1,
  };
  /*!
   * \brief The core data structure radix tree.
   */
  PagedRadixTree radix_tree;
  /*!
   * \brief The map from sequence to LRU time stamps.
   */
  std::unordered_map<int64_t, size_t> recycling_seq_lrus;
  /*!
   * \brief The map from LRU time stamps to sequence, used to find the sequence with earlist LRU
   * time stamp.
   */
  std::unordered_map<size_t, int64_t> reversed_recycling_seq_lrus;
  /*!
   * \brief The maximum number of sequences in prefix cache.
   */
  int max_num_seqs = -1;
  /*!
   * \brief The LRU counter.
   */
  size_t lru_counter = 0;
  /*!
   * \brief The optional callback function to call when removing a sequence. This can be used to
   * removing sequence in KVCache and return sequence ID to ID manager lazily
   */
  std::optional<TypedPackedFunc<void(int64_t)>> remove_callback = std::nullopt;
  /*!
   * \brief The map from sequence to its sequence states.
   */
  std::unordered_map<int64_t, SequenceState> seq_states;
  /*!
   * \brief The map from sequence to its sliding window information. The sliding window information
   * is a pair of sliding window size and attention sink size. The sliding window size is -1 for
   * sliding window disabled, or positive for sliding window size. The attention sink size is
   * non-negative and used when sliding window size is positive.
   */
  std::unordered_map<int64_t, std::pair<int, size_t>> seq_sliding_window_infos;
};  // namespace serve

TVM_REGISTER_OBJECT_TYPE(PrefixCacheImpl);

/*!
 * \brief The implementation of no prefix cache.
 */
class NoPrefixCache : public PrefixCacheObj {
 public:
  /*!
   * \brief Insert a new tokenized sequence into Prefix Cache.
   * \param tokens The tokens of tokenized sequence.
   * \return Always return as a new sequence.
   */
  PrefixCacheMatchedResult InsertSequence(int64_t seq_id, IntTuple tokens, int sliding_window_size,
                                          int attention_sink_size) {
    return PrefixCacheMatchedResult{seq_id, -1, 0, 0};
  }

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * \param seq_id The sequence to be extneded.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if called since this should never be called.
   */
  void ExtendSequence(int64_t seq_id, IntTuple tokens) { LOG(FATAL) << "Unreachable code."; }

  /*!
   * \brief Roll back a sequence by number of tokens.
   * \param seq_id The sequence ID for index.
   * \param num_tokens The number of tokens to be rolled back.
   * \throw Error if called since this should never be called.
   */
  void RollBackSequence(int64_t seq_id, size_t num_tokens) { LOG(FATAL) << "Unreachable code."; }

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   memory is sufficient. And it will be reused again in the future request.
   * \param seq_id The sequence to be recycled.
   * \param lazy The flag if the sequence should be removed lazily or intermediary.
   * \throw Error if called since this should never be called.
   */
  void RecycleSequence(int64_t seq_id, bool lazy = true) { LOG(FATAL) << "Unreachable code."; }

  /*!
   * \brief Try to remove recycling sequence to free up memory. It will remove the oldest
   recycling sequence.
   * \return Always return false as no sequence stored.
   */
  bool TryFreeMemory() { return false; }

  /*!
   * \brief Check if a sequence exists.
   * \param seq_id The sequence ID for index.
   * \return Always return false as no sequence stored.
   */
  bool HasSequence(int64_t seq_id) { return false; }
};

TVM_REGISTER_OBJECT_TYPE(NoPrefixCache);

PrefixCache PrefixCache::Create(size_t max_num_seqs,
                                std::optional<TypedPackedFunc<void(int64_t)>> remove_callback) {
  if (max_num_seqs == 0) {
    // If maximum number of sequence in prefix cache is 0, prefix cache is not enabled and return a
    // dummy one.
    ObjectPtr<NoPrefixCache> n = make_object<NoPrefixCache>();
    return PrefixCache(std::move(n));
  } else {
    // If maximum number of sequence in prefix cache is positive, prefix cache is enabled.
    ObjectPtr<PrefixCacheImpl> n = make_object<PrefixCacheImpl>(max_num_seqs, remove_callback);
    return PrefixCache(std::move(n));
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
