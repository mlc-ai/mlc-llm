/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/prefix_cache.cc
 */
#include "prefix_cache.h"

#include <tvm/ffi/function.h>
#include <tvm/runtime/nvtx.h>

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
   * \brief Constructor of paged radix tree.
   * \param max_num_recycling_seqs The maximum number of sequences in prefix cache.
   * \param remove_callback The optional callback function to call when removing a sequence.
   */
  explicit PrefixCacheImpl(size_t max_num_recycling_seqs, PrefixCacheRemoveCallback remove_callback)
      : radix_tree_(PagedRadixTree::Create()),
        max_num_recycling_seqs_(max_num_recycling_seqs),
        remove_callback_(std::move(remove_callback)) {
    recycling_seq_lrus_.clear();
    reversed_recycling_seq_lrus_.clear();
    seq_states_.clear();
    seq_sliding_window_infos_.clear();
    lru_counter_ = 0;
  }

  /*!
   * \brief Insert a new tokenized sequence into Prefix Cache.
   * \param seq_id The sequence ID.
   * \param tokens The tokens of tokenized sequence.
   * \param sliding_window_size The sliding window size for the sequence, -1 as sliding window
   * disabled.
   * \param attention_sink_size The attention sink size for the sequence, 0 by default.
   * \return The matched result.
   */
  PrefixCacheMatchedResult InsertSequence(int64_t seq_id, std::vector<int32_t> tokens,
                                          int sliding_window_size, int attention_sink_size) final {
    CHECK_NE(sliding_window_size, 0);
    CHECK_GE(attention_sink_size, 0);
    CHECK(seq_states_.find(seq_id) == seq_states_.end());
    CHECK(seq_sliding_window_infos_.find(seq_id) == seq_sliding_window_infos_.end());
    CHECK(!tokens.empty());
    CommitSequenceExtention();
    tokens.pop_back();
    auto [matched_offset, matched_seqs] = radix_tree_->MatchPrefix(tokens);
    std::pair<int, size_t> sliding_window_info{sliding_window_size, attention_sink_size};
    // No prefix matched, directly adding new sequence.
    if (!matched_offset) {
      radix_tree_->AddSequence(seq_id);
      seq_states_.emplace(seq_id, SequenceState::kActive);
      seq_sliding_window_infos_.emplace(seq_id, sliding_window_info);
      return PrefixCacheMatchedResult{0, -1, -1, 0};
    }

    CHECK(!matched_seqs.empty());

    // The reusage of recycling sequences logic is different between with/without sliding window
    // enabled.
    if (sliding_window_size != -1) {
      // If sliding window enabled, the reusage of recycling sequences should be limited to exactly
      // matched. And no rolling back is allowed due to the sliding window.
      for (int64_t matched_seq_id : matched_seqs) {
        if (seq_states_.at(matched_seq_id) == SequenceState::kRecycling &&
            seq_sliding_window_infos_.at(matched_seq_id) == sliding_window_info) {
          size_t matched_seq_length = radix_tree_->GetSequenceLength(matched_seq_id);
          if (matched_seq_length == matched_offset) {
            ReuseRecyclingSequence(matched_seq_id);
            return PrefixCacheMatchedResult{matched_offset, -1, matched_seq_id, 0};
          }
        }
      }
    } else {
      // If sliding window is not enabled, we can greedily reuse the shortest recycling sequence
      // without sliding window, so that the loss or roll back of trailing tokens will be minimum.
      size_t shortest_recycling_seq_length = 0;
      int64_t shortest_recycling_seq_id = -1;

      for (int64_t matched_seq_id : matched_seqs) {
        if (seq_states_.at(matched_seq_id) == SequenceState::kRecycling &&
            seq_sliding_window_infos_.at(matched_seq_id) == sliding_window_info) {
          size_t matched_seq_length = radix_tree_->GetSequenceLength(matched_seq_id);
          if (shortest_recycling_seq_id == -1 ||
              matched_seq_length < shortest_recycling_seq_length) {
            shortest_recycling_seq_id = matched_seq_id;
            shortest_recycling_seq_length = matched_seq_length;
          }
        }
      }
      if (shortest_recycling_seq_id != -1 && matched_offset > shortest_recycling_seq_length * 0.9) {
        ReuseRecyclingSequence(shortest_recycling_seq_id);
        if (shortest_recycling_seq_length > matched_offset) {
          // Recycling sequence is longer than new sequence, rolling back the redundant trailing
          // tokens, to match the new sequence.
          radix_tree_->RollBackSequence(shortest_recycling_seq_id,
                                        shortest_recycling_seq_length - matched_offset);
        }
        return PrefixCacheMatchedResult{matched_offset, -1, shortest_recycling_seq_id,
                                        shortest_recycling_seq_length - matched_offset};
      }
      // No reusage of recycling sequence, fallback to forking matched sequence. Currently, we only
      // fork from sequence without sliding window, due to current paged KVCache implementation.
      size_t longest_forking_offset = 0;
      int64_t longest_forking_seq_id = -1;
      for (int64_t matched_seq_id : matched_seqs) {
        auto [matched_seq_sliding_window_size, matched_seq_attention_sink_size] =
            seq_sliding_window_infos_.at(matched_seq_id);
        if (matched_seq_sliding_window_size != -1) {
          continue;
        }
        // If the matched is not enabled with sliding window, we can fork within matched offset
        // tokens arbitrarily.
        if (matched_offset > longest_forking_offset) {
          longest_forking_offset = matched_offset;
          longest_forking_seq_id = matched_seq_id;
        }
      }
      if (longest_forking_offset > 0) {
        radix_tree_->ForkSequence(seq_id, longest_forking_seq_id, longest_forking_offset);
        seq_states_.emplace(seq_id, SequenceState::kActive);
        seq_sliding_window_infos_.emplace(seq_id, sliding_window_info);
        return PrefixCacheMatchedResult{longest_forking_offset, longest_forking_seq_id, -1, 0};
      }
    }
    // No forking from matched sequence, fallback to adding new sequence.
    radix_tree_->AddSequence(seq_id);
    seq_states_.emplace(seq_id, SequenceState::kActive);
    seq_sliding_window_infos_.emplace(seq_id, sliding_window_info);
    return PrefixCacheMatchedResult{0, -1, -1, 0};
  }

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * \param seq_id The sequence to be extended.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if the given sequence id is not valid or active.
   */
  void ExtendSequence(int64_t seq_id, const std::vector<int32_t>& tokens) final {
    uncommitted_extended_token_ids_.emplace_back(seq_id, tokens);
  }

  void CommitSequenceExtention() final {
    if (uncommitted_extended_token_ids_.empty()) {
      return;
    }
    NVTXScopedRange nvtx_scope("PrefixCache commit sequence extension");
    for (const auto& [seq_id, uncommitted_token_ids] : uncommitted_extended_token_ids_) {
      if (!HasSequence(seq_id)) {
        // The sequence has been removed. Hence no action is needed.
        continue;
      }
      const auto& it = seq_states_.find(seq_id);
      CHECK(it == seq_states_.end() || it->second == SequenceState::kActive);
      radix_tree_->ExtendSequence(seq_id, uncommitted_token_ids);
    }
    uncommitted_extended_token_ids_.clear();
  }

  /*!
   * \brief Roll back a sequence by number of tokens.
   * \param seq_id The sequence ID for index.
   * \param num_tokens The number of tokens to be rolled back.
   * \throw Error if the given sequence id is not valid or active.
   */
  void RollBackSequence(int64_t seq_id, size_t num_tokens) final {
    CommitSequenceExtention();
    CHECK(seq_states_.at(seq_id) == SequenceState::kActive);
    radix_tree_->RollBackSequence(seq_id, num_tokens);
  }

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   * memory is sufficient and the number of sequence in prefix cache belows the maximum number of
   * sequence. And it will be reused again in the future request.
   * \param seq_id The sequence to be recycled.
   * \param lazy The flag if the sequence should be removed lazily or intermediary.
   * \throw Error if the given sequence id is not valid.
   */
  void RecycleSequence(int64_t seq_id, bool lazy = true) final {
    CommitSequenceExtention();
    CHECK(seq_states_.at(seq_id) == SequenceState::kActive);
    CHECK(recycling_seq_lrus_.find(seq_id) == recycling_seq_lrus_.end());
    if (lazy && max_num_recycling_seqs_ != 0) {
      // Remove the sequence lazily.
      if (recycling_seq_lrus_.size() == max_num_recycling_seqs_) {
        // If prefix cache has reached maximum number of recycling sequences, try to pop one
        // recycling sequence.
        CHECK(TryFreeMemory());
        CHECK_EQ(recycling_seq_lrus_.size(), max_num_recycling_seqs_ - 1);
      }
      seq_states_.at(seq_id) = SequenceState::kRecycling;
      ++lru_counter_;
      recycling_seq_lrus_.emplace(seq_id, lru_counter_);
      reversed_recycling_seq_lrus_.emplace(lru_counter_, seq_id);
    } else {
      // Remove the sequence intermediately.
      radix_tree_->RemoveSequence(seq_id);
      if (remove_callback_ != nullptr) {
        remove_callback_(seq_id);
      }
      CHECK(seq_states_.erase(seq_id));
      CHECK(seq_sliding_window_infos_.erase(seq_id));
    }
  }

  /*!
   * \brief Try to remove recycling sequence to free up memory. It will remove the oldest recycling
   sequence.
   * \return The flag if there is a sequence removed. In other word, return true when memory is
   freed successfully.
   * \throw Error if the given sequence id is not valid.
   */
  bool TryFreeMemory() final {
    NVTXScopedRange nvtx_scope("PrefixCache TryFreeMemory");
    if (reversed_recycling_seq_lrus_.empty()) {
      // There is no recycling sequence. No memory can be freed.
      return false;
    }
    auto [lru, seq_id] = *reversed_recycling_seq_lrus_.begin();
    CHECK(seq_states_.at(seq_id) == SequenceState::kRecycling);
    CHECK_EQ(recycling_seq_lrus_.at(seq_id), lru);
    radix_tree_->RemoveSequence(seq_id);
    if (remove_callback_ != nullptr) {
      remove_callback_(seq_id);
    }
    CHECK(seq_states_.erase(seq_id));
    CHECK(recycling_seq_lrus_.erase(seq_id));
    CHECK(reversed_recycling_seq_lrus_.erase(lru));
    CHECK(seq_sliding_window_infos_.erase(seq_id));
    return true;
  }

  /*!
   * \brief Check if a sequence exists.
   * \param seq_id The sequence ID for index.
   * \return The sequence existence.
   * \throw Error if sequence ID is not valid.
   */
  bool HasSequence(int64_t seq_id) final { return radix_tree_->HasSequence(seq_id); }

  /*!
   * \brief Reset the prefix cache to initial status.
   */
  void Reset() final {
    radix_tree_->Reset();
    recycling_seq_lrus_.clear();
    reversed_recycling_seq_lrus_.clear();
    seq_states_.clear();
    seq_sliding_window_infos_.clear();
    uncommitted_extended_token_ids_.clear();
    lru_counter_ = 0;
  }

  PrefixCacheMode Mode() final { return PrefixCacheMode::kRadix; }

 private:
  void ReuseRecyclingSequence(int64_t seq_id) {
    CHECK(seq_states_.at(seq_id) == SequenceState::kRecycling);
    size_t lru = recycling_seq_lrus_.at(seq_id);
    CHECK_EQ(reversed_recycling_seq_lrus_.at(lru), seq_id);
    seq_states_.at(seq_id) = SequenceState::kActive;
    CHECK(recycling_seq_lrus_.erase(seq_id));
    CHECK(reversed_recycling_seq_lrus_.erase(lru));
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
  PagedRadixTree radix_tree_;
  /*!
   * \brief The map from sequence to LRU time stamps.
   */
  std::unordered_map<int64_t, size_t> recycling_seq_lrus_;
  /*!
   * \brief The map from LRU time stamps to sequence, used to find the sequence with earliest LRU
   * time stamp.
   */
  std::unordered_map<size_t, int64_t> reversed_recycling_seq_lrus_;
  /*!
   * \brief The maximum number of recycling sequences in prefix cache. Set -1 as infinite prefix
   * cache.
   */
  int max_num_recycling_seqs_ = -1;
  /*!
   * \brief The LRU counter.
   */
  size_t lru_counter_ = 0;
  /*!
   * \brief The callback function to call when removing a sequence. This can be used to
   * removing sequence in KVCache and return sequence ID to ID manager lazily
   */
  PrefixCacheRemoveCallback remove_callback_ = nullptr;
  /*!
   * \brief The map from sequence to its sequence states.
   */
  std::unordered_map<int64_t, SequenceState> seq_states_;
  /*!
   * \brief The map from sequence to its sliding window information. The sliding window information
   * is a pair of sliding window size and attention sink size. The sliding window size is -1 for
   * sliding window disabled, or positive for sliding window size. The attention sink size is
   * non-negative and used when sliding window size is positive.
   */
  std::unordered_map<int64_t, std::pair<int, size_t>> seq_sliding_window_infos_;
  /*!
   * \brief The collection of uncommitted extended token ids of sequences.
   * The "ExtendSequence" method only lazily add token ids into this collection,
   * and these uncommitted token ids will be committed when needed.
   *
   * Note: Since the tokens stored are references, CommitSequenceExtention should be called after
   * each action, to avoid the uncaught changes of uncomitted extended token ids.
   */
  std::vector<std::pair<int64_t, const std::vector<int32_t>&>> uncommitted_extended_token_ids_;
};  // namespace serve

TVM_REGISTER_OBJECT_TYPE(PrefixCacheImpl);

/*!
 * \brief The implementation of no prefix cache.
 */
class NoPrefixCache : public PrefixCacheObj {
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
  PrefixCacheMatchedResult InsertSequence(int64_t seq_id, std::vector<int32_t> tokens,
                                          int sliding_window_size, int attention_sink_size) final {
    // Since there is no prefix cache, always return as new sequence.
    return PrefixCacheMatchedResult{0, -1, -1, 0};
  }

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * \param seq_id The sequence to be extended.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if called since this should never be called.
   */
  void ExtendSequence(int64_t seq_id, const std::vector<int32_t>& tokens) final {
    // No-op;
  }

  void CommitSequenceExtention() final {
    // No-op;
  }

  /*!
   * \brief Roll back a sequence by number of tokens.
   * \param seq_id The sequence ID for index.
   * \param num_tokens The number of tokens to be rolled back.
   * \throw Error if called since this should never be called.
   */
  void RollBackSequence(int64_t seq_id, size_t num_tokens) final {
    // Since there is no prefix cache, this method should never be called.
    LOG(FATAL) << "Unreachable code.";
  }

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   * memory is sufficient and the number of sequence in prefix cache belows the maximum number of
   * sequence. And it will be reused again in the future request.
   * \param seq_id The sequence to be recycled.
   * \param lazy The flag if the sequence should be removed lazily or intermediary.
   * \throw Error if the given sequence id is not valid.
   */
  void RecycleSequence(int64_t seq_id, bool lazy = true) final {
    // Since there is no prefix cache, this method should never be called.
    LOG(FATAL) << "Unreachable code.";
  }

  /*!
   * \brief Try to remove recycling sequence to free up memory. It will remove the oldest
   recycling sequence.
   * \return Always return false as no sequence stored.
   */
  bool TryFreeMemory() final {
    // Since there is no prefix cache, always return false.
    return false;
  }

  /*!
   * \brief Check if a sequence exists.
   * \param seq_id The sequence ID for index.
   * \return Always return false as no sequence stored.
   */
  bool HasSequence(int64_t seq_id) final {
    // Since there is no prefix cache, always return false.
    return false;
  }

  /*!
   * \brief Reset the prefix cache to initial status. Do nothing and return.
   */
  void Reset() final {}

  PrefixCacheMode Mode() final { return PrefixCacheMode::kDisable; }
};

TVM_REGISTER_OBJECT_TYPE(NoPrefixCache);

PrefixCache PrefixCache::CreateRadixPrefixCache(size_t max_num_recycling_seqs,
                                                PrefixCacheRemoveCallback remove_callback) {
  ObjectPtr<PrefixCacheImpl> n =
      make_object<PrefixCacheImpl>(max_num_recycling_seqs, std::move(remove_callback));
  return PrefixCache(std::move(n));
}

PrefixCache PrefixCache::CreateNoPrefixCache() {
  ObjectPtr<NoPrefixCache> n = make_object<NoPrefixCache>();
  return PrefixCache(std::move(n));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
