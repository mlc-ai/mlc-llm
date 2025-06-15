/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/radix_tree.cc
 */
#include "radix_tree.h"

#include <tvm/ffi/function.h>
#include <tvm/runtime/logging.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The sequence ID linked list structure in paged radix tree node.
 */
struct SequenceIDNode {
  /*! \brief The stored sequence ID. */
  int64_t id = 0;
  /*! \brief The pointer to the next sequence ID. */
  SequenceIDNode* next = nullptr;
};

/*!
 * \brief The sequence ID node pool.
 *
 * The sequence ID node pool allocates a block of sequence ID nodes when pool is full,
 * and frees all when destruction, to avoid frequent memory operation.
 */
class SequenceIDNodePool {
 public:
  /*! \brief The constructor of sequence ID node pool, allocating a new sequence ID node block. */
  SequenceIDNodePool() {
    NewNodeBlock_();
    used_nodes_.clear();
  }

  /*!
   * \brief Get a sequence ID node from pool, and assign the fields.
   * If there is no available node, it will allocate a new sequence ID node block.
   * \param seq_id The assigned sequence ID of allocated sequence ID node.
   * \param node The next sequence ID node pointer of allocated sequence ID node.
   * \return The allocated radix page.
   */
  SequenceIDNode* Allocate(int64_t seq_id, SequenceIDNode* next) {
    if (free_node_indices_.empty()) {
      NewNodeBlock_();
      CHECK(!free_node_indices_.empty());
    }
    size_t id = free_node_indices_.back();
    free_node_indices_.pop_back();
    SequenceIDNode* node = nodes_[id];
    used_nodes_[node] = id;
    node->id = seq_id;
    node->next = next;
    return node;
  }

  /*!
   * \brief Free a sequence ID node to pool.
   * \param node The sequence ID node to free.
   */
  void Free(SequenceIDNode* node) {
    CHECK(used_nodes_.find(node) != used_nodes_.end());
    free_node_indices_.push_back(used_nodes_[node]);
    used_nodes_.erase(node);
  }

  /*!
   * \brief Reset the sequence ID node pool to initial status.
   */
  void Reset() {
    used_nodes_.clear();
    free_node_indices_.reserve(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
      nodes_[i]->id = 0;
      nodes_[i]->next = nullptr;
      free_node_indices_[i] = i;
    }
  }

  /*! \brief The destructor of sequence ID node pool, freeing memory for each node. */
  ~SequenceIDNodePool() {
    for (SequenceIDNode* node_block : node_blocks_) {
      delete[] node_block;
    }
  }

 private:
  /*! \brief The size of each node pool block. */
  static constexpr size_t kNodeBlockSize_ = 64;
  /*! \brief The raw sequence ID node block pool, each element is a sequence ID node array. */
  std::vector<SequenceIDNode*> node_blocks_;
  /*! \brief The sequence ID node pool, each element is a sequence ID node pointer. */
  std::vector<SequenceIDNode*> nodes_;
  /*! \brief The indices of free sequence ID node in node pool. */
  std::vector<size_t> free_node_indices_;
  /*! \brief The map from used paged sequence ID node to its index in node pool. */
  std::unordered_map<SequenceIDNode*, size_t> used_nodes_;

  /*! \brief Allocate a new node pool block. */
  void NewNodeBlock_() {
    size_t node_id_offset = node_blocks_.size() * kNodeBlockSize_;
    node_blocks_.push_back(new SequenceIDNode[kNodeBlockSize_]);
    nodes_.reserve(nodes_.size() + kNodeBlockSize_);
    free_node_indices_.reserve(free_node_indices_.size() + kNodeBlockSize_);
    for (size_t i = 0; i < kNodeBlockSize_; ++i) {
      nodes_.push_back(&node_blocks_.back()[i]);
      free_node_indices_.push_back(i + node_id_offset);
    }
  }
};

/*!
 * \brief The paged radix tree node data structure.
 *
 * The paged radix tree node is similar to original radix tree node, but with the limited length for
 * prefix in page, so that the memory usage in each page is the same and is fixed once allocated.
 * Since the page only consists of pointers and int tokens, the page memory layout is int array
 * indeed. The lower offset is the pointers and page information, while the higher offset is the
 * stored prefix tokens.
 *
 * And since the vocabulary size may be very large, the paged Radix tree is represented
 * as left-child, right-sibling binary tree.
 *
 * Also, due to possible pop/push front/back tokens in page, the page is designed as circular
 * buffer, to make full use of each page.
 *
 * Each page records the sequence exactly ends with the prefix tokens stored in page. In other word,
 * all sequences locate in the boundary of each page, or the end of each page.
 */
struct RadixPage {
  /*! \brief The parent page. */
  RadixPage* parent;
  /*! \brief The first child page. */
  RadixPage* first_child;
  /*! \brief The sibling page sharing the same parent page. */
  RadixPage* next_sibling;
  /*! \brief The head of sequence ID linked list. */
  SequenceIDNode* seq_ids;
  /*! \brief The capacity of maximum stored prefix tokens. */
  size_t capacity;
  /*! \brief The start offset of stored prefix tokens. The legal value is of [0, capacity). */
  size_t offset;
  /*! \brief The length of stored prefix tokens. The legal value is of [0, capacity). */
  size_t length;
  /*! \brief The offset of first prefix token in memory layout. */
  static constexpr int kDataOffset = (sizeof(RadixPage*) * 3 + sizeof(SequenceIDNode*) +
                                      sizeof(size_t) * 3 + sizeof(int32_t) - 1) /
                                     sizeof(int32_t);

  /*!
   * \brief Overload operator [] to get the prefix tokens by index as simple int array.
   * \param i The prefix token index.
   * \return The value of i-th prefix token.
   */
  int32_t& operator[](size_t i) {
    return reinterpret_cast<int32_t*>(this)[kDataOffset + (i + offset) % capacity];
  }

  /*!
   * \brief Extend or push back a suffix tokens in page.
   * \param suffix The suffix tokens array.
   * \param suffix_length The suffix length to extend.
   * \throw Error if suffix length is larger than current vacant space.
   */
  void Extend(const int32_t* suffix, size_t suffix_length) {
    CHECK_LE(suffix_length + length, capacity);
    for (int i = 0; i < suffix_length; ++i) {
      (*this)[i + length] = suffix[i];
    }
    length += suffix_length;
  }

  /*!
   * \brief Add a sequence ID in page.
   * \param pool The sequence ID node pool to allocate new node.
   * \param id The sequence ID to add.
   */
  void AddSequence(SequenceIDNodePool* pool, int64_t id) { seq_ids = pool->Allocate(id, seq_ids); }

  /*!
   * \brief Pop a sequence ID in page.
   * \param pool The sequence ID node pool to free popped node.
   * \param id The sequence ID to pop.
   * \throw Error if no such sequence ID in page.
   */
  void PopSequence(SequenceIDNodePool* pool, int64_t id) {
    if (seq_ids->id == id) {
      // If the popped sequence ID is the first node in linked list,
      // directly skip from head and free it.
      SequenceIDNode* next = seq_ids->next;
      pool->Free(seq_ids);
      seq_ids = next;
    } else {
      // If the popped sequence ID is not the first node in linked list,
      // skip it from previous node and free it.
      SequenceIDNode* last = seq_ids;
      SequenceIDNode* cur = seq_ids->next;
      while (cur) {
        if (cur->id == id) {
          last->next = cur->next;
          pool->Free(cur);
          return;
        }
        last = cur;
        cur = cur->next;
      }
      LOG(FATAL) << "Sequence ID = " << id << " not found.";
    }
  }

  /*!
   * \brief Get all sequence ID in page.
   * \return The std::vector of sequence ID in page.
   */
  std::vector<int64_t> GetLocalSequence() {
    std::vector<int64_t> output;
    for (SequenceIDNode* node = seq_ids; node; node = node->next) {
      output.push_back(node->id);
    }
    return output;
  }

  /*!
   * \brief Get any sequence ID in current page or child pages.
   * Since there is always a sequence in leaf pages, it only check first child if no sequence ID in
   * current page.
   * \return The any sequence ID in current page or child pages.
   */
  int32_t FindAnyChildSequence() {
    if (seq_ids) return seq_ids->id;
    return first_child->FindAnyChildSequence();
  }

  /*!
   * \brief Get all sequence ID in current page and child pages, using Iterate method with lambda
   * expression as callback to avoid frequently memory allocation of std::vector.
   * \return The std::vector of all sequence ID in current page and child pages.
   */
  std::vector<int64_t> FindAllChildSequence() {
    std::vector<int64_t> output = GetLocalSequence();
    if (first_child) {
      first_child->Iterate([&output](const RadixPage* page) {
        for (SequenceIDNode* node = page->seq_ids; node; node = node->next) {
          output.push_back(node->id);
        }
      });
    }
    return output;
  }

  /*!
   * \brief The iteration method for tree or sub-tree traverse.
   * \param f The callback function to invoke at each radix page visited.
   */
  template <class CallbackFunc>
  void Iterate(CallbackFunc f) {
    f(this);
    if (next_sibling) next_sibling->Iterate(f);
    if (first_child) first_child->Iterate(f);
  }

  /*!
   * \brief Get the last sibling of current page.
   * \return The page whose next_sibling is current page, or nullptr if current is the first_child
   * of its parent page.
   */
  RadixPage* GetLastSibling() {
    if (parent == nullptr) return nullptr;
    if (parent->first_child == this) return nullptr;
    for (RadixPage* child = parent->first_child; child; child = child->next_sibling) {
      if (child->next_sibling == this) return child;
    }
    return nullptr;
  }

  /*!
   * \brief Find the child indexed by first token.
   * \return The child page started with first token, or nullptr if no such child page.
   */
  RadixPage* FindChild(int64_t first_token) {
    int32_t casted = first_token;
    // Iterate all child radix pages, as the child radix pages are stored unorderly.
    for (RadixPage* child = first_child; child; child = child->next_sibling) {
      if ((*child)[0] == casted) return child;
    }
    return nullptr;
  }

  /*! \brief Insert a new child page. */
  void InsertChild(RadixPage* child) {
    child->parent = this;
    child->next_sibling = first_child;
    first_child = child;
  }

  /*!
   * \brief Remove a child page.
   * \throw Error if page to be removed is not child page.
   */
  void RemoveChild(RadixPage* child) {
    CHECK(child->parent == this);
    if (first_child == child) {
      first_child = child->next_sibling;
    } else {
      child->GetLastSibling()->next_sibling = child->next_sibling;
    }
  }

  /*!
   * \brief Check current page is mergable with its child page.
   * The page is mergable if and only if
   * 1. No sequence ID in current page, as sequence ID is not allowed to exist within page.
   * 2. The current page has child page.
   * 3. The current page has only one child page.
   * 4. The current page prefix and the child page prefix can be concatenated into one page.
   * \return True if current page is mergable, or false.
   */
  bool Mergeable() {
    if (seq_ids) return false;
    if (!first_child) return false;
    if (first_child->next_sibling) return false;
    if (length + first_child->length > capacity) return false;
    return true;
  }

  /*!
   * \brief Match the given prefix within page.
   * \param prefix The prefix token array.
   * \param prefix_length The length of prefix token array.
   * \return The matched prefix offset within page, or the first mismatched token position. The
   * possible return value is [0, page->length], where page->length means the page is completely the
   * prefix of given prefix.
   */
  size_t MatchPrefix(const int32_t* prefix, size_t prefix_length) {
    size_t n = std::min(length, prefix_length);
    for (int i = 0; i < n; ++i) {
      if ((*this)[i] != prefix[i]) return i;
    }
    return n;
  }
};

/*!
 * \brief The paged radix tree page pool.
 *
 * The paged radix tree page pool allocates a block of radix tree pages when pool is full,
 * and frees all when destruction, to avoid frequent memory operation.
 */
class RadixPagePool {
 public:
  /*! \brief The constructor of paged radix tree page pool, allocating memory for each page. */
  RadixPagePool() {
    NewPageBlock_();
    used_pages_.clear();
  }

  /*!
   * \brief Get a radix page from pool.
   * If there is no available page, it will allocate a new radix page block.
   * \return The allocated radix page.
   */
  RadixPage* Allocate() {
    if (free_page_indices_.empty()) {
      NewPageBlock_();
      CHECK(!free_page_indices_.empty());
    }
    int id = free_page_indices_.back();
    free_page_indices_.pop_back();
    RadixPage* page = pages_[id];
    used_pages_[page] = id;
    page->parent = page->first_child = page->next_sibling = nullptr;
    page->capacity = kPageCapacity_;
    page->offset = page->length = 0;
    page->seq_ids = nullptr;
    return page;
  }

  /*!
   * \brief Free a radix page to pool.
   * \param page The radix page to free.
   */
  void Free(RadixPage* page) {
    CHECK_EQ(page->seq_ids, nullptr);
    CHECK(used_pages_.find(page) != used_pages_.end());
    free_page_indices_.push_back(used_pages_[page]);
    CHECK(used_pages_.erase(page));
  }

  /*!
   * \brief Get the token capacity of free pages.
   * \return The the token capacity of free pages.
   */
  size_t FreeCapacity() { return free_page_indices_.size() * kPageCapacity_; }

  /*!
   * \brief Reset the paged radix tree page pool to initial status.
   */
  void Reset() {
    used_pages_.clear();
    free_page_indices_.reserve(pages_.size());
    for (int i = 0; i < pages_.size(); ++i) {
      pages_[i]->parent = pages_[i]->first_child = pages_[i]->next_sibling = nullptr;
      pages_[i]->capacity = kPageCapacity_;
      pages_[i]->offset = pages_[i]->length = 0;
      pages_[i]->seq_ids = nullptr;
      free_page_indices_[i] = i;
    }
  }

  /*! \brief The destructor of paged radix tree page pool, freeing memory for each page. */
  ~RadixPagePool() {
    for (int32_t* page_block : page_blocks_) {
      delete[] page_block;
    }
  }

 private:
  /*! \brief The size of each page pool block. */
  static constexpr size_t kPageBlockSize_ = 64;
  /*! \brief The page capacity of each paged radix tree page. */
  static constexpr size_t kPageCapacity_ = 64;
  /*! \brief The page size of each paged radix tree page. */
  static constexpr size_t kPageSize_ = kPageCapacity_ + RadixPage::kDataOffset;
  /*! \brief The raw paged radix tree page block pool,
  each element is a raw paged radix tree page array. */
  std::vector<int32_t*> page_blocks_;
  /*! \brief The paged radix tree page pool,
  each element is a raw paged radix tree page pointer. */
  std::vector<RadixPage*> pages_;
  /*! \brief The indices of free paged radix page in page pool. */
  std::vector<size_t> free_page_indices_;
  /*! \brief The map from used paged radix tree page to its index in page pool. */
  std::unordered_map<RadixPage*, size_t> used_pages_;

  /*! \brief Allocate a new page pool block. */
  void NewPageBlock_() {
    size_t page_id_offset = page_blocks_.size() * kPageBlockSize_;
    page_blocks_.push_back(new int32_t[kPageBlockSize_ * kPageSize_]);
    pages_.reserve(pages_.size() + kPageBlockSize_);
    free_page_indices_.reserve(free_page_indices_.size() + kPageBlockSize_);
    for (size_t i = 0; i < kPageBlockSize_; ++i) {
      pages_.push_back(reinterpret_cast<RadixPage*>(page_blocks_.back() + i * kPageSize_));
      free_page_indices_.push_back(i + page_id_offset);
    }
  }
};

// PagedRadixTree

/*!
 * \brief The paged radix tree data structure.
 */
class PagedRadixTreeImpl : public PagedRadixTreeObj {
 public:
  /*! \brief The map from sequence to paged radix tree node it is stored. */
  std::unordered_map<int32_t, RadixPage*> seq2page;
  /*! \brief The sequence ID node pool. */
  SequenceIDNodePool* seq_id_node_pool = nullptr;
  /*! \brief The radix page pool. */
  RadixPagePool* radix_page_pool = nullptr;
  /*! \brief The root page of paged radix tree. */
  RadixPage* root = nullptr;

  explicit PagedRadixTreeImpl() {
    seq_id_node_pool = new SequenceIDNodePool();
    radix_page_pool = new RadixPagePool();

    root = reinterpret_cast<RadixPage*>(new int32_t[RadixPage::kDataOffset]);
    root->parent = root->first_child = root->next_sibling = nullptr;
    root->offset = root->length = root->capacity = 0;
    root->seq_ids = nullptr;
  }

  /*!
   * \brief Check if a sequence exists.
   * \param seq_id The sequence ID for index.
   * \return The sequence existence.
   * \throw Error if sequence ID is not valid.
   */
  bool HasSequence(int64_t seq_id) { return seq2page.find(seq_id) != seq2page.end(); }

  /*!
   * \brief Get a sequence's all tokens.
   * \param seq_id The sequence ID for index.
   * \return The sequence tokens.
   * \throw Error if sequence ID is not valid.
   */
  IntTuple GetSequence(int64_t seq_id) {
    CHECK(seq2page.find(seq_id) != seq2page.end());
    size_t length = GetSequenceLength(seq_id);
    std::vector<int64_t> output(length);
    size_t offset = length;
    for (RadixPage* page = seq2page[seq_id]; page; page = page->parent) {
      offset -= page->length;
      for (int i = 0; i < page->length; ++i) {
        output[offset + i] = (*page)[i];
      }
    }
    return IntTuple(output);
  }

  /*!
   * \brief Get all sequences with longest common prefix with give prefix tokens.
   * \param tokens The prefix tokens for reference.
   * \return The pair of matched prefix length and the array of matched sequences indices.
   */
  std::pair<size_t, std::vector<int64_t>> MatchPrefix(const std::vector<int32_t>& tokens) {
    const int32_t* prefix = tokens.data();
    size_t length = tokens.size();
    auto [page, offset, in_page_offset] = MatchSequence(root, prefix, length);
    if (!offset) return std::make_pair(0, std::vector<int64_t>());
    return std::make_pair(offset, page->FindAllChildSequence());
  }

  /*!
   * \brief Get a sequence's length.
   * \param seq_id The sequence ID for index.
   * \return The sequence length.
   * \throw Error if sequence ID is not valid.
   */
  size_t GetSequenceLength(int64_t seq_id) {
    CHECK(seq2page.find(seq_id) != seq2page.end());
    size_t length = 0;
    for (RadixPage* page = seq2page[seq_id]; page; page = page->parent) {
      length += page->length;
    }
    return length;
  }

  /*!
   * \brief Fork a sequence from parent sequence at given position.
   * \param seq_id The new sequence ID.
   * \param parent_seq_id The parent sequence ID to fork from.
   * \param forked_offset The position of parent sequence to fork at.
   * The valid value is [1, length of forked sequence]. If the position equals the length of forked
   * sequence, the new sequence will copy the entire forked sequence.
   * \throw Error if sequence ID or
   * forked postion is not valid.
   */
  void ForkSequence(int64_t seq_id, int64_t parent_seq_id, size_t forked_offset) {
    CHECK(seq2page.find(seq_id) == seq2page.end());
    CHECK(seq2page.find(parent_seq_id) != seq2page.end());
    CHECK_GT(forked_offset, 0);
    size_t length = GetSequenceLength(parent_seq_id);
    CHECK_LE(forked_offset, length);
    for (RadixPage* page = seq2page[parent_seq_id]; page; page = page->parent) {
      if (forked_offset > length - page->length) {
        if (forked_offset < length) {
          // Split radix page if forked position is within page
          page = SplitPage(page, forked_offset + page->length - length);
        }
        page->AddSequence(seq_id_node_pool, seq_id);
        seq2page[seq_id] = page;
        return;
      }
      length -= page->length;
    }
  }

  /*!
   * \brief Add an empty sequence at root.
   * \param seq_id The new sequence ID.
   * \throw Error if sequence ID is not valid.
   */
  void AddSequence(int64_t seq_id) {
    CHECK(seq2page.find(seq_id) == seq2page.end())
        << "Sequence ID = " << seq_id << " has been added.";
    root->AddSequence(seq_id_node_pool, seq_id);
    seq2page[seq_id] = root;
  }

  /*!
   * \brief Extend a sequence with given tokens.
   * \param seq_id The sequence ID for index.
   * \param tokens The given tokens to extend.
   * \throw Error if sequence ID is not valid.
   */
  void ExtendSequence(int64_t seq_id, const std::vector<int32_t>& tokens) {
    CHECK(seq2page.find(seq_id) != seq2page.end());
    const int32_t* suffix = tokens.data();
    size_t length = tokens.size();
    RadixPage* original_page = seq2page[seq_id];
    original_page->PopSequence(seq_id_node_pool, seq_id);
    auto [page, offset, in_page_offset] = MatchSequence(original_page, suffix, length);
    if (in_page_offset < page->length) {
      // Split page if extended sequence mismatches within page
      page = SplitPage(page, in_page_offset);
    }
    if (offset < length && !page->seq_ids && !page->first_child && page->capacity > page->length) {
      // Extend in the existing leaf page first if possible.
      size_t suffix_length = std::min(page->capacity - page->length, length - offset);
      page->Extend(suffix + offset, suffix_length);
      offset += suffix_length;
    }
    while (offset < length) {
      // Allocate new radix page and extend tokens
      RadixPage* new_page = radix_page_pool->Allocate();
      page->InsertChild(new_page);
      page = new_page;
      size_t suffix_length = std::min(page->capacity - page->length, length - offset);
      page->Extend(suffix + offset, suffix_length);
      offset += suffix_length;
    }
    page->AddSequence(seq_id_node_pool, seq_id);
    seq2page[seq_id] = page;
    if (original_page->Mergeable()) {
      // The original page may be mergeable, as the sequence ID changes
      MergePage(original_page);
    }
  }

  /*!
   * \brief Roll back a sequence by number of tokens.
   * \param seq_id The sequence ID for index.
   * \param num_tokens The number of tokens to be rolled back.
   * \throw Error if sequence ID is not valid.
   */
  void RollBackSequence(int64_t seq_id, size_t num_tokens) {
    size_t length = GetSequenceLength(seq_id);
    CHECK_GT(num_tokens, 0);
    CHECK_LE(num_tokens, length);
    if (num_tokens == length) {
      // If rolling back whole sequence, just remove the sequence and add it again equivalently.
      RemoveSequence(seq_id);
      AddSequence(seq_id);
      return;
    }
    RadixPage* page = seq2page[seq_id];
    // Remove the sequence temporarily, but keeping the data and starting rolling back.
    page->PopSequence(seq_id_node_pool, seq_id);
    seq2page.erase(seq_id);
    while (page->length <= num_tokens) {
      // Roll back entire page
      num_tokens -= page->length;
      RadixPage* parent = page->parent;
      if (page->seq_ids == nullptr && page->first_child == nullptr) {
        // The leaf page is removable
        parent->RemoveChild(page);
        radix_page_pool->Free(page);
      }
      page = parent;
    }
    if (page->seq_ids == nullptr && page->first_child == nullptr) {
      // The page is leaf page, directly roll back in page length
      page->length -= num_tokens;
      // Update the mapping from sequence to page
      page->AddSequence(seq_id_node_pool, seq_id);
      seq2page[seq_id] = page;
      return;
    }
    // Split page for rolled back sequence
    if (num_tokens) {
      page = SplitPage(page, page->length - num_tokens);
    }
    // Update the mapping from sequence to page
    page->AddSequence(seq_id_node_pool, seq_id);
    seq2page[seq_id] = page;
  }

  /*!
   * \brief Remove a sequence.
   * \param seq_id The sequence ID to remove.
   * \throw Error if sequence ID is not valid.
   */
  void RemoveSequence(int64_t seq_id) {
    RadixPage* page = seq2page[seq_id];
    page->PopSequence(seq_id_node_pool, seq_id);
    seq2page.erase(seq_id);
    while (page->parent && !page->seq_ids && !page->first_child) {
      RadixPage* parent = page->parent;
      parent->RemoveChild(page);
      radix_page_pool->Free(page);
      page = parent;
    }
    if (page && page->Mergeable()) {
      // The remaining page may be mergeable, as the sequence ID changes
      MergePage(page);
    }
  }

  /*!
   * \brief Get the remaining token capacity of the paged radix tree.
   * \return The the remaining token capacity of the paged radix tree.
   */
  size_t FreeCapacity() { return radix_page_pool->FreeCapacity(); }

  void Reset() {
    radix_page_pool->Reset();
    seq_id_node_pool->Reset();
    seq2page.clear();
    root->parent = root->first_child = root->next_sibling = nullptr;
    root->offset = root->length = root->capacity = 0;
    root->seq_ids = nullptr;
  }

  /*! \brief The destructor to free root page. */
  ~PagedRadixTreeImpl() {
    delete[] reinterpret_cast<int32_t*>(root);
    delete seq_id_node_pool;
    delete radix_page_pool;
  }

 private:
  /*!
   * \brief Merge a radix tree page with its child radix tree page, to save radix tree page.
   * e.g. MergePage([1, 2, _, _, _] -> [3, 4, 5, _, _]) = [1, 2, 3, 4, 5].
   * And the page to be merged should be page->Mergeable().
   * \param page The parent radix tree page.
   */
  void MergePage(RadixPage* page) {
    CHECK(page->Mergeable());
    RadixPage* child = page->first_child;
    for (int i = 0; i < child->length; ++i) {
      (*page)[i + page->length] = (*child)[i];
    }
    page->length += child->length;
    page->first_child = child->first_child;
    for (RadixPage* p = child->first_child; p; p = p->next_sibling) {
      p->parent = page;
    }
    page->seq_ids = child->seq_ids;
    std::vector<int64_t> seq_ids = page->GetLocalSequence();
    for (int64_t id : seq_ids) seq2page[id] = page;
    child->seq_ids = nullptr;
    radix_page_pool->Free(child);
  }

  /*!
   * \brief Split a radix tree page at given position, to accept new sequence.
   * e.g. SplitPage([1, 2, 3, 4, 5], 2) = [1, 2, _, _, _] -> [3, 4, 5, _, _].
   * \param page The radix tree page to split.
   * \param offset The position to split the radix tree page.
   * \return The splitted radix tree page. It can be different from the input radix tree page, as
   * there may be implicit radix tree page merge.
   */
  RadixPage* SplitPage(RadixPage* page, size_t offset) {
    CHECK_LT(offset, page->length);
    RadixPage* child = radix_page_pool->Allocate();
    child->parent = page;
    child->first_child = page->first_child;
    for (RadixPage* p = page->first_child; p; p = p->next_sibling) {
      p->parent = child;
    }
    page->first_child = child;
    for (int i = offset; i < page->length; ++i) {
      (*child)[i - offset] = (*page)[i];
    }
    child->length = page->length - offset;
    page->length = offset;
    child->seq_ids = page->seq_ids;
    std::vector<int64_t> seq_ids = page->GetLocalSequence();
    for (int64_t id : seq_ids) seq2page[id] = child;
    page->seq_ids = nullptr;
    if (child->Mergeable()) {
      // The child page may be mergeable
      MergePage(child);
    }
    if (page->parent && page->parent->Mergeable()) {
      // The parent page may be mergeable
      page = page->parent;
      MergePage(page);
    }
    return page;
  }

  /*!
   * \brief Match with given token from a radix tree page, stopping at first mismatch.
   * \param page The radix tree page to start matching.
   * \param tokens The given tokens to match.
   * \param length The length of given tokens.
   */
  std::tuple<RadixPage*, size_t, size_t> MatchSequence(RadixPage* page, const int32_t* tokens,
                                                       size_t length) {
    size_t offset = 0;
    while (offset < length) {
      if (RadixPage* child = page->FindChild(tokens[offset])) {
        // If child page starts with offset-th token, common prefix at least ends with child page
        size_t matched_offset = child->MatchPrefix(tokens + offset, length - offset);
        offset += matched_offset;
        if (matched_offset < child->length) {
          // Common prefix ends within child page
          return std::make_tuple(child, offset, matched_offset);
        }
        page = child;
      } else {
        // No child page starts with offset-th token, common prefix ends with current page
        return std::make_tuple(page, offset, page->length);
      }
    }
    return std::make_tuple(page, length, page->length);
  }
};

TVM_REGISTER_OBJECT_TYPE(PagedRadixTreeImpl);

PagedRadixTree PagedRadixTree::Create() {
  return PagedRadixTree(tvm::ffi::make_object<PagedRadixTreeImpl>());
}

TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTree").set_body_typed([]() {
  return PagedRadixTree::Create();
});
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeMatchPrefix")
    .set_body_typed([](PagedRadixTree paged_radix_tree, IntTuple tokens) {
      std::vector<int32_t> token_ids{tokens.begin(), tokens.end()};
      auto [offset, seq_ids] = paged_radix_tree->MatchPrefix(token_ids);
      seq_ids.insert(seq_ids.begin(), offset);
      return IntTuple(seq_ids);
    });
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeExtendSequence")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id, IntTuple tokens) {
      std::vector<int32_t> token_ids{tokens.begin(), tokens.end()};
      paged_radix_tree->ExtendSequence(seq_id, std::move(token_ids));
    });
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeRollBackSequence")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id, int64_t num_tokens) {
      paged_radix_tree->RollBackSequence(seq_id, num_tokens);
    });
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeForkSequence")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id, int64_t parent_seq_id,
                       uint64_t forked_offset) {
      paged_radix_tree->ForkSequence(seq_id, parent_seq_id, forked_offset);
    });
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeHasSequence")
    .set_body_method(&PagedRadixTreeObj::HasSequence);
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeAddSequence")
    .set_body_method(&PagedRadixTreeObj::AddSequence);
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeRemoveSequence")
    .set_body_method(&PagedRadixTreeObj::RemoveSequence);
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeGetSequence")
    .set_body_method(&PagedRadixTreeObj::GetSequence);
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeGetSequenceLength")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id) {
      return (int64_t)paged_radix_tree->GetSequenceLength(seq_id);
    });
TVM_FFI_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeFreeCapacity")
    .set_body_typed([](PagedRadixTree paged_radix_tree) {
      return (int64_t)paged_radix_tree->FreeCapacity();
    });
}  // namespace serve
}  // namespace llm
}  // namespace mlc
