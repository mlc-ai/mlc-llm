/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/radix_tree.cc
 */
#include "radix_tree.h"

#include <tvm/runtime/registry.h>

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
 * The sequence ID node pool allocates all sequence ID nodes when construction and frees when
 * destruction, to avoid frequent memory operation.
 */
class SequenceIDNodePool {
 public:
  /*! \brief The constructor of sequence ID node pool, allocating memory for each node. */
  SequenceIDNodePool(size_t num_nodes) : num_nodes_(num_nodes) {
    nodes_.reserve(num_nodes);
    free_node_indicess_.reserve(num_nodes);
    used_nodes_.clear();
    raw_pool_ = new SequenceIDNode[num_nodes_];
    for (size_t i = 0; i < num_nodes; ++i) {
      nodes_.push_back(&raw_pool_[i]);
      free_node_indicess_.push_back(i);
    }
  }

  /*!
   * \brief Get a radix page from pool, and assign the fields.
   * \param seq_id The assigned sequence ID of allocated sequence ID node.
   * \param node The next sequence ID node pointer of allocated sequence ID node.
   * \return The allocated radix page.
   * \throw Error if no free radix page available in pool.
   */
  SequenceIDNode* Allocate(int64_t seq_id, SequenceIDNode* next) {
    CHECK(!free_node_indicess_.empty()) << "Sequence ID node pool has no free sequence ID nodes.";
    size_t id = free_node_indicess_.back();
    free_node_indicess_.pop_back();
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
    free_node_indicess_.push_back(used_nodes_[node]);
    used_nodes_.erase(node);
  }

  /*!
   * \brief Reset the sequence ID node pool to initial status.
   */
  void Reset() {
    used_nodes_.clear();
    free_node_indicess_.reserve(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
      nodes_[i]->id = 0;
      nodes_[i]->next = nullptr;
      free_node_indicess_[i] = i;
    }
  }

  /*! \brief The destructor of sequence ID node pool, freeing memory for each node. */
  ~SequenceIDNodePool() { delete[] raw_pool_; }

 private:
  /*! \brief The number of nodes in sequence ID node pool. */
  size_t num_nodes_;
  /*! \brief The raw sequence ID node pool. */
  SequenceIDNode* raw_pool_;
  /*! \brief The sequence ID node pool. */
  std::vector<SequenceIDNode*> nodes_;
  /*! \brief The indices of free sequence ID node in node pool. */
  std::vector<size_t> free_node_indicess_;
  /*! \brief The map from used paged sequence ID node to its index in node pool. */
  std::unordered_map<SequenceIDNode*, size_t> used_nodes_;
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
 * Each page records the sequence excatly ends with the prefix tokens stored in page. In other word,
 * all sequences locate in the boundary of each page, or the end of each page.
 */
struct RedixPage {
  /*! \brief The parent page. */
  RedixPage* parent;
  /*! \brief The first child page. */
  RedixPage* first_child;
  /*! \brief The sibling page shareing the same parent page. */
  RedixPage* next_sibiling;
  /*! \brief The head of sequence ID linked list. */
  SequenceIDNode* seq_ids;
  /*! \brief The capacity of maximum stored prefix tokens. */
  size_t capacity;
  /*! \brief The start offset of stored prefix tokens. The legal value is of [0, capacity). */
  size_t offset;
  /*! \brief The length of stored prefix tokens. The legal value is of [0, capacity). */
  size_t length;
  /*! \brief The offset of first prefix token in memory layout. */
  static constexpr int DATA_OFFSET = (sizeof(RedixPage*) * 3 + sizeof(SequenceIDNode*) +
                                      sizeof(size_t) * 3 + sizeof(int32_t) - 1) /
                                     sizeof(int32_t);

  /*!
   * \brief Overload opeartor [] to get the prefix tokens by index as simple int array.
   * \param i The prefix token index.
   * \return The value of i-th prefix token.
   */
  int32_t& operator[](size_t i) {
    return reinterpret_cast<int32_t*>(this)[DATA_OFFSET + (i + offset) % capacity];
  }

  /*!
   * \brief Extend or push back a suffix tokens in page.
   * \param suffix The suffix tokens array.
   * \param suffix_length The suffix length to extend.
   * \throw Error if suffix length is larger than current vacant space.
   */
  void Extend(const int64_t* suffix, size_t suffix_length) {
    CHECK_LE(suffix_length + length, capacity);
    for (int i = 0; i < suffix_length; ++i) {
      (*this)[i + length] = (int32_t)suffix[i];
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
      // If the popped sequencs ID is the first node in linked list,
      // directly skip from head and free it.
      SequenceIDNode* next = seq_ids->next;
      pool->Free(seq_ids);
      seq_ids = next;
    } else {
      // If the popped sequencs ID is not the first node in linked list,
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
      first_child->Iterate([&output](const RedixPage* page) {
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
    if (next_sibiling) next_sibiling->Iterate(f);
    if (first_child) first_child->Iterate(f);
  }

  /*!
   * \brief Get the last sibling of current page.
   * \return The page whose next_sibling is current page, or nullptr if current is the fisrt_child
   * of its parent page.
   */
  RedixPage* GetLastSibling() {
    if (parent == nullptr) return nullptr;
    if (parent->first_child == this) return nullptr;
    for (RedixPage* child = parent->first_child; child; child = child->next_sibiling) {
      if (child->next_sibiling == this) return child;
    }
    return nullptr;
  }

  /*!
   * \brief Find the child indexed by first token.
   * \return The child page started with first token, or nullptr if no such child page.
   */
  RedixPage* FindChild(int64_t first_token) {
    int32_t casted = first_token;
    // Iterate all child radix pages, as the child radix pages are stored unorderly.
    for (RedixPage* child = first_child; child; child = child->next_sibiling) {
      if ((*child)[0] == casted) return child;
    }
    return nullptr;
  }

  /*! \brief Insert a new child page. */
  void InsertChild(RedixPage* child) {
    child->parent = this;
    child->next_sibiling = first_child;
    first_child = child;
  }

  /*!
   * \brief Remove a child page.
   * \throw Error if page to be removed is not child page.
   */
  void RemoveChild(RedixPage* child) {
    CHECK(child->parent == this);
    if (first_child == child) {
      first_child = child->next_sibiling;
    } else {
      child->GetLastSibling()->next_sibiling = child->next_sibiling;
    }
  }

  /*!
   * \brief Check current page is mergable with its child page.
   * The page is mergable if and only if
   * 1. No sequence ID in current page, as sequence ID is not allowed to exist within page.
   * 2. The current page has child page.
   * 3. The current page has only one child page.
   * 4. The current page perfix and the child page prefix can be concatenated into one page.
   * \return True if current page is mergable, or false.
   */
  bool Mergeable() {
    if (seq_ids) return false;
    if (!first_child) return false;
    if (first_child->next_sibiling) return false;
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
  size_t MatchPrefix(const int64_t* prefix, size_t prefix_length) {
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
 * The paged radix tree page pool allocates all radix tree pages when construction and frees when
 * destruction, to avoid frequent memory operation.
 */
class RadixPagePool {
 public:
  /*! \brief The constructor of paged radix tree page pool, allocating memory for each page. */
  RadixPagePool(size_t page_size, size_t num_pages) : page_size_(page_size), num_pages_(num_pages) {
    CHECK_GT(page_size_ / sizeof(int32_t), RedixPage::DATA_OFFSET);
    pages_.reserve(num_pages);
    free_page_indices_.reserve(num_pages);
    raw_pool_ = new int32_t[num_pages * page_size / sizeof(int32_t)];
    int32_t num_int = page_size / sizeof(int32_t);
    for (size_t i = 0; i < num_pages; ++i) {
      pages_.push_back(reinterpret_cast<RedixPage*>(raw_pool_ + i * num_int));
      free_page_indices_.push_back(i);
    }
  }

  /*!
   * \brief Get a radix page from pool.
   * \return The allocated radix page.
   * \throw Error if no free radix page available in pool.
   */
  RedixPage* Allocate() {
    CHECK(!free_page_indices_.empty()) << "Radix page pool has no free radix tree pages.";
    int id = free_page_indices_.back();
    free_page_indices_.pop_back();
    RedixPage* page = pages_[id];
    used_pages_[page] = id;
    page->parent = page->first_child = page->next_sibiling = nullptr;
    page->capacity = page_size_ / sizeof(int32_t) - RedixPage::DATA_OFFSET;
    page->offset = page->length = 0;
    page->seq_ids = nullptr;
    return page;
  }

  /*!
   * \brief Free a radix page to pool.
   * \param page The radix page to free.
   */
  void Free(RedixPage* page) {
    CHECK_EQ(page->seq_ids, nullptr);
    CHECK(used_pages_.find(page) != used_pages_.end());
    free_page_indices_.push_back(used_pages_[page]);
    CHECK(used_pages_.erase(page));
  }

  /*!
   * \brief Get the token capacity of free pages.
   * \return The the token capacity of free pages.
   */
  size_t FreeCapacity() {
    return free_page_indices_.size() * (page_size_ / sizeof(int32_t) - RedixPage::DATA_OFFSET);
  }

  /*!
   * \brief Reset the paged radix tree page pool to initial status.
   */
  void Reset() {
    used_pages_.clear();
    free_page_indices_.reserve(num_pages_);
    for (int i = 0; i < num_pages_; ++i) {
      pages_[i]->parent = pages_[i]->first_child = pages_[i]->next_sibiling = nullptr;
      pages_[i]->capacity = page_size_ / sizeof(int32_t) - RedixPage::DATA_OFFSET;
      pages_[i]->offset = pages_[i]->length = 0;
      pages_[i]->seq_ids = nullptr;
      free_page_indices_[i] = i;
    }
  }

  /*! \brief The destructor of paged radix tree page pool, freeing memory for each page. */
  ~RadixPagePool() { delete[] raw_pool_; }

 private:
  /*! \brief The page size of each paged radix tree page. */
  size_t page_size_;
  /*! \brief The number of pages in paged radix tree page pool. */
  size_t num_pages_;
  /*! \brief The raw paged radix tree page pool. */
  int32_t* raw_pool_;
  /*! \brief The paged radix tree page pool. */
  std::vector<RedixPage*> pages_;
  /*! \brief The indices of free paged radix page in page pool. */
  std::vector<size_t> free_page_indices_;
  /*! \brief The map from used paged radix tree page to its index in page pool. */
  std::unordered_map<RedixPage*, size_t> used_pages_;
};

// PagedRadixTree

/*!
 * \brief The paged radix tree data structure.
 */
class PagedRadixTreeImpl : public PagedRadixTreeObj {
 public:
  /*! \brief The page size of each paged radix tree node. */
  size_t page_size;
  /*! \brief The number of pages in paged radix tree page pool. */
  size_t num_pages;
  /*! \brief The maximum number of sequence ID in paged radix tree page pool. */
  size_t num_seqs;
  /*! \brief The map from sequence to paged radix tree node it is stored. */
  std::unordered_map<int32_t, RedixPage*> seq2page;
  /*! \brief The sequence ID node pool. */
  SequenceIDNodePool* seq_id_node_pool = nullptr;
  /*! \brief The radix page pool. */
  RadixPagePool* radix_page_pool = nullptr;
  /*! \brief The root page of paged radix tree. */
  RedixPage* root = nullptr;

  explicit PagedRadixTreeImpl(size_t num_pages, size_t page_size, size_t num_seqs) {
    num_pages = num_pages;
    page_size = page_size;
    num_seqs = num_seqs;

    seq_id_node_pool = new SequenceIDNodePool(num_seqs);
    radix_page_pool = new RadixPagePool(page_size, num_pages);

    root = reinterpret_cast<RedixPage*>(new int32_t[RedixPage::DATA_OFFSET]);
    root->parent = root->first_child = root->next_sibiling = nullptr;
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
    for (RedixPage* page = seq2page[seq_id]; page; page = page->parent) {
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
  std::pair<size_t, std::vector<int64_t>> MatchPrefix(IntTuple tokens) {
    const int64_t* prefix = tokens.data();
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
    for (RedixPage* page = seq2page[seq_id]; page; page = page->parent) {
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
    for (RedixPage* page = seq2page[parent_seq_id]; page; page = page->parent) {
      if (forked_offset >= length - page->length) {
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
  void ExtendSequence(int64_t seq_id, IntTuple tokens) {
    CHECK(seq2page.find(seq_id) != seq2page.end());
    const int64_t* suffix = tokens.data();
    size_t length = tokens.size();
    RedixPage* original_page = seq2page[seq_id];
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
      RedixPage* new_page = radix_page_pool->Allocate();
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
    RedixPage* page = seq2page[seq_id];
    // Remove the sequence temporarily, but keeping the data and starting rolling back.
    page->PopSequence(seq_id_node_pool, seq_id);
    seq2page.erase(seq_id);
    while (page->length <= num_tokens) {
      // Roll back entire page
      num_tokens -= page->length;
      RedixPage* parent = page->parent;
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
    // Split page for rolled back seuqence
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
    RedixPage* page = seq2page[seq_id];
    page->PopSequence(seq_id_node_pool, seq_id);
    seq2page.erase(seq_id);
    while (page->parent && !page->seq_ids && !page->first_child) {
      RedixPage* parent = page->parent;
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
    root->parent = root->first_child = root->next_sibiling = nullptr;
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
  void MergePage(RedixPage* page) {
    CHECK(page->Mergeable());
    RedixPage* child = page->first_child;
    for (int i = 0; i < child->length; ++i) {
      (*page)[i + page->length] = (*child)[i];
    }
    page->length += child->length;
    page->first_child = child->first_child;
    for (RedixPage* p = child->first_child; p; p = p->next_sibiling) {
      p->parent = page;
    }
    page->seq_ids = child->seq_ids;
    std::vector<int64_t> seq_ids = page->GetLocalSequence();
    for (int64_t id : seq_ids) seq2page[id] = page;
    child->seq_ids = nullptr;
    radix_page_pool->Free(child);
  }

  /*!
   * \brief Split a radix tree page at given postition, to accept new sequence.
   * e.g. SplitPage([1, 2, 3, 4, 5], 2) = [1, 2, _, _, _] -> [3, 4, 5, _, _].
   * \param page The radix tree page to split.
   * \param offset The position to split the radix tree page.
   * \return The splitted radix tree page. It can be different from the input radix tree page, as
   * there may be implicit radix tree page merge.
   */
  RedixPage* SplitPage(RedixPage* page, size_t offset) {
    CHECK_LT(offset, page->length);
    RedixPage* child = radix_page_pool->Allocate();
    child->parent = page;
    child->first_child = page->first_child;
    for (RedixPage* p = page->first_child; p; p = p->next_sibiling) {
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
  std::tuple<RedixPage*, size_t, size_t> MatchSequence(RedixPage* page, const int64_t* tokens,
                                                       size_t length) {
    size_t offset = 0;
    while (offset < length) {
      if (RedixPage* child = page->FindChild(tokens[offset])) {
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

PagedRadixTree::PagedRadixTree(size_t num_pages, size_t page_size, size_t num_seqs) {
  data_ = std::move(make_object<PagedRadixTreeImpl>(num_pages, page_size, num_pages));
}

TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree")
    .set_body_typed([](uint64_t num_pages, uint64_t page_size, uint64_t num_seqs) {
      return PagedRadixTree(num_pages, page_size, num_seqs);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeMatchPrefix")
    .set_body_typed([](PagedRadixTree paged_radix_tree, IntTuple tokens) {
      auto [offset, seq_ids] = paged_radix_tree->MatchPrefix(tokens);
      seq_ids.insert(seq_ids.begin(), offset);
      return IntTuple(seq_ids);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeExtendSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::ExtendSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeRollBackSequence")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id, int64_t num_tokens) {
      paged_radix_tree->RollBackSequence(seq_id, num_tokens);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeForkSequence")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id, int64_t parent_seq_id,
                       uint64_t forked_offset) {
      paged_radix_tree->ForkSequence(seq_id, parent_seq_id, forked_offset);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeHasSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::HasSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeAddSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::AddSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeRemoveSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::RemoveSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeGetSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::GetSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeGetSequenceLength")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id) {
      return (int64_t)paged_radix_tree->GetSequenceLength(seq_id);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTreeFreeCapacity")
    .set_body_typed([](PagedRadixTree paged_radix_tree) {
      return (int64_t)paged_radix_tree->FreeCapacity();
    });
}  // namespace serve
}  // namespace llm
}  // namespace mlc
