/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/radix_tree.h
 */
#ifndef MLC_LLM_SERVE_RADIX_TREE_H_
#define MLC_LLM_SERVE_RADIX_TREE_H_
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/object.h>

#include <unordered_map>
#include <unordered_set>

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
 * \brief The sequence Id node pool.
 *
 * The sequence Id node pool allocates all sequence ID nodes when construction and frees when
 * destruction, to avoid frequent memory operation.
 */
class SequenceIDNodePool {
 public:
  /*! \brief The constructor of sequence Id node pool, allocating memory for each node. */
  SequenceIDNodePool(size_t num_nodes);

  /*!
   * \brief Get a radix page from pool, and assign the fields.
   * \param seq_id The assigned sequence ID of allocated sequence ID node.
   * \param node The next sequence ID node pointer of allocated sequence ID node.
   * \return The allocated radix page.
   * \throw Error if no free radix page available in pool.
   */
  SequenceIDNode* Allocate(int64_t seq_id, SequenceIDNode* next);

  /*!
   * \brief Free a sequence ID node to pool.
   * \param node The sequence ID node to free.
   */
  void Free(SequenceIDNode* node);

  /*! \brief The destructor of sequence Id node pool, freeing memory for each node. */
  ~SequenceIDNodePool();

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
  void Extend(const int64_t* suffix, size_t suffix_length);

  /*!
   * \brief Add a sequence ID in page.
   * \param pool The sequence ID node pool to allocate new node.
   * \param id The sequence ID to add.
   */
  void AddSequence(SequenceIDNodePool* pool, int64_t id);

  /*!
   * \brief Pop a sequence ID in page.
   * \param pool The sequence ID node pool to free popped node.
   * \param id The sequence ID to pop.
   * \throw Error if no such sequence ID in page.
   */
  void PopSequence(SequenceIDNodePool* pool, int64_t id);

  /*!
   * \brief Get all sequence ID in page.
   * \return The std::vector of sequence ID in page.
   */
  std::vector<int64_t> GetLocalSequence();

  /*!
   * \brief Get any sequence ID in current page or child pages.
   * Since there is always a sequence in leaf pages, it only check first child if no sequence ID in
   * current page.
   * \return The any sequence ID in current page or child pages.
   */
  int32_t FindAnyChildSequence();

  /*!
   * \brief Get all sequence ID in current page and child pages, using Iterate method with lambda
   * expression as callback to avoid frequently memory allocation of std::vector.
   * \return The std::vector of all sequence ID in current page and child pages.
   */
  std::vector<int64_t> FindAllChildSequence();

  /*!
   * \brief The iteration method for tree or sub-tree traverse.
   * \param f The callback function to invoke at each radix page visited.
   */
  template <class CallbackFunc>
  void Iterate(CallbackFunc f);

  /*!
   * \brief Get the last sibling of current page.
   * \return The page whose next_sibling is current page, or nullptr if current is the fisrt_child
   * of its parent page.
   */
  RedixPage* GetLastSibling();

  /*!
   * \brief Find the child indexed by first token.
   * \return The child page started with first token, or nullptr if no such child page.
   */
  RedixPage* FindChild(int64_t first_token);

  /*! \brief Insert a new child page. */
  void InsertChild(RedixPage* child);

  /*!
   * \brief Remove a child page.
   * \throw Error if page to be removed is not child page.
   */
  void RemoveChild(RedixPage* child);

  /*!
   * \brief Check current page is mergable with its child page.
   * The page is mergable if and only if
   * 1. No sequence ID in current page, as sequence ID is not allowed to exist within page.
   * 2. The current page has child page.
   * 3. The current page has only one child page.
   * 4. The current page perfix and the child page prefix can be concatenated into one page.
   * \return True if current page is mergable, or false.
   */
  bool Mergeable();

  /*!
   * \brief Match the given prefix within page.
   * \param prefix The prefix token array.
   * \param prefix_length The length of prefix token array.
   * \return The matched prefix offset within page, or the first mismatched token position. The
   * possible return value is [0, page->length], where page->length means the page is completely the
   * prefix of given prefix.
   */
  size_t MatchPrefix(const int64_t* prefix, size_t prefix_length);
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
  RadixPagePool(size_t page_size, size_t num_pages);

  /*!
   * \brief Get a radix page from pool.
   * \return The allocated radix page.
   * \throw Error if no free radix page available in pool.
   */
  RedixPage* Allocate();

  /*!
   * \brief Free a radix page to pool.
   * \param page The radix page to free.
   */
  void Free(RedixPage* page);

  /*!
   * \brief Get the token capacity of free pages.
   * \return The the token capacity of free pages.
   */
  size_t FreeCapacity();

  /*! \brief The destructor of paged radix tree page pool, freeing memory for each page. */
  ~RadixPagePool();

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

/*!
 * \brief The paged radix tree data structure.
 */
class PagedRadixTreeObj : public Object {
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

  /*!
   * \brief Get a sequence's all tokens.
   * \param seq_id The sequence ID for index.
   * \return The sequence tokens.
   * \throw Error if sequence ID is not valid.
   */
  IntTuple GetSequence(int64_t seq_id);

  /*!
   * \brief Get all sequences with longest common prefix with give prefix tokens.
   * \param tokens The prefix tokens for reference.
   * \return The pair of matched prefix length and the array of matched sequences indices.
   */
  std::pair<size_t, std::vector<int64_t>> MatchPrefix(IntTuple tokens);

  /*!
   * \brief Get a sequence's length.
   * \param seq_id The sequence ID for index.
   * \return The sequence length.
   * \throw Error if sequence ID is not valid.
   */
  size_t GetSequenceLength(int64_t seq_id);

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
  void ForkSequence(int64_t seq_id, int64_t parent_seq_id, size_t forked_offset);

  /*!
   * \brief Add an empty sequence at root.
   * \param seq_id The new sequence ID.
   * \throw Error if sequence ID is not valid.
   */
  void AddSequence(int64_t seq_id);

  /*!
   * \brief Extend a sequence with given tokens.
   * \param seq_id The sequence ID for index.
   * \param tokens The given tokens to extend.
   * \throw Error if sequence ID is not valid.
   */
  void ExtendSequence(int64_t seq_id, IntTuple tokens);

  /*!
   * \brief Remove a sequence.
   * \param seq_id The sequence ID to remove.
   * \throw Error if sequence ID is not valid.
   */
  void RemoveSequence(int64_t seq_id);

  /*!
   * \brief Get the remaining token capacity of the paged radix tree.
   * \return The the remaining token capacity of the paged radix tree.
   */
  size_t FreeCapacity();

  /*! \brief The destructor to free root page. */
  ~PagedRadixTreeObj();

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mlc.serve.PagedRadixTree";
  TVM_DECLARE_FINAL_OBJECT_INFO(PagedRadixTreeObj, Object)

 private:
  /*!
   * \brief Merge a radix tree page with its child radix tree page, to save radix tree page.
   * e.g. MergePage([1, 2, _, _, _] -> [3, 4, 5, _, _]) = [1, 2, 3, 4, 5].
   * And the page to be merged should be page->Mergeable().
   * \param page The parent radix tree page.
   */
  void MergePage(RedixPage* page);

  /*!
   * \brief Split a radix tree page at given postition, to accept new sequence.
   * e.g. SplitPage([1, 2, 3, 4, 5], 2) = [1, 2, _, _, _] -> [3, 4, 5, _, _].
   * \param page The radix tree page to split.
   * \param offset The position to split the radix tree page.
   * \return The splitted radix tree page. It can be different from the input radix tree page, as
   * there may be implicit radix tree page merge.
   */
  RedixPage* SplitPage(RedixPage* page, size_t offset);

  /*!
   * \brief Match with given token from a radix tree page, stopping at first mismatch.
   * \param page The radix tree page to start matching.
   * \param tokens The given tokens to match.
   * \param length The length of given tokens.
   */
  std::tuple<RedixPage*, size_t, size_t> MatchSequence(RedixPage* page, const int64_t* tokens,
                                                       size_t length);
};

TVM_REGISTER_OBJECT_TYPE(PagedRadixTreeObj);

class PagedRadixTree : public ObjectRef {
 public:
  /*!
   * \brief Constructor of paged radix tree.
   * \param num_pages The number of radix tree pages.
   * \param page_size The page size of each radix tree page.
   * \param num_seqs The maximum number of sequence ID.
   */
  PagedRadixTree(size_t num_pages, size_t page_size, size_t num_seqs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PagedRadixTree, ObjectRef, PagedRadixTreeObj);
};
}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_RADIX_TREE_H_
