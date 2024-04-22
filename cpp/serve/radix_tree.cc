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

// SequenceIDNodePool
SequenceIDNodePool::SequenceIDNodePool(size_t num_nodes) : num_nodes_(num_nodes) {
  nodes_.reserve(num_nodes);
  free_node_indicess_.reserve(num_nodes);
  used_nodes_.clear();
  raw_pool_ = new SequenceIDNode[num_nodes_];
  for (size_t i = 0; i < num_nodes; ++i) {
    nodes_.push_back(&raw_pool_[i]);
    free_node_indicess_.push_back(i);
  }
}

SequenceIDNode* SequenceIDNodePool::Allocate(int64_t seq_id, SequenceIDNode* next) {
  CHECK(!free_node_indicess_.empty()) << "Sequence ID node pool has no free sequence ID nodes.";
  size_t id = free_node_indicess_.back();
  free_node_indicess_.pop_back();
  SequenceIDNode* node = nodes_[id];
  used_nodes_[node] = id;
  node->id = seq_id;
  node->next = next;
  return node;
}

void SequenceIDNodePool::Free(SequenceIDNode* node) {
  CHECK(used_nodes_.find(node) != used_nodes_.end());
  free_node_indicess_.push_back(used_nodes_[node]);
  used_nodes_.erase(node);
}

SequenceIDNodePool::~SequenceIDNodePool() { delete[] raw_pool_; }

// RedixPage
void RedixPage::Extend(const int64_t* suffix, size_t suffix_length) {
  CHECK_LE(suffix_length + length, capacity);
  for (int i = 0; i < suffix_length; ++i) {
    (*this)[i + length] = (int32_t)suffix[i];
  }
  length += suffix_length;
}

void RedixPage::AddSequence(SequenceIDNodePool* pool, int64_t id) {
  seq_ids = pool->Allocate(id, seq_ids);
}

void RedixPage::PopSequence(SequenceIDNodePool* pool, int64_t id) {
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
    }
    LOG_FATAL << "Sequence ID = " << id << " not found.";
  }
}

std::vector<int64_t> RedixPage::GetLocalSequence() {
  std::vector<int64_t> output;
  for (SequenceIDNode* node = seq_ids; node; node = node->next) {
    output.push_back(node->id);
  }
  return output;
}

int32_t RedixPage::FindAnyChildSequence() {
  if (seq_ids) return seq_ids->id;
  return first_child->FindAnyChildSequence();
}

std::vector<int64_t> RedixPage::FindAllChildSequence() {
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

template <class CallbackFunc>
void RedixPage::Iterate(CallbackFunc f) {
  f(this);
  if (next_sibiling) next_sibiling->Iterate(f);
  if (first_child) first_child->Iterate(f);
}

RedixPage* RedixPage::GetLastSibling() {
  if (parent == nullptr) return nullptr;
  if (parent->first_child == this) return nullptr;
  for (RedixPage* child = parent->first_child; child; child = child->next_sibiling) {
    if (child->next_sibiling == this) return child;
  }
  return nullptr;
}

RedixPage* RedixPage::FindChild(int64_t first_token) {
  int32_t casted = first_token;
  // Iterate all child radix pages, as the child radix pages are stored unorderly.
  for (RedixPage* child = first_child; child; child = child->next_sibiling) {
    if ((*child)[0] == casted) return child;
  }
  return nullptr;
}

void RedixPage::InsertChild(RedixPage* child) {
  child->parent = this;
  child->next_sibiling = first_child;
  first_child = child;
}

void RedixPage::RemoveChild(RedixPage* child) {
  CHECK(child->parent == this);
  if (first_child == child) {
    first_child = child->next_sibiling;
  } else {
    child->GetLastSibling()->next_sibiling = child->next_sibiling;
  }
}

bool RedixPage::Mergeable() {
  if (seq_ids) return false;
  if (!first_child) return false;
  if (first_child->next_sibiling) return false;
  if (length + first_child->length > capacity) return false;
  return true;
}

size_t RedixPage::MatchPrefix(const int64_t* prefix, size_t prefix_length) {
  size_t n = std::min(length, prefix_length);
  for (int i = 0; i < n; ++i) {
    if ((*this)[i] != prefix[i]) return i;
  }
  return n;
}

// RadixPagePool

RadixPagePool::RadixPagePool(size_t page_size, size_t num_pages)
    : page_size_(page_size), num_pages_(num_pages) {
  pages_.reserve(num_pages);
  free_page_indices_.reserve(num_pages);
  raw_pool_ = new int32_t[num_pages * page_size / sizeof(int32_t)];
  int32_t num_int = page_size / sizeof(int32_t);
  for (size_t i = 0; i < num_pages; ++i) {
    pages_.push_back(reinterpret_cast<RedixPage*>(raw_pool_ + i * num_int));
    free_page_indices_.push_back(i);
  }
}

RedixPage* RadixPagePool::Allocate() {
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

void RadixPagePool::Free(RedixPage* page) {
  CHECK_EQ(page->seq_ids, nullptr);
  CHECK(used_pages_.find(page) != used_pages_.end());
  free_page_indices_.push_back(used_pages_[page]);
  CHECK(used_pages_.erase(page));
}

size_t RadixPagePool::FreeCapacity() {
  return free_page_indices_.size() * (page_size_ / sizeof(int32_t) - RedixPage::DATA_OFFSET);
}

RadixPagePool::~RadixPagePool() { delete[] raw_pool_; }

// PagedRadixTreeObj

IntTuple PagedRadixTreeObj::GetSequence(int64_t seq_id) {
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

std::pair<size_t, std::vector<int64_t>> PagedRadixTreeObj::MatchPrefix(IntTuple tokens) {
  const int64_t* prefix = tokens.data();
  size_t length = tokens.size();
  auto [page, offset, in_page_offset] = MatchSequence(root, prefix, length);
  if (!offset) return std::make_pair(0, std::vector<int64_t>());
  return std::make_pair(offset, page->FindAllChildSequence());
}

size_t PagedRadixTreeObj::GetSequenceLength(int64_t seq_id) {
  CHECK(seq2page.find(seq_id) != seq2page.end());
  size_t length = 0;
  for (RedixPage* page = seq2page[seq_id]; page; page = page->parent) {
    length += page->length;
  }
  return length;
}

void PagedRadixTreeObj::ForkSequence(int64_t seq_id, int64_t parent_seq_id, size_t forked_offset) {
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

void PagedRadixTreeObj::AddSequence(int64_t seq_id) {
  CHECK(seq2page.find(seq_id) == seq2page.end());
  root->AddSequence(seq_id_node_pool, seq_id);
  seq2page[seq_id] = root;
}

void PagedRadixTreeObj::ExtendSequence(int64_t seq_id, IntTuple tokens) {
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

void PagedRadixTreeObj::RemoveSequence(int64_t seq_id) {
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

size_t PagedRadixTreeObj::FreeCapacity() { return radix_page_pool->FreeCapacity(); }

PagedRadixTreeObj::~PagedRadixTreeObj() {
  delete[] reinterpret_cast<int32_t*>(root);
  delete seq_id_node_pool;
  delete radix_page_pool;
}

void PagedRadixTreeObj::MergePage(RedixPage* page) {
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

RedixPage* PagedRadixTreeObj::SplitPage(RedixPage* page, size_t offset) {
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

std::tuple<RedixPage*, size_t, size_t> PagedRadixTreeObj::MatchSequence(RedixPage* page,
                                                                        const int64_t* tokens,
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

// PagedRadixTree

PagedRadixTree::PagedRadixTree(size_t num_pages, size_t page_size, size_t num_seqs) {
  ObjectPtr<PagedRadixTreeObj> n = make_object<PagedRadixTreeObj>();
  n->num_pages = num_pages;
  n->page_size = page_size;
  n->num_seqs = num_seqs;

  n->seq_id_node_pool = new SequenceIDNodePool(num_seqs);
  n->radix_page_pool = new RadixPagePool(page_size, num_pages);

  n->root = reinterpret_cast<RedixPage*>(new int32_t[RedixPage::DATA_OFFSET]);
  n->root->parent = n->root->first_child = n->root->next_sibiling = nullptr;
  n->root->offset = n->root->length = n->root->capacity = 0;
  n->root->seq_ids = nullptr;

  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree")
    .set_body_typed([](uint64_t num_pages, uint64_t page_size, uint64_t num_seqs) {
      return PagedRadixTree(num_pages, page_size, num_seqs);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_MatchPrefix")
    .set_body_typed([](PagedRadixTree paged_radix_tree, IntTuple tokens) {
      auto [offset, seq_ids] = paged_radix_tree->MatchPrefix(tokens);
      seq_ids.insert(seq_ids.begin(), offset);
      return IntTuple(seq_ids);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_ExtendSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::ExtendSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_ForkSequence")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id, int64_t parent_seq_id,
                       uint64_t forked_offset) {
      paged_radix_tree->ForkSequence(seq_id, parent_seq_id, forked_offset);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_AddSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::AddSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_RemoveSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::RemoveSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_GetSequence")
    .set_body_method<PagedRadixTree>(&PagedRadixTreeObj::GetSequence);
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_GetSequenceLength")
    .set_body_typed([](PagedRadixTree paged_radix_tree, int64_t seq_id) {
      return (int64_t)paged_radix_tree->GetSequenceLength(seq_id);
    });
TVM_REGISTER_GLOBAL("mlc.serve.PagedRadixTree_FreeCapacity")
    .set_body_typed([](PagedRadixTree paged_radix_tree) {
      return (int64_t)paged_radix_tree->FreeCapacity();
    });
}  // namespace serve
}  // namespace llm
}  // namespace mlc
