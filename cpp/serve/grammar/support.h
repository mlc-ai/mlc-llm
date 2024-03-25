/*!
 * Copyright (c) 2023 by Contributors
 * \file serve/grammar/support.h
 * \brief The header for utilities used in grammar-guided generation.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_
#define MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_

#include <tvm/runtime/logging.h>

#include <cstdint>
#include <cstring>

namespace mlc {
namespace llm {
namespace serve {

/*! \brief Manages a segment of externally provided memory and use it as a bitset. */
class BitsetManager {
 public:
  BitsetManager(uint32_t* data, int buffer_size, int element_cnt)
      : data_(data), buffer_size_(buffer_size), element_cnt_(element_cnt) {
    DCHECK(buffer_size >= CalculateBufferSize(element_cnt));
  }

  static int CalculateBufferSize(int element_cnt) { return (element_cnt + 31) / 32; }

  bool operator[](int index) const {
    DCHECK(index >= 0 && index < element_cnt_);
    return (data_[index / 32] >> (index % 32)) & 1;
  }

  void Set(int index, bool value) {
    DCHECK(index >= 0 && index < element_cnt_);
    if (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  void Reset(bool value) { std::memset(data_, value ? 0xFF : 0, buffer_size_ * sizeof(uint32_t)); }

  int GetElementCnt() const { return element_cnt_; }

 private:
  uint32_t* const data_;
  const int buffer_size_;
  const int element_cnt_;
};

/*!
 * \brief Let lhs be the union of lhs and rhs. Suppose that both sets are sorted.
 * \note No additional vectors are allocated, and the time complexity is O(n)
 */
inline void IntsetUnion(std::vector<int32_t>* lhs, const std::vector<int32_t>& rhs) {
  int original_lhs_size = lhs->size();
  int rhs_size = rhs.size();

  lhs->resize(original_lhs_size + rhs_size);

  auto it_lhs = lhs->rbegin() + rhs_size;
  auto it_rhs = rhs.rbegin();
  auto it_result = lhs->rbegin();

  while (it_lhs != lhs->rend() && it_rhs != rhs.rend()) {
    if (*it_lhs > *it_rhs) {
      *it_result = *it_lhs;
      ++it_lhs;
    } else if (*it_lhs < *it_rhs) {
      *it_result = *it_rhs;
      ++it_rhs;
    } else {
      *it_result = *it_lhs;
      ++it_lhs;
      ++it_rhs;
    }
    ++it_result;
  }

  while (it_rhs != rhs.rend()) {
    *it_result = *it_rhs;
    ++it_result;
    ++it_rhs;
  }

  auto last = std::unique(lhs->begin(), lhs->end());
  lhs->erase(last, lhs->end());
}

/*!
 * \brief Let lhs be the intersection of lhs and rhs. Suppose that both sets are sorted.
 * \note No additional vector is allocated, and the time complexity is O(n).
 * \note Support the case where lhs is the universal set by setting lhs to {-1}. The result will be
 * rhs then.
 */
inline void IntsetIntersection(std::vector<int32_t>* lhs, const std::vector<int32_t>& rhs) {
  if (lhs->size() == 1 && (*lhs)[0] == -1) {
    *lhs = rhs;
    return;
  }

  auto it_lhs = lhs->begin();
  auto it_rhs = rhs.begin();
  auto it_result = lhs->begin();

  while (it_lhs != lhs->end() && it_rhs != rhs.end()) {
    if (*it_lhs < *it_rhs) {
      ++it_lhs;
    } else if (*it_lhs > *it_rhs) {
      ++it_rhs;
    } else {
      *it_result = *it_lhs;
      ++it_lhs;
      ++it_rhs;
      ++it_result;
    }
  }
  lhs->erase(it_result, lhs->end());
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_
