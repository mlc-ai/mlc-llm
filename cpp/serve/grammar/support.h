/*!
 * Copyright (c) 2023 by Contributors
 * \file serve/grammar/support.h
 * \brief The header for utilities used in grammar-guided generation.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_
#define MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlc {
namespace llm {
namespace serve {

/*! \brief A bitset with runtime specified length. It manages memory internally or the memory
 * provided externally with enough size. */
class DynamicBitset {
 public:
  static int CalculateBufferSize(int element_size) { return (element_size + 31) / 32; }

  DynamicBitset() : size_(0), buffer_size_(0), data_(nullptr), is_internal_(true) {}

  DynamicBitset(int size, uint32_t* data = nullptr)
      : size_(size), buffer_size_(CalculateBufferSize(size)) {
    if (data == nullptr) {
      internal_buffer_.resize(buffer_size_, 0);
      data_ = internal_buffer_.data();
      is_internal_ = true;
    } else {
      data_ = data;
      is_internal_ = false;
    }
  }

  DynamicBitset& operator=(const DynamicBitset& other) {
    DCHECK(is_internal_ || size_ >= other.size_) << "Expanding bitset size is not allowed when the "
                                                    "memory of the bitset is externally managed";
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    if (is_internal_) {
      internal_buffer_.reserve(buffer_size_);
      data_ = internal_buffer_.data();
    }
    if (data_ != other.data_) {
      std::memcpy(data_, other.data_, buffer_size_ * sizeof(uint32_t));
    }
    return *this;
  }

  DynamicBitset& operator=(DynamicBitset&& other) {
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    is_internal_ = other.is_internal_;
    if (is_internal_) {
      internal_buffer_ = std::move(other.internal_buffer_);
      data_ = internal_buffer_.data();
    } else {
      data_ = other.data_;
    }
    return *this;
  }

  bool operator[](int index) const {
    DCHECK(data_ && index >= 0 && index < size_);
    return (data_[index / 32] >> (index % 32)) & 1;
  }

  int Size() const { return size_; }

  void Set(int index, bool value) {
    DCHECK(data_ && index >= 0 && index < size_);
    if (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  void Set() {
    DCHECK(data_);
    std::memset(data_, 0xFF, buffer_size_ * sizeof(uint32_t));
  }

  void Reset() {
    DCHECK(data_);
    std::memset(data_, 0, buffer_size_ * sizeof(uint32_t));
  }

  DynamicBitset& operator|=(const DynamicBitset& other) {
    DCHECK(buffer_size_ <= other.buffer_size_);
    for (int i = 0; i < buffer_size_; ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }

 private:
  int size_;
  int buffer_size_;
  uint32_t* data_;
  std::vector<uint32_t> internal_buffer_;
  bool is_internal_;
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
