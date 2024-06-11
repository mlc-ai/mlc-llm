/*!
 * Copyright (c) 2023 by Contributors
 * \file grammar/support.h
 * \brief The header for utilities used in grammar-guided generation.
 */
#ifndef MLC_LLM_GRAMMAR_SUPPORT_H_
#define MLC_LLM_GRAMMAR_SUPPORT_H_

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlc {
namespace llm {
namespace serve {

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

#endif  // MLC_LLM_GRAMMAR_SUPPORT_H_
