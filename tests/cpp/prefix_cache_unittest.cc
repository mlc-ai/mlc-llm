#include "serve/prefix_cache.h"

#include <gtest/gtest.h>

namespace mlc {
namespace llm {
namespace serve {

void AddRecycledSequence(const PrefixCache& cache, int64_t seq_id, int32_t token) {
  std::vector<int32_t> tokens{token, token + 1000};
  PrefixCacheMatchedResult result = cache->InsertSequence(seq_id, tokens);
  ASSERT_EQ(result.prefilled_offset, 0);
  cache->ExtendSequence(seq_id, tokens);
  cache->RecycleSequence(seq_id);
}

TEST(PrefixCache, EvictsLeastRecentlyRecycledSequence) {
  std::vector<int64_t> removed_seq_ids;
  PrefixCache cache = PrefixCache::CreateRadixPrefixCache(
      3, [&](int64_t seq_id) { removed_seq_ids.push_back(seq_id); });

  AddRecycledSequence(cache, 1, 101);
  AddRecycledSequence(cache, 2, 102);
  AddRecycledSequence(cache, 3, 103);
  AddRecycledSequence(cache, 4, 104);

  EXPECT_EQ(removed_seq_ids, std::vector<int64_t>{1});
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
