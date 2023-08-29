/*!
 *  Copyright (c) 2023 by Contributors daquexian
 * \file rwkv_world_tokenizer.h
 * \brief Implementation of llm chat.
 */

#include <unordered_map>
#include <string>
#include <vector>

namespace mlc {
namespace llm {
class RWKVWorldTokenizer {
public:
  RWKVWorldTokenizer(const std::string &path);
  std::vector<int> encode(std::string_view str) const;
  std::string decode(const std::vector<int> &ids) const;
  std::string decode(int id) const;

private:
  std::unordered_map<std::string, int> _word2idx;
  std::unordered_map<int, std::string> _idx2word;
};
} // namespace llm
} // namespace mlc
