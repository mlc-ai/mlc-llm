/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizer.cc
 */

#include "tokenizers.h"

#include <tokenizers_cpp.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "./support/load_bytes_from_file.h"

namespace mlc {
namespace llm {

TVM_REGISTER_OBJECT_TYPE(TokenizerObj);

Tokenizer::Tokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer) {
  ObjectPtr<TokenizerObj> n = make_object<TokenizerObj>();
  n->tokenizer = std::move(tokenizer);
  data_ = std::move(n);
}

std::vector<int32_t> TokenizerObj::Encode(const std::string& text) const {
  return tokenizer->Encode(text);
}

std::string TokenizerObj::Decode(const std::vector<int32_t>& token_ids) const {
  return tokenizer->Decode(token_ids);
}

size_t TokenizerObj::GetVocabSize() const { return tokenizer->GetVocabSize(); }

std::string TokenizerObj::IdToToken(int32_t token_id) const {
  return tokenizer->IdToToken(token_id);
}

int32_t TokenizerObj::TokenToId(const std::string& token) const {
  return tokenizer->TokenToId(token);
}

Tokenizer Tokenizer::FromPath(const String& _path) {
  std::filesystem::path path(_path.operator std::string());
  std::filesystem::path sentencepiece;
  std::filesystem::path huggingface;
  std::filesystem::path rwkvworld;
  CHECK(std::filesystem::exists(path)) << "Cannot find tokenizer via path: " << _path;
  if (std::filesystem::is_directory(path)) {
    sentencepiece = path / "tokenizer.model";
    huggingface = path / "tokenizer.json";
    rwkvworld = path / "tokenizer_model";
    // Check ByteLevelBPE
    {
      std::filesystem::path merges_path = path / "merges.txt";
      std::filesystem::path vocab_path = path / "vocab.json";
      std::filesystem::path added_tokens_path = path / "added_tokens.json";
      if (std::filesystem::exists(merges_path) && std::filesystem::exists(vocab_path) &&
          std::filesystem::exists(added_tokens_path)) {
        std::string vocab = LoadBytesFromFile(vocab_path.string());
        std::string merges = LoadBytesFromFile(merges_path.string());
        std::string added_tokens = LoadBytesFromFile(added_tokens_path.string());
        return Tokenizer(tokenizers::Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens));
      }
    }
  } else {
    sentencepiece = path.parent_path() / "tokenizer.model";
    huggingface = path.parent_path() / "tokenizer.json";
    rwkvworld = path.parent_path() / "tokenizer_model";
  }
  if (std::filesystem::exists(huggingface)) {
    return Tokenizer(tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(huggingface.string())));
  }
  if (std::filesystem::exists(sentencepiece)) {
    LOG(WARNING)
        << "Using `tokenizer.model` since we cannot locate `tokenizer.json`.\n"
        << "It is recommended to use `tokenizer.json` to ensure all token mappings are included, "
        << "since currently, files like `added_tokens.json`, `tokenizer_config.json` are ignored.\n"
        << "Consider converting `tokenizer.model` to `tokenizer.json` by compiling the model "
        << "with MLC again, or see if MLC's huggingface provides this file.";
    return Tokenizer(
        tokenizers::Tokenizer::FromBlobSentencePiece(LoadBytesFromFile(sentencepiece.string())));
  }
  if (std::filesystem::exists(rwkvworld)) {
    return Tokenizer(tokenizers::Tokenizer::FromBlobRWKVWorld(rwkvworld.string()));
  }
  LOG(FATAL) << "Cannot find any tokenizer under: " << _path;
}

/*!
 * \brief Post-process a raw token (which may be a raw byte or contain lower
 * one eights block) to the actual token.
 * We do this in order to conform with the tokenizers' setup.
 */
inline std::string PostProcessToken(std::string token) {
  // 1. The token represents a byte.
  if (token.length() == 6 && token.substr(0, 3) == "<0x" && token.back() == '>') {
    int byte = 0;
    for (int i = 0; i < 2; ++i) {
      byte *= 16;
      byte +=
          token[3 + i] >= '0' && token[3 + i] <= '9' ? token[3 + i] - '0' : token[3 + i] - 'A' + 10;
    }
    ICHECK(byte >= 0 && byte < 256);
    return std::string(/*n=*/1, static_cast<char>(byte));
  }

  // 2. The token contains "\u2581" which means space.
  static const std::string& lower_one_eighth_block = "\u2581";
  size_t pos = token.find(lower_one_eighth_block);
  while (pos != std::string::npos) {
    token.replace(pos, /*n=*/lower_one_eighth_block.length(), /*str=*/" ");
    pos = token.find(lower_one_eighth_block);
  }
  return token;
}

const std::vector<std::string>& TokenizerObj::TokenTable() {
  if (!token_table_.empty()) {
    return token_table_;
  }

  int vocab_size = tokenizer->GetVocabSize();
  token_table_.reserve(vocab_size);
  for (int32_t token_id = 0; token_id < vocab_size; ++token_id) {
    std::string token = tokenizer->IdToToken(token_id);
    token_table_.push_back(PostProcessToken(token));
  }
  return token_table_;
}

TVM_REGISTER_GLOBAL("mlc.Tokenizer").set_body_typed([](const String& path) {
  return Tokenizer::FromPath(path);
});

TVM_REGISTER_GLOBAL("mlc.TokenizerEncode")
    .set_body_typed([](const Tokenizer& tokenizer, const String& text) {
      std::vector<int32_t> token_ids = tokenizer->Encode(text);
      return IntTuple{token_ids.begin(), token_ids.end()};
    });

TVM_REGISTER_GLOBAL("mlc.TokenizerDecode")
    .set_body_typed([](const Tokenizer& tokenizer, const IntTuple& token_ids) {
      return tokenizer->Decode({token_ids->data, token_ids->data + token_ids->size});
    });

}  // namespace llm
}  // namespace mlc
