/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizer.cc
 */

#include "tokenizers.h"

#include <tokenizers_cpp.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include "./support/encoding.h"
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

/*! \brief ByteFallback decoder: transform tokens like <0x1B> to hex char byte 1B */
inline std::string ByteFallbackDecoder(const std::string& token) {
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
  return token;
}

/*! \brief SpaceReplacer decoder: transform "\u2581" back to space */
inline std::string SpaceReplacerDecoder(const std::string& token) {
  // \u2581 is the unicode for "lower one eighth block"
  // UTF8 encoding for \u2581 is 0xE2 0x96 0x81
  std::string result;
  for (size_t i = 0; i < token.size(); ++i) {
    if (i + 2 < token.size() && token[i] == char(0xE2) && token[i + 1] == char(0x96) &&
        token[i + 2] == char(0x81)) {
      result += ' ';
      i += 2;
    } else {
      result += token[i];
    }
  }
  return result;
}

/*! \brief ByteLevel decoder: inverses the bytes-to-unicode transformation in the encoding
 * process as in
 * https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59
 */
inline std::string ByteLevelDecoder(const std::string& token) {
  // clang-format off
  // The inverse map of bytes_to_unicode. -1 means there is no mapping to this unicode.
  static const std::array<int, 324> unicode_to_byte_map = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
    92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, -1,
    174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
    246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
    147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 173
  };
  // clang-format on

  auto unicode_codepoints = ParseUTF8(token.c_str());
  std::string decoded;

  for (auto unicode_codepoint : unicode_codepoints) {
    ICHECK(unicode_codepoint >= 0 &&
           unicode_codepoint < static_cast<int>(unicode_to_byte_map.size()));
    int byte = unicode_to_byte_map[unicode_codepoint];
    if (byte == -1) {
      // If there is no mapping, add the codepoint itself to the result string
      // Some tokenizer like Phi-2 have  raw tokens like \t\t
      decoded += static_cast<char>(unicode_codepoint);
    } else {
      decoded += static_cast<char>(byte);
    }
  }
  return decoded;
}

/*!
 * \brief Post-process a raw token to the actual token with the given post-processing method.
 */
inline std::string PostProcessToken(const std::string& token, const std::string& postproc_method) {
  if (postproc_method == "byte_fallback") {
    return SpaceReplacerDecoder(ByteFallbackDecoder(token));
  } else if (postproc_method == "byte_level") {
    return ByteLevelDecoder(token);
  } else {
    LOG(FATAL) << "Unknown post-processing method: " << postproc_method;
  }
}

const std::vector<std::string>& TokenizerObj::TokenTable() {
  if (!token_table_.empty()) {
    return token_table_;
  }

  int vocab_size = tokenizer->GetVocabSize();
  token_table_.reserve(vocab_size);
  for (int32_t token_id = 0; token_id < vocab_size; ++token_id) {
    token_table_.push_back(tokenizer->IdToToken(token_id));
  }
  return token_table_;
}

std::vector<std::string> Tokenizer::PostProcessTokenTable(
    const std::vector<std::string>& token_table, const std::string& postproc_method) {
  std::vector<std::string> postprocessed_token_table;
  postprocessed_token_table.reserve(token_table.size());
  for (const std::string& token : token_table) {
    postprocessed_token_table.push_back(PostProcessToken(token, postproc_method));
  }
  return postprocessed_token_table;
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
