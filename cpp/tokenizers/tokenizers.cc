/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file tokenizer.cc
 */

#include "tokenizers.h"

#include <picojson.h>
#include <tokenizers_cpp.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/int_tuple.h>
#include <tvm/runtime/logging.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "./../support/encoding.h"
#include "./../support/load_bytes_from_file.h"

namespace mlc {
namespace llm {

TVM_FFI_STATIC_INIT_BLOCK() {
  TokenizerInfoNode::RegisterReflection();
  TokenizerObj::RegisterReflection();
}

#ifndef COMPILE_MLC_WASM_RUNTIME

String TokenizerInfoNode::AsJSONString() const {
  picojson::object obj;
  obj["token_postproc_method"] = picojson::value(token_postproc_method);
  obj["prepend_space_in_encode"] = picojson::value(prepend_space_in_encode);
  obj["strip_space_in_decode"] = picojson::value(strip_space_in_decode);
  return picojson::value(obj).serialize(false);
}

TokenizerInfo TokenizerInfo::FromJSONString(String json_string) {
  picojson::value v;
  std::string err = picojson::parse(v, json_string.operator std::string());
  TVM_FFI_ICHECK(err.empty()) << "Failed to parse JSON: " << err;

  TVM_FFI_ICHECK(v.is<picojson::object>()) << "JSON must be an object.";
  const picojson::object& obj = v.get<picojson::object>();

  ObjectPtr<TokenizerInfoNode> n = tvm::ffi::make_object<TokenizerInfoNode>();
  if (obj.count("token_postproc_method")) {
    TVM_FFI_ICHECK(obj.at("token_postproc_method").is<std::string>());
    n->token_postproc_method = obj.at("token_postproc_method").get<std::string>();
  }
  if (obj.count("prepend_space_in_encode")) {
    TVM_FFI_ICHECK(obj.at("prepend_space_in_encode").is<bool>());
    n->prepend_space_in_encode = obj.at("prepend_space_in_encode").get<bool>();
  }
  if (obj.count("strip_space_in_decode")) {
    TVM_FFI_ICHECK(obj.at("strip_space_in_decode").is<bool>());
    n->strip_space_in_decode = obj.at("strip_space_in_decode").get<bool>();
  }

  return TokenizerInfo(n);
}

Tokenizer::Tokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer, TokenizerInfo info) {
  ObjectPtr<TokenizerObj> n = tvm::ffi::make_object<TokenizerObj>();
  n->tokenizer = std::move(tokenizer);
  n->info_ = std::move(info);
  data_ = std::move(n);
}

std::vector<int32_t> TokenizerObj::Encode(const std::string& text) const {
  return tokenizer->Encode(text);
}

std::vector<int32_t> TokenizerObj::EncodeNoPrependSpace(const std::string& text) const {
  // TODO(yixin): now this only supports tokenizers with tokenizer.json
  // other tokenizers should be supported.
  static const constexpr char* kPaddingPrefix = "\x01";
  if (!info_->prepend_space_in_encode) {
    return tokenizer->Encode(text);
  }

  auto result = tokenizer->Encode(kPaddingPrefix + text);
  // remove the first two tokens: "▁" and "<0x01>"
  result.erase(result.begin(), result.begin() + 2);
  return result;
}

std::vector<std::vector<int32_t>> TokenizerObj::EncodeBatch(const Array<String>& texts) const {
  std::vector<std::string> texts_vec;
  for (const String& text : texts) {
    texts_vec.push_back(text);
  }
  return tokenizer->EncodeBatch(texts_vec);
}

std::string TokenizerObj::Decode(const std::vector<int32_t>& token_ids) const {
  return tokenizer->Decode(token_ids);
}

const DynamicBitset& TokenizerObj::GetPrefixTokenMask() {
  if (prefix_token_mask_.Size() != 0) {
    return prefix_token_mask_;
  }

  int vocab_size = GetVocabSize();
  prefix_token_mask_ = DynamicBitset(vocab_size);

  // Sort all tokens
  const auto& token_table = PostProcessedTokenTable();
  std::vector<std::pair<std::string, int>> sorted_tokens;
  for (int32_t token_id = 0; token_id < vocab_size; ++token_id) {
    sorted_tokens.emplace_back(token_table[token_id], token_id);
  }
  std::sort(sorted_tokens.begin(), sorted_tokens.end());

  // Check every token if it is a prefix of another token
  for (int idx = 0; idx < vocab_size - 1; ++idx) {
    auto cur_token = sorted_tokens[idx].first;
    auto nxt_token = sorted_tokens[idx + 1].first;
    if (cur_token.length() <= nxt_token.length() &&
        std::string_view(nxt_token).substr(0, cur_token.length()) == cur_token) {
      prefix_token_mask_.Set(sorted_tokens[idx].second);
    }
  }

  return prefix_token_mask_;
}

size_t TokenizerObj::GetVocabSize() const { return tokenizer->GetVocabSize(); }

std::string TokenizerObj::IdToToken(int32_t token_id) const {
  return tokenizer->IdToToken(token_id);
}

int32_t TokenizerObj::TokenToId(const std::string& token) const {
  return tokenizer->TokenToId(token);
}

Tokenizer Tokenizer::FromPath(const String& _path, std::optional<TokenizerInfo> info) {
  TokenizerInfo info_value = info.value_or(DetectTokenizerInfo(_path));
  std::filesystem::path path(_path.operator std::string());
  std::filesystem::path sentencepiece;
  std::filesystem::path huggingface;
  std::filesystem::path rwkvworld;
  CHECK(std::filesystem::exists(path)) << "Cannot find tokenizer via path: " << _path;
  if (std::filesystem::is_directory(path)) {
    sentencepiece = path / "tokenizer.model";
    huggingface = path / "tokenizer.json";
    rwkvworld = path / "tokenizer_model";
  } else {
    sentencepiece = path.parent_path() / "tokenizer.model";
    huggingface = path.parent_path() / "tokenizer.json";
    rwkvworld = path.parent_path() / "tokenizer_model";
  }
  if (std::filesystem::exists(huggingface)) {
    // Check HuggingFace
    return Tokenizer(tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(huggingface.string())),
                     info_value);
  }
  if (std::filesystem::exists(sentencepiece)) {
    // Check SentencePiece
    LOG(WARNING)
        << "Using `tokenizer.model` since we cannot locate `tokenizer.json`.\n"
        << "It is recommended to use `tokenizer.json` to ensure all token mappings are included, "
        << "since currently, files like `added_tokens.json`, `tokenizer_config.json` are ignored.\n"
        << "Consider converting `tokenizer.model` to `tokenizer.json` by compiling the model "
        << "with MLC again, or see if MLC's huggingface provides this file.";
    return Tokenizer(
        tokenizers::Tokenizer::FromBlobSentencePiece(LoadBytesFromFile(sentencepiece.string())),
        info_value);
  }
  {
    // Check ByteLevelBPE
    std::filesystem::path merges_path = path / "merges.txt";
    std::filesystem::path vocab_path = path / "vocab.json";
    std::filesystem::path added_tokens_path = path / "added_tokens.json";
    if (std::filesystem::exists(merges_path) && std::filesystem::exists(vocab_path) &&
        std::filesystem::exists(added_tokens_path)) {
      std::string vocab = LoadBytesFromFile(vocab_path.string());
      std::string merges = LoadBytesFromFile(merges_path.string());
      std::string added_tokens = LoadBytesFromFile(added_tokens_path.string());
      return Tokenizer(tokenizers::Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens),
                       info_value);
    }
  }
  if (std::filesystem::exists(rwkvworld)) {
    // Check RWKV
    return Tokenizer(tokenizers::Tokenizer::FromBlobRWKVWorld(rwkvworld.string()), info_value);
  }
  LOG(FATAL) << "Cannot find any tokenizer under: " << _path;
}

TokenizerInfo Tokenizer::DetectTokenizerInfo(const String& path_str) {
  std::filesystem::path path(path_str.operator std::string());
  CHECK(std::filesystem::exists(path)) << "Cannot find tokenizer via path: " << path_str;
  if (!std::filesystem::is_directory(path)) {
    path = path.parent_path();
  }
  path = path / "tokenizer.json";
  if (!std::filesystem::exists(path)) {
    LOG(WARNING) << "Tokenizer info is not detected as tokenizer.json is not found. The default "
                 << "tokenizer info will be used.";
    return TokenizerInfo(tvm::ffi::make_object<TokenizerInfoNode>());
  }

  std::string tokenizer_json = LoadBytesFromFile(path.string());
  picojson::value v;
  std::string err = picojson::parse(v, tokenizer_json);
  TVM_FFI_ICHECK(err.empty()) << "Failed to parse JSON: " << err;
  TVM_FFI_ICHECK(v.is<picojson::object>()) << "JSON must be an object.";
  const picojson::object& obj = v.get<picojson::object>();

  ObjectPtr<TokenizerInfoNode> n = tvm::ffi::make_object<TokenizerInfoNode>();

  // Step 1. Detect token_postproc_method: byte_fallback or byte_level
  // Detect {"type": "ByteLevel"} or {"type": "ByteFallback"} in "decoder" field of the tokenizer
  if (!obj.count("decoder") || !obj.at("decoder").is<picojson::object>()) {
    LOG(WARNING) << "Decoder field is not found in tokenizer.json. Use ByteFallback as default.";
    n->token_postproc_method = "byte_fallback";
  } else {
    auto decoder_obj = obj.at("decoder").get<picojson::object>();
    TVM_FFI_ICHECK(decoder_obj.count("type") && decoder_obj.at("type").is<std::string>());
    auto type = decoder_obj.at("type").get<std::string>();

    auto f_detect_decoder_type = [](ObjectPtr<TokenizerInfoNode> n,
                                    const picojson::value& decoder_json) {
      TVM_FFI_ICHECK(decoder_json.is<picojson::object>());
      TVM_FFI_ICHECK(decoder_json.get<picojson::object>().count("type") &&
                     decoder_json.get<picojson::object>().at("type").is<std::string>());
      auto type = decoder_json.get<picojson::object>().at("type").get<std::string>();
      if (type == "ByteLevel") {
        n->token_postproc_method = "byte_level";
        return true;
      } else if (type == "ByteFallback") {
        n->token_postproc_method = "byte_fallback";
        return true;
      }
      return false;
    };

    bool found = false;

    // For sequence, examine every decoder
    if (type == "Sequence") {
      TVM_FFI_ICHECK(decoder_obj.count("decoders") &&
                     decoder_obj.at("decoders").is<picojson::array>());
      for (const picojson::value& decoder : decoder_obj.at("decoders").get<picojson::array>()) {
        if (f_detect_decoder_type(n, decoder)) {
          found = true;
        }
      }
    } else {
      if (f_detect_decoder_type(n, obj.at("decoder"))) {
        found = true;
      }
    }

    if (!found) {
      LOG(WARNING) << "Neither ByteLevel nor ByteFallback decoder is detected in tokenizer.json. "
                   << "Use ByteFallback as default.";
      n->token_postproc_method = "byte_fallback";
    }
  }

  // Step 2. Detect prepend_space_in_encode
  // Find {"type": "Prepend", "prepend": "▁"} in "normalizer" field of the tokenizer
  if (obj.count("normalizer") && obj.at("normalizer").is<picojson::object>()) {
    const picojson::value& normalizer_json = obj.at("normalizer");

    auto f_handle_normalizer = [](ObjectPtr<TokenizerInfoNode> n,
                                  const picojson::value& normalizer_json) {
      TVM_FFI_ICHECK(normalizer_json.is<picojson::object>());
      auto obj = normalizer_json.get<picojson::object>();
      TVM_FFI_ICHECK(obj.count("type") && obj.at("type").is<std::string>());
      if (obj.at("type").get<std::string>() == "Prepend" && obj.count("prepend") &&
          obj.at("prepend").is<std::string>() && obj.at("prepend").get<std::string>() == "▁") {
        n->prepend_space_in_encode = true;
        return true;
      }
      return false;
    };

    auto type = normalizer_json.get<picojson::object>().at("type").get<std::string>();
    if (type == "Sequence") {
      TVM_FFI_ICHECK(
          normalizer_json.get<picojson::object>().count("normalizers") &&
          normalizer_json.get<picojson::object>().at("normalizers").is<picojson::array>());
      for (const picojson::value& normalizer :
           normalizer_json.get<picojson::object>().at("normalizers").get<picojson::array>()) {
        if (f_handle_normalizer(n, normalizer)) {
          break;
        }
      }
    } else {
      f_handle_normalizer(n, normalizer_json);
    }
  }

  // Step 3. Detect strip_space_in_decode
  // Find {"type": "Strip", "content": " ", "start": 1, "stop": 0} in "decoder" field of the
  // tokenizer
  if (obj.count("decoder") && obj.at("decoder").is<picojson::object>()) {
    const picojson::value& decoders_json = obj.at("decoder");

    auto f_handle_decoder = [](ObjectPtr<TokenizerInfoNode> n,
                               const picojson::value& decoder_json) {
      TVM_FFI_ICHECK(decoder_json.is<picojson::object>());
      auto obj = decoder_json.get<picojson::object>();
      TVM_FFI_ICHECK(obj.count("type") && obj.at("type").is<std::string>());
      if (obj.at("type").get<std::string>() == "Strip" && obj.count("content") &&
          obj.at("content").is<std::string>() && obj.at("content").get<std::string>() == " " &&
          obj.count("start") && obj.at("start").is<int64_t>() &&
          obj.at("start").get<int64_t>() == 1 && obj.count("stop") &&
          obj.at("stop").is<int64_t>() && obj.at("stop").get<int64_t>() == 0) {
        n->strip_space_in_decode = true;
        return true;
      }
      return false;
    };

    auto type = decoders_json.get<picojson::object>().at("type").get<std::string>();
    if (type == "Sequence") {
      TVM_FFI_ICHECK(decoders_json.get<picojson::object>().count("decoders") &&
                     decoders_json.get<picojson::object>().at("decoders").is<picojson::array>());
      for (const picojson::value& decoder :
           decoders_json.get<picojson::object>().at("decoders").get<picojson::array>()) {
        if (f_handle_decoder(n, decoder)) {
          break;
        }
      }
    } else {
      f_handle_decoder(n, decoders_json);
    }
  }

  return TokenizerInfo(n);
}
#endif

/*! \brief ByteFallback decoder: transform tokens like <0x1B> to hex char byte 1B */
inline std::string ByteFallbackDecoder(const std::string& token) {
  if (token.length() == 6 && token.substr(0, 3) == "<0x" && token.back() == '>') {
    int byte = 0;
    for (int i = 0; i < 2; ++i) {
      byte *= 16;
      byte +=
          token[3 + i] >= '0' && token[3 + i] <= '9' ? token[3 + i] - '0' : token[3 + i] - 'A' + 10;
    }
    TVM_FFI_ICHECK(byte >= 0 && byte < 256);
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
  static const std::array<int, 324> char_to_byte_map = {
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

  auto unicode_codepoints = ParseUTF8(token.c_str(), UTF8ErrorPolicy::kReturnInvalid);
  if (unicode_codepoints.size() == 1 && unicode_codepoints[0] == kInvalidUTF8) {
    return token;
  }

  std::string decoded;

  for (auto unicode_codepoint : unicode_codepoints) {
    TVM_FFI_ICHECK(unicode_codepoint >= 0);
    if (unicode_codepoint >= static_cast<int>(char_to_byte_map.size()) ||
        char_to_byte_map[unicode_codepoint] == -1) {
      // If there is no mapping, return the original token
      return token;
    }
    decoded += static_cast<char>(char_to_byte_map[unicode_codepoint]);
  }
  return decoded;
}

/*!
 * \brief Post-process a raw token to the actual token with the given post-processing method.
 */
inline std::string PostProcessToken(const std::string& token,
                                    const std::string& token_postproc_method) {
  if (token_postproc_method == "byte_fallback") {
    return SpaceReplacerDecoder(ByteFallbackDecoder(token));
  } else if (token_postproc_method == "byte_level") {
    return ByteLevelDecoder(token);
  } else {
    LOG(FATAL) << "Unknown post-processing method: " << token_postproc_method;
  }
}

std::vector<std::string> Tokenizer::PostProcessTokenTable(
    const std::vector<std::string>& token_table, const std::string& token_postproc_method) {
  std::vector<std::string> post_processed_token_table;
  post_processed_token_table.reserve(token_table.size());
  for (const std::string& token : token_table) {
    post_processed_token_table.push_back(PostProcessToken(token, token_postproc_method));
  }
  return post_processed_token_table;
}

#ifndef COMPILE_MLC_WASM_RUNTIME
const std::vector<std::string>& TokenizerObj::PostProcessedTokenTable() {
  if (!post_processed_token_table_.empty()) {
    return post_processed_token_table_;
  }

  std::vector<std::string> raw_token_table;
  int vocab_size = tokenizer->GetVocabSize();
  raw_token_table.reserve(vocab_size);
  for (int32_t token_id = 0; token_id < vocab_size; ++token_id) {
    raw_token_table.push_back(tokenizer->IdToToken(token_id));
  }
  post_processed_token_table_ =
      Tokenizer::PostProcessTokenTable(raw_token_table, info_->token_postproc_method);
  return post_processed_token_table_;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("mlc.tokenizers.Tokenizer", [](const String& path) { return Tokenizer::FromPath(path); })
      .def("mlc.tokenizers.TokenizerEncode",
           [](const Tokenizer& tokenizer, const String& text) {
             std::vector<int32_t> token_ids = tokenizer->Encode(text);
             return IntTuple{token_ids.begin(), token_ids.end()};
           })
      .def("mlc.tokenizers.TokenizerEncodeBatch",
           [](const Tokenizer& tokenizer, const Array<String>& texts) {
             std::vector<std::vector<int32_t>> results = tokenizer->EncodeBatch(texts);
             Array<IntTuple> ret;
             ret.reserve(results.size());
             for (const auto& result : results) {
               ret.push_back(IntTuple{result.begin(), result.end()});
             }
             return ret;
           })
      .def("mlc.tokenizers.TokenizerDecode",
           [](const Tokenizer& tokenizer, const IntTuple& token_ids) {
             return tokenizer->Decode({token_ids->data, token_ids->data + token_ids->size});
           })
      .def("mlc.tokenizers.DetectTokenizerInfo",
           [](const String& path) { return Tokenizer::DetectTokenizerInfo(path)->AsJSONString(); });
}

#endif

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("mlc.tokenizers.PostProcessTokenTable",
                  [](tvm::ffi::PackedArgs args, tvm::ffi::Any* rv) {
                    Array<String> token_table_arr = args[0].cast<Array<String>>();
                    std::string token_postproc_method = args[args.size() - 1].cast<String>();
                    std::vector<std::string> token_table;
                    for (int i = 0; i < token_table_arr.size(); ++i) {
                      token_table.push_back(token_table_arr[i]);
                    }
                    std::vector<std::string> processed_token_table =
                        Tokenizer::PostProcessTokenTable(token_table, token_postproc_method);

                    // Convert std::vector<std::string> to Array<String>
                    Array<String> processed_token_table_tvm;
                    for (int i = 0; i < processed_token_table.size(); ++i) {
                      processed_token_table_tvm.push_back(processed_token_table[i]);
                    }
                    *rv = processed_token_table_tvm;
                  })
      .def("mlc.tokenizers.PostProcessToken",
           [](const String& token, const String& token_postproc_method) {
             return PostProcessToken(token, token_postproc_method);
           });
}

}  // namespace llm
}  // namespace mlc
