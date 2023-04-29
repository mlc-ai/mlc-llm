/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers.cc
 * \brief C binding to tokenizers
 */
#ifndef TOKENIZERS_H_
#define TOKENIZERS_H_


// The C API
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* TokenizerHandle;

TokenizerHandle tokenizers_new_from_str(const char* json, size_t len);

void tokenizers_encode(TokenizerHandle handle, const char* data, size_t len, int add_special_token);

void tokenizers_decode(TokenizerHandle handle, const uint32_t* data, size_t len, int skip_special_token);

void tokenizers_get_decode_str(TokenizerHandle handle, const char** data, size_t* len);

void tokenizers_get_encode_ids(TokenizerHandle handle, const uint32_t** id_data, size_t* len);

void tokenizers_free(TokenizerHandle handle);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <string>
#include <vector>

// simple c++ binding
namespace tokenizers {

/*!
 * \brief A simple c++ header of tokenizer via C API.
 */
class Tokenizer {
 public:
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&& other) {
    std::swap(other.handle_, handle_);
  }

  ~Tokenizer() {
    if (handle_ != nullptr) {
      tokenizers_free(handle_);
    }
  }

  // use i32 to be consistent with sentencepiece
  std::vector<int32_t> Encode(const std::string& text, bool add_special_token) {
    tokenizers_encode(handle_, text.data(), text.length(), static_cast<int>(add_special_token));
    const uint32_t* data;
    size_t len;
    tokenizers_get_encode_ids(handle_, &data, &len);
    return std::vector<int32_t>(data, data + len);
  }

  // use i32 to be consistent with sentencepiece
  std::string Decode(const std::vector<int32_t>& ids, bool skip_special_token) {
    tokenizers_decode(
      handle_,
      reinterpret_cast<const uint32_t*>(ids.data()), ids.size(),
      static_cast<int>(skip_special_token));
    const char* data;
    size_t len;
    tokenizers_get_decode_str(handle_, &data, &len);
    return std::string(data, len);
  }

  // avoid from_file so we don't have file system
  static Tokenizer FromJSON(const std::string& json) {
    return Tokenizer(tokenizers_new_from_str(json.data(), json.length()));
  }

 private:
  Tokenizer(TokenizerHandle handle) : handle_(handle) {}
  // internal handle
  TokenizerHandle handle_{nullptr};
};
}  // namespace tokenizers
#endif
#endif  // TOKENIZERS_H_