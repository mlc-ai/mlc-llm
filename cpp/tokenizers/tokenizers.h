/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file tokenizers.h
 * \brief Header of tokenizer related functions.
 */

#ifndef MLC_LLM_TOKENIZER_H_
#define MLC_LLM_TOKENIZER_H_

#include <tokenizers_cpp.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/object.h>

#include <optional>
#include <unordered_map>

#include "../base.h"
#include "../support/dynamic_bitset.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;
using tvm::ffi::Array;
using tvm::ffi::String;

/*! \brief Useful information of the tokenizer during generation. */
class TokenizerInfoNode : public Object {
 public:
  /*! \brief The method to post-process the tokens to their original strings.
   * Possible values (each refers to a kind of tokenizer):
   * - "byte_fallback": The same as the byte-fallback BPE tokenizer, including LLaMA-2,
   *   Mixtral-7b, etc. E.g. "▁of" -> " of", "<0x1B>" -> "\x1B".
   *   This method:
   *   1) Transform tokens like <0x1B> to hex char byte 1B. (so-called byte-fallback)
   *   2) Replace \\u2581 "▁" with space.
   * - "byte_level": The same as the byte-level BPE tokenizer, including LLaMA-3, GPT-2,
   *   Phi-2, etc. E.g. "Ġin" -> " in", "ě" -> "\x1B"
   *   This method inverses the bytes-to-unicode transformation in the encoding process in
   *   https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59
   */
  String token_postproc_method = "byte_fallback";
  /*! \brief Whether to prepend a space during encoding. */
  bool prepend_space_in_encode = false;
  /*! \brief Whether to strip the first space during decoding. */
  bool strip_space_in_decode = false;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.TokenizerInfo";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(TokenizerInfoNode, Object);
};

class TokenizerInfo : public ObjectRef {
 public:
  /*! \brief Create a TokenizerInfo object from a dumped string. */
  static TokenizerInfo FromJSONString(String json_string);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TokenizerInfo, ObjectRef, TokenizerInfoNode);
};

/*! \brief A wrapper object class for tokenizer. */
class TokenizerObj : public Object {
 public:
  /*! \brief The underlying tokenizer. */
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;

  /*! \brief Encode text into ids. */
  std::vector<int32_t> Encode(const std::string& text) const;

  /*! \brief Encode text into ids. Some tokenizers may prepend a space in encoding, this method
   * guarantees the space is not prepended. */
  std::vector<int32_t> EncodeNoPrependSpace(const std::string& text) const;

  /*! \brief Encode texts into ids. */
  std::vector<std::vector<int32_t>> EncodeBatch(const Array<String>& texts) const;

  /*! \brief Decode token ids into text. */
  std::string Decode(const std::vector<int32_t>& token_ids) const;

  /*! \brief Return the post-processed token table of the tokenizer. Special tokens are included. */
  const std::vector<std::string>& PostProcessedTokenTable();

  /*! \brief Get the prefix token mask as a bitset. The tokens which is a prefix of another token
   * are set to true, and others are set to false in the bitset. */
  const DynamicBitset& GetPrefixTokenMask();

  /*!
   * \brief Returns the vocabulary size. Special tokens are considered. This may be smaller than the
   * `vocab_size` in config.json (length of logits), see https://github.com/QwenLM/Qwen2/issues/147
   * and https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/47.
   */
  size_t GetVocabSize() const;

  /*!
   * \brief Convert the given id to its corresponding token if it exists. If not, return an
   * empty string.
   */
  std::string IdToToken(int32_t token_id) const;

  /*!
   * \brief Convert the given token to its corresponding id if it exists. If not, return -1.
   */
  int32_t TokenToId(const std::string& token) const;

  friend class Tokenizer;
  static constexpr const char* _type_key = "mlc.Tokenizer";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(TokenizerObj, Object);

 private:
  /*! \brief Useful information of the tokenizer during generation. */
  TokenizerInfo info_;
  /*! \brief The cached token table. */
  std::vector<std::string> post_processed_token_table_;
  /*! \brief The cached prefix token mask. */
  DynamicBitset prefix_token_mask_;
};

class Tokenizer : public ObjectRef {
 public:
  /*!
   * \brief Create a tokenizer from a directory path on disk.
   * \param path The path to the tokenizer or the tokenizer directory.
   * \param info The tokenizer info. If not provided, the info will be detected automatically.
   */
  MLC_LLM_DLL static Tokenizer FromPath(const String& path,
                                        std::optional<TokenizerInfo> info = std::nullopt);

  /*! \brief Detect the tokenizer info from the given path of the tokenizer. */
  MLC_LLM_DLL static TokenizerInfo DetectTokenizerInfo(const String& path);

  /*!
   * \brief Post-process the token table to their original strings.
   * \param token_table The raw token table.
   * \param postproc_method The postprocessing method to use.
   * \returns The postprocessed token table containing the original strings.
   */
  static std::vector<std::string> PostProcessTokenTable(const std::vector<std::string>& token_table,
                                                        const std::string& token_postproc_method);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Tokenizer, ObjectRef, TokenizerObj);

 private:
  explicit Tokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer, TokenizerInfo info);
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_TOKENIZER_H_
