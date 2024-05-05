/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers.h
 * \brief Header of tokenizer related functions.
 */

#ifndef MLC_LLM_TOKENIZER_H_
#define MLC_LLM_TOKENIZER_H_

#include <tokenizers_cpp.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include <unordered_map>

#include "base.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;

/*! \brief A wrapper object class for tokenizer. */
class TokenizerObj : public Object {
 public:
  /*! \brief The underlying tokenizer. */
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;

  /*! \brief Encode text into ids. */
  std::vector<int32_t> Encode(const std::string& text) const;
  /*! \brief Decode token ids into text. */
  std::string Decode(const std::vector<int32_t>& token_ids) const;
  /*! \brief Return the token table of the tokenizer. Special tokens are included. */
  const std::vector<std::string>& TokenTable();

  /*!
   * \brief Returns the vocabulary size. Special tokens are considered.
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

  static constexpr const char* _type_key = "mlc.Tokenizer";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(TokenizerObj, Object);

 private:
  /*! \brief The cached token table. */
  std::vector<std::string> token_table_;
};

class Tokenizer : public ObjectRef {
 public:
  /*! \brief Create a tokenizer from a directory path on disk. */
  MLC_LLM_DLL static Tokenizer FromPath(const String& path);

  /*!
   * \brief Convert raw tokens provided by the tokenizer to their original string to simplify
   * later processing. E.g. For LLaMA-2, convert "▁of" to " of".
   *
   * \param token_table The raw token table.
   * \param postproc_method The postprocessing method to use. Now we only support "byte_fallback"
   * and "byte_level", which refers to the type of the decoder of the tokenizer.
   *   - "byte_fallback": Use the decoding method in the byte-fallback BPE tokenizer. This is used
   *     by LLaMA-2, Mixtral-7b, etc. This method: 1) transform tokens like <0x1B> to hex char
   *     byte 1B. (known as the byte-fallback method); 2) transform \\u2581 to space.
   *   - "byte_level": Use the decoding method in the byte-level BPE tokenizer. This is used by
   *     LLaMA-3, GPT-2, Phi-2, etc. This method inverses the bytes-to-unicode transformation in
   *     the encoding process as in
   * https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59
   * \returns The postprocessed token table containing the original strings.
   */
  static std::vector<std::string> PostProcessTokenTable(const std::vector<std::string>& token_table,
                                                        const std::string& postproc_method);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Tokenizer, ObjectRef, TokenizerObj);

 private:
  explicit Tokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer);
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_TOKENIZER_H_
