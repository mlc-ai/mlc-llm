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
  /*! \brief Return the token table of the tokenizer. */
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

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Tokenizer, ObjectRef, TokenizerObj);

 private:
  explicit Tokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer);
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_TOKENIZER_H_
