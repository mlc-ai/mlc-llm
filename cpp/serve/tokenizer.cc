/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/tokenizer.cc
 * \brief The implementation of runtime module of tokenizer encode/decode functions.
 */
#define __STDC_FORMAT_MACROS

#include "tokenizer.h"

#include <tokenizers_cpp.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../tokenizers.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The tokenizer runtime module.
 * It contains the encode and decode functions of the tokenizer.
 */
class TokenizerModule : public ModuleNode {
 public:
  explicit TokenizerModule(String model_path) { tokenizer_ = TokenizerFromPath(model_path); }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "tokenize") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1);
        *rv = Tokenize(args[0]);
      });
    } else if (name == "decode") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1);
        *rv = Decode(args[0]);
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  const char* type_key() const final { return "mlc.serve.Tokenizer"; }

 private:
  /*!
   * \brief Encode the input text to token ids.
   * \param text The input to be tokenized
   * \return The tokenization result.
   */
  ShapeTuple Tokenize(std::string text) {
    CHECK(tokenizer_ != nullptr) << "Tokenizer is not initialized.";
    std::vector<int32_t> token_ids = this->tokenizer_->Encode(text);
    return ShapeTuple(token_ids.begin(), token_ids.end());
  }

  /*!
   * \brief Decode the input token ids to text.
   * \param token_ids The input token ids to decode.
   * \return The decode result.
   */
  std::string Decode(ShapeTuple token_ids) {
    CHECK(tokenizer_ != nullptr) << "Tokenizer is not initialized.";
    return this->tokenizer_->Decode(std::vector<int32_t>(token_ids.begin(), token_ids.end()));
  }

  /*! \brief The tokenizer pointer. */
  std::unique_ptr<Tokenizer> tokenizer_;
};

tvm::runtime::Module CreateTokenizerModule(String model_path) {
  ObjectPtr<TokenizerModule> n = make_object<TokenizerModule>(model_path);
  return Module(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
