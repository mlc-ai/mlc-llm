/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers.h
 * \brief Header of tokenizer related functions.
 */

#ifndef MLC_LLM_TOKENIZER_H_
#define MLC_LLM_TOKENIZER_H_

#include <tokenizers_cpp.h>

#include "base.h"

namespace mlc {
namespace llm {

using tokenizers::Tokenizer;

MLC_LLM_DLL std::unique_ptr<Tokenizer> TokenizerFromPath(const std::string& _path);

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_TOKENIZER_H_
