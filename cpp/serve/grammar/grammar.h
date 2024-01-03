/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar and
 * provides utilities to parse and print the AST. User should provide a BNF/EBNF (Extended
 * Backus-Naur Form) grammar, and use BNFGrammar::FromEBNFString to parse and simplify the grammar
 * into an AST of BNF grammar.
 *
 * \sa For the design and implementation details of the AST, see ./grammar_impl.h.
 */
class BNFGrammarNode : public Object {
 public:
  /*!
   * \brief Print the BNF grammar to a string, in standard BNF format.
   */
  virtual String AsString() const = 0;
  /*!
   * \brief Serialize the AST. Dump the raw representation of the AST to a JSON file.
   * \param prettify Whether to format the JSON string. If false, all whitespaces will be removed.
   * \sa For the format of the JSON file, see ./grammar_impl.h.
   */
  virtual String AsJSON(bool prettify = true) const = 0;

  static constexpr const char* _type_key = "mlc.serve.BNFGrammar";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(BNFGrammarNode, Object);
};

class BNFGrammar : public ObjectRef {
 public:
  /*!
   * \brief Parse a BNF grammar from a string in BNF/EBNF format.
   * \details This function accepts the EBNF notation from the W3C XML Specification, which is a
   * popular standard, with the following changes:
   * - Using # as comment mark instead of /**\/
   * - Using C-style unicode escape sequence \u01AB, \U000001AB, \xAB instead of #x0123
   * - Do not support A-B (match A and not match B) yet
   *
   * See tests/python/serve/json.ebnf for an example.
   *
   * \param ebnf_string The grammar string.
   * \return The parsed BNF grammar.
   */
  TVM_DLL static BNFGrammar FromEBNFString(String ebnf_string);
  /*!
   * \brief Load a BNF grammar from the raw representation of the AST in JSON format.
   * \param json_string The JSON string.
   * \return The loaded BNF grammar.
   */
  TVM_DLL static BNFGrammar FromJSON(String json_string);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BNFGrammar, ObjectRef, BNFGrammarNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_
