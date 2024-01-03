/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_printer.h
 * \brief The header for printing the AST of a BNF grammar.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PRINTER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PRINTER_H_

#include <string>

#include "grammar.h"
#include "grammar_impl.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief Prints the abstract syntax tree of a BNF grammar to a string.
 */
class BNFGrammarPrinter {
 private:
  using TData = BNFGrammarImpl::TData;
  using TSubruleId = BNFGrammarImpl::TSubruleId;
  using DataKind = BNFGrammarImpl::DataKind;

 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to print.
   */
  explicit BNFGrammarPrinter(const BNFGrammarImpl& grammar) : grammar_(grammar) {}

  /*!
   * \brief Print the complete grammar.
   */
  std::string PrintGrammar();

  /*!
   * \brief Print a subrule with the given id. Mainly for debug use.
   */
  std::string PrintSubrule(TSubruleId subrule_id);

  /*!
   * \brief Print subrules according to the type.
   * \param begin The beginning iterator of the content of subrule.
   * \param end The end iterator of the content of subrule.
   */
  std::string PrintCharacterRange(const TData* begin, const TData* end, bool is_not_range);
  std::string PrintEmpty();
  std::string PrintRuleRef(const TData* begin);
  std::string PrintSequence(const TData* begin, const TData* end);
  std::string PrintOrRule(const TData* begin, const TData* end);

 private:
  const BNFGrammarImpl& grammar_;
  // Only print parentheses when necessary (i.e. when this subrule contains multiple elements
  // and is nested within another multi-element subrule)
  bool require_parentheses_ = false;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PRINTER_H_
