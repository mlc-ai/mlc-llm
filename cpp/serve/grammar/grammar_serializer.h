/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_printer.h
 * \brief The header for printing the AST of a BNF grammar.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PRINTER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PRINTER_H_

#include <string>

#include "grammar.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief Serialize the abstract syntax tree of a BNF grammar to a string.
 */
class BNFGrammarSerializer {
 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to print.
   */
  explicit BNFGrammarSerializer(const BNFGrammar& grammar) : grammar_(grammar) {}

  /*!
   * \brief Serialize the grammar to string.
   */
  virtual String ToString() = 0;

 protected:
  const BNFGrammar& grammar_;
};

/*!
 * \brief Prints the BNF AST with standard BNF format.
 */
class BNFGrammarPrinter : public BNFGrammarSerializer {
 private:
  using TSubruleData = BNFGrammarNode::TSubruleData;
  using TSubruleId = BNFGrammarNode::TSubruleId;
  using DataKind = BNFGrammarNode::DataKind;
  using Subrule = BNFGrammarNode::Subrule;

 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to print.
   */
  explicit BNFGrammarPrinter(const BNFGrammar& grammar) : BNFGrammarSerializer(grammar) {}

  /*!
   * \brief Print the complete grammar.
   */
  String ToString() final;

  /*!
   * \brief Print a subrule with the given id. Mainly for debug use.
   */
  std::string PrintSubrule(TSubruleId subrule_id);

  /*!
   * \brief Print subrules according to the type.
   * \param begin The beginning iterator of the content of subrule.
   * \param end The end iterator of the content of subrule.
   */
  std::string PrintCharacterRange(const Subrule& subrule);
  std::string PrintEmpty(const Subrule& subrule);
  std::string PrintRuleRef(const Subrule& subrule);
  std::string PrintSequence(const Subrule& subrule);
  std::string PrintOrRule(const Subrule& subrule);

 private:
  // Only print parentheses when necessary (i.e. when this subrule contains multiple elements
  // and is nested within another multi-element subrule)
  bool require_parentheses_ = false;
};

/*!
 * \brief Serialize the the raw representation of the BNF AST to a string with JSON format.
 * \sa BNFJSONParser::Parse for parsing the JSON string.
 * \details JSON format:
 *  {
 *    "rules": [
 *      {"name": "...", "subrule": subrule_id},
 *      {"name": "...", "subrule": subrule_id},
 *    ],
 *    "subrule_data": [integers...],
 *    "subrule_indptr": [integers...],
 *  }
 */
class BNFGrammarJSONSerializer : public BNFGrammarSerializer {
 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to print.
   */
  explicit BNFGrammarJSONSerializer(const BNFGrammar& grammar, bool prettify = true)
      : BNFGrammarSerializer(grammar), prettify_(prettify) {}

  /*!
   * \brief Dump the raw representation of the AST to a JSON file.
   * \param prettify Whether to format the JSON string. If false, all whitespaces will be removed.
   */
  String ToString() final;

 private:
  bool prettify_;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PRINTER_H_
