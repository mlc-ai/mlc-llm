/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_parser.h
 * \brief The header for the parser of BNF/EBNF grammar into BNF AST.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PARSER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PARSER_H_

#include "grammar_impl.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief This class parses a BNF/EBNF grammar string into an abstract syntax tree (AST).
 */
class EBNFParser {
 public:
  /*!
   * \brief The constructor.
   * \param bnf_grammar The grammar to parse into.
   */
  explicit EBNFParser(BNFGrammarImpl* bnf_grammar) : bnf_grammar_(bnf_grammar) {}

  /*!
   * \brief Parse the grammar string into the provided bnf_grammar. If fails, throw ParseError with
   * the error message.
   * \param ebnf_string The grammar string.
   */
  void Parse(String ebnf_string);

  /*!
   * \brief The exception thrown when parsing fails.
   */
  class ParseError : public Error {
   public:
    ParseError(const std::string& msg) : Error(msg) {}
  };

 private:
  // Parsing different parts of the grammar
  std::string ParseName(bool accept_empty = false);
  BNFGrammarImpl::TSubruleId ParseCharacterRange();
  BNFGrammarImpl::TSubruleId ParseString();
  BNFGrammarImpl::TSubruleId ParseRuleRef();
  BNFGrammarImpl::TSubruleId ParseElement();
  BNFGrammarImpl::TSubruleId ParseQuantifier();
  BNFGrammarImpl::TSubruleId ParseSequence();
  BNFGrammarImpl::TSubruleId ParseOrRule();
  BNFGrammarImpl::Rule ParseRule();

  // Helper functions
  // Helper for ParseQuantifier
  BNFGrammarImpl::TRuleId HandleStarQuantifier(BNFGrammarImpl::TSubruleId subrule_id);
  BNFGrammarImpl::TRuleId HandlePlusQuantifier(BNFGrammarImpl::TSubruleId subrule_id);
  BNFGrammarImpl::TRuleId HandleQuestionQuantifier(BNFGrammarImpl::TSubruleId subrule_id);

  // When parsing, we first find the names of all rules, and build the mapping from name to rule id.
  void BuildRuleNameToId();
  // Consumes several spaces (newline, space, tab, comment, etc.)
  void ConsumeSpace(bool allow_newline = true);
  // Check the validity of a name
  static bool IsNameChar(TCodepoint c, bool first_char = false);
  // Create a new rule and return the rule id.
  BNFGrammarImpl::TRuleId NewRule(std::string name_hint);
  // Reset the parser to the beginning of the string.
  void Reset(const char* cur);

  // Consume a specified number of characters, and maintain the line and column number.
  void Consume(int cnt = 1) {
    for (int i = 0; i < cnt; ++i) {
      // \n \r \r\n
      if (Peek() == '\n' || (Peek() == '\r' && Peek(1) != '\n')) {
        ++cur_line_;
        cur_column_ = 1;
      } else {
        ++cur_column_;
      }
      ++cur_;
    }
  }

  // Peek the next character.
  char Peek(int delta = 0) const { return *(cur_ + delta); }

  // Throw a ParseError with the given message and the line and column number.
  [[noreturn]] void ThrowParseError(const std::string& msg) {
    throw ParseError(msg + " at line " + std::to_string(cur_line_) + ", column " +
                     std::to_string(cur_column_));
  }

  // The grammar to parse into.
  BNFGrammarImpl* bnf_grammar_;
  // A pointer to the current parse position in the string
  const char* cur_ = nullptr;
  // The current line and column number
  int cur_line_ = 1;
  int cur_column_ = 1;
  // The current rule name. Help to generate a name for a new rule.
  std::string cur_rule_name_;
  // Whether the current element is in parentheses.
  // A sequence expression cannot contain newline, unless it is in parentheses.
  bool in_parentheses_ = false;
  // The mapping from rule name to rule id.
  std::unordered_map<std::string, BNFGrammarImpl::TRuleId> rule_name_to_id_;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_PARSER_H_
