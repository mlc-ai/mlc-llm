/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_impl.h
 * \brief The header for data structures for grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_IMPL_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_IMPL_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/registry.h>

#include "../encoding.h"
#include "grammar.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief Stores the abstract syntax tree of a BNF grammar, and handles the grammar. The BNF
 * definition here is standard BNF, and the characters are represented using regex-style character
 * ranges (e.g. [a-z], [^a-z]).
 *
 * \details The AST contains two sets: a set of rules, and a set of subrules.
 *
 * Each rule represents a production of the BNF grammar. The BNF grammar is a set of rules.
 *
 * Subrule means a part of the rule definition. For example, in the following rule:
 * rule ::= ("a" "b") | "c"
 * ("a" "b"), "c", ("a" "b") | "c" are all subrules.
 *
 * Rule is a represented with subrule and a name. Each rule has a rule id for reference. Each rule
 * uniquely corresponds to one subrule, but a subrule may not correspond to any rule and may only be
 * a part of a certain rule.
 *
 * Subrule has several types:
 * - Character range: a range of characters (each character is a unicode codepoint),
 *   e.g. [a-z], [ac-z]
 * - Not character range: all characters that are not in the range, e.g. [^a-z], [^ac-z]
 * - Empty: an empty string, i.e. ""
 * - Rule reference: a reference to another rule
 * - Sequence: a sequence of subrules, e.g. ("a" "b"). These subrules are concatenated together.
 * - Or rule: a choice of subrules, e.g. ("a" "b") | "c". Each subrule can be matched.
 *
 * Every subrule is represented by a variable-length vector of TData (int32_t), where the first
 * element indicates the type of the subrule, and the subsequent elements represent the data of the
 * subrule. We store all subrules in one vector and record the starting position of each subrule. At
 * the same time, each subrule corresponds to an subrule id for reference.
 *
 * For the format in every subrule, see BNFGrammarImpl::DataKind.
 */
class BNFGrammarImpl : public BNFGrammarNode {
 public:
  /*! \brief The id of a rule. Refers to the index of a rule in the rule vector. */
  using TRuleId = int32_t;
  /*! \brief The id of a subrule. */
  using TSubruleId = int32_t;

  /****************** Rule handling ******************/
  /*! \brief A rule with name. */
  struct Rule {
    std::string name;
    TSubruleId subrule;
  };

  /*! \brief Insert a rule and return the rule id. */
  TRuleId InsertRule(const Rule& rule) {
    rules_.push_back(rule);
    return rules_.size() - 1;
  }

  /*! \brief Get the rule with the given id. */
  Rule& operator[](TRuleId rule_id) { return rules_[rule_id]; }

  /****************** Subrule handling ******************/
  /*! \brief The data type of the content of subrules. */
  using TData = int32_t;

  enum class DataKind : TData {
    // Format: [kCharacterRange, lower0, upper0, lower1, upper1, ...]
    // to represent a single character, just add the same lower and upper bound.
    kCharacterRange,
    // Format: [kNotCharacterRange, lower0, upper0, lower1, upper1, ...]
    kNotCharacterRange,
    // Format: [kEmpty]
    kEmpty,
    // Format: [kRuleRef, rule_id]
    kRuleRef,
    // Format: [kSequence, subrule_id0, subrule_id1, ...]
    kSequence,
    // Format: [kOrRule, subrule_id0, subrule_id1, ...]
    kOrRule,
  };

  /*! \brief Stores the data of all subrules. */
  class SubruleStorage {
   public:
    TSubruleId InsertSubrule(const std::vector<TData>& data) {
      start_index_.push_back(data_.size());
      data_.insert(data_.end(), data.begin(), data.end());
      return start_index_.size() - 1;
    }

    std::pair<const TData*, const TData*> GetSubrule(TSubruleId subrule_id) const {
      if (subrule_id == start_index_.size() - 1) {
        return {data_.data() + start_index_[subrule_id], data_.data() + data_.size()};
      }
      return {data_.data() + start_index_[subrule_id], data_.data() + start_index_[subrule_id + 1]};
    }

   private:
    std::vector<TData> data_;
    std::vector<TSubruleId> start_index_;
    friend class BNFGrammarImpl;
    friend class BNFGrammar;
  };

  /*!
   * \brief One element of a character range, containing a lower and a upper bound. Both bounds are
   * inclusive.
   */
  struct CharacterRangeElement {
    int32_t lower;
    int32_t upper;
  };

  /*! \brief Helper functions to insert subrules.*/
  TSubruleId InsertCharacterRange(const std::vector<CharacterRangeElement>& elements);
  TSubruleId InsertNotCharacterRange(const std::vector<CharacterRangeElement>& elements);
  TSubruleId InsertEmpty();
  TSubruleId InsertRuleRef(TRuleId rule_id);
  TSubruleId InsertSequence(const std::vector<TSubruleId>& elements);
  TSubruleId InsertOrRule(const std::vector<TSubruleId>& choices);

  /*! \brief Get the subrule with the given id. */
  std::pair<const TData*, const TData*> GetSubrule(TSubruleId subrule_id) const {
    return subrule_storage_.GetSubrule(subrule_id);
  }

  /****************** Utility functions ******************/

  /*!
   * \brief Print the BNF grammar to a string, in standard BNF format.
   */
  String AsString() const final;

  /*!
   * \brief Serialize the AST. Dump the raw representation of the AST to a JSON file.
   * \details JSON format:
   *  {
   *    "subrule_storage_json": {
   *      "data": [...]
   *      "start_index": [...],
   *    },
   *    "rules": [
   *      {"name": "...", "subrule": subrule_id},
   *      {"name": "...", "subrule": subrule_id},
   *    ],
   *  }
   */
  String AsJSON(bool prettify = true) const final;

 private:
  // Stores all subrules
  SubruleStorage subrule_storage_;
  // Stores all rules
  std::vector<Rule> rules_;
  friend class BNFGrammarPrinter;
  friend class BNFGrammar;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_IMPL_H_
