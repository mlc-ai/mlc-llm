/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar.
 * The BNF definition here is standard BNF, and the characters are represented using regex-style
 * character ranges (e.g. [a-z], [^a-z]).
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
 * - Choices: a choice of subrules, e.g. ("a" "b") | "c". Each subrule can be matched.
 *
 * Every subrule is represented by a variable-length vector of TSubruleData (int32_t), where the
 * first element indicates the type of the subrule, and the subsequent elements represent the data
 * of the subrule. For the format in every subrule, see BNFGrammarNode::DataKind. Each subrule
 * corresponds to an subrule id for reference.
 *
 * We store all subrules in csr_matrix style. That is, they are stored togeth in one vector (data
 * vector) and the starting position of each subrule is recorded in the indptr vector.
 */
class BNFGrammarNode : public Object {
 public:
  /*! \brief The id of a rule. Refers to the index of a rule in the rule vector. */
  using TRuleId = int32_t;
  /*! \brief The id of a subrule. Refers to the index of a subrule in the subrule_indptr vector. */
  using TSubruleId = int32_t;

  /****************** Rule definition ******************/

  /*! \brief A rule with name. */
  struct Rule {
    std::string name;
    TSubruleId subrule;
  };

  /*! \brief Get the number of rules. */
  size_t NumRules() const { return rules.size(); }
  /*! \brief Get the rule with the given id. */
  const Rule& GetRule(TRuleId rule_id) const { return rules[rule_id]; }

  /****************** Subrule definition ******************/

  using TSubruleData = int32_t;

  struct Subrule {
    TSubruleData* data;
    size_t size;

    TSubruleData& operator[](int i) const { return data[i]; }
  };

  /*! \brief The data type of the content of subrules. */
  enum class DataKind : TSubruleData {
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
    // Format: [kChoices, subrule_id0, subrule_id1, ...]
    kChoices,
  };

  /*! \brief Get the number of subrules. */
  size_t NumSubrules() const { return subrule_indptr.size(); }
  /*! \brief Get the subrule with the given id. */
  Subrule GetSubrule(TSubruleId subrule_id) const {
    int start_index = subrule_indptr[subrule_id];
    size_t len;
    if (subrule_id == subrule_indptr.size() - 1) {
      len = subrule_data.size() - subrule_indptr[subrule_id];
    } else {
      len = subrule_indptr[subrule_id + 1] - subrule_indptr[subrule_id];
    }
    return {const_cast<TSubruleData*>(subrule_data.data() + start_index), len};
  }

  static constexpr const char* _type_key = "mlc.serve.BNFGrammar";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(BNFGrammarNode, Object);

  /*! \brief The rules of the grammar. */
  std::vector<Rule> rules;
  /*! \brief The data of all subrules. */
  std::vector<TSubruleData> subrule_data;
  /*! \brief The start index of every subrule in subrule_data. */
  std::vector<int> subrule_indptr;
};

class BNFGrammar : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BNFGrammar, ObjectRef, BNFGrammarNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_
