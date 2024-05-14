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
#include <optional>
#include <string>
#include <vector>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm;
using namespace tvm::runtime;

/*!
 * \brief This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar.
 * The BNF definition here is standard BNF, and the characters are represented using regex-style
 * character classes (e.g. [a-z], [^a-z]).
 *
 * \details
 * ### Rules
 * The BNF grammar AST consists of a set of rules. Each rule contains a name and a definition, and
 * corresponds to a production in the grammar. The definition of a rule is a RuleExpr. Each rule
 * has a rule_id for reference.
 *
 * ### RuleExprs
 * RuleExpr is the definition of a rule or part of the definition of a rule. It can contain
 * elements, empty string, reference to other RuleExprs, or reference to other rules. Each RuleExpr
 * corresponds to an rule_expr_id for reference.
 *
 * For example, in the following rule: rule ::= ("a" "b") | "c"
 * ("a" "b"), "c", ("a" "b") | "c" are all RuleExprs.
 *
 * #### Types of RuleExprs
 * Every RuleExpr is represented by a type as well as a variable-length array containing its data.
 * RuleExpr has several types:
 * - Byte string: a string of bytes (0~255). Supports UTF-8 strings.
 * - Character class: a range of characters (each character is a unicode codepoint), e.g. [a-z],
 *   [ac-z]. Can be negated: [^a-z], [^ac-z]. Now only ascii chars is allowed in [], but this
 *   expression can accept/reject unicode chars.
 * - Character class star: a star quantifier of a character class. e.g. [a-z]*, [^a-z]*.
 * - EmptyStr: an empty string, i.e. ""
 * - Rule reference: a reference to another rule
 * - Sequence: a sequence of rule_exprs, e.g. ("a" "b"). These rule_exprs are concatenated together.
 * - Choices: a choice of rule_exprs, e.g. ("a" "b") | "c". Each rule_expr can be matched.
 *
 * #### Storage of RuleExprs
 * Each type of RuleExpr has a different data format. For the format of each type of RuleExpr, see
 * docs in BNFGrammarNode::RuleExprType.
 *
 * We store all RuleExprs in csr_matrix style. That is, they are stored consecutively in one vector
 * (data vector) and the starting position of each RuleExpr is recorded in the indptr vector.
 *
 * \remark The character class star RuleExpr is for the special support for elements like [a-z]*
 * in the grammar. We add it to make the matching more efficient, as we can avoid recursion into
 * rules when matching a sequence of characters. It should be used like:
 * rule1 ::= ((element1 element2 rule2 ...) | ...)
 * rule2 ::= character_class_star_rule_expr(id_of_a_character_class_rule_expr)
 */
class BNFGrammarNode : public Object {
 public:
  /*! \brief A rule with name. */
  struct Rule {
    /*! \brief The name of the rule. */
    std::string name;
    /*! \brief The RuleExpr id of the body of the rule. */
    int32_t body_expr_id;
    /*! \brief The id of the associated lookahead assertion expr. For now it must be a id of a
     * sequence RuleExpr. -1 if not exists. */
    int32_t lookahead_assertion_id = -1;
  };

  /*! \brief Get the number of rules. */
  size_t NumRules() const { return rules_.size(); }
  /*! \brief Get the rule with the given id. */
  const Rule& GetRule(int32_t rule_id) const {
    DCHECK(rule_id >= 0 && rule_id < static_cast<int32_t>(rules_.size()))
        << "rule_id " << rule_id << " is out of bound";
    return rules_[rule_id];
  }
  /*! \brief Get the main rule id of the grammar. */
  int32_t GetMainRuleId() const { return main_rule_id_; }
  /*! \brief Get the main rule of the grammar. */
  const Rule& GetMainRule() const {
    DCHECK(main_rule_id_ >= 0 && main_rule_id_ < static_cast<int32_t>(rules_.size()))
        << "main_rule_id " << main_rule_id_ << " is out of bound";
    return rules_[main_rule_id_];
  }

  /*! \brief The type of the rule expr. */
  enum class RuleExprType : int32_t {
    // data format: [byte0, byte1, ...]
    kByteString,
    // data format: [is_negative, lower0, upper0, lower1, upper1, ...]
    kCharacterClass,
    kCharacterClassStar,
    // data format: []
    kEmptyStr,
    // data format: [rule_id]
    kRuleRef,
    // data format: [rule_expr_id0, rule_expr_id1, ...]
    kSequence,
    // data format: [rule_expr_id0, rule_expr_id1, ...]
    kChoices,
  };

  /*! \brief The object representing a rule expr. */
  struct RuleExpr {
    /*! \brief The type of the rule expr. */
    RuleExprType type;
    /*! \brief The data of the RuleExpr. A variable-length array. */
    const int32_t* data;
    /*! \brief The length of the data array. */
    int32_t data_len;

    const int32_t size() const { return data_len; }
    /*! \brief Get the i-th element of the data array. */
    const int32_t& operator[](int i) const {
      DCHECK(i >= 0 && i < static_cast<int32_t>(data_len)) << "Index " << i << " is out of bound";
      return data[i];
    }
    const int32_t* begin() const { return data; }
    const int32_t* end() const { return data + data_len; }
  };

  /*! \brief Get the number of rule_exprs. */
  size_t NumRuleExprs() const { return rule_expr_indptr_.size(); }
  /*! \brief Get the rule_expr with the given id. */
  RuleExpr GetRuleExpr(int32_t rule_expr_id) const {
    DCHECK(rule_expr_id >= 0 && rule_expr_id < static_cast<int32_t>(rule_expr_indptr_.size()))
        << "rule_expr_id " << rule_expr_id << " is out of bound";
    int start_index = rule_expr_indptr_[rule_expr_id];
    auto start_ptr = rule_expr_data_.data() + start_index;
    auto type = static_cast<RuleExprType>(start_ptr[0]);
    auto data_ptr = start_ptr + 2;
    auto data_len = start_ptr[1];
    return {type, data_ptr, data_len};
  }

  static constexpr const char* _type_key = "mlc.serve.BNFGrammar";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(BNFGrammarNode, Object);

 private:
  /*! \brief The rules of the grammar. rule_id corresponds the index of this vector. */
  std::vector<Rule> rules_;
  /*! \brief The data of all rule_exprs. */
  std::vector<int32_t> rule_expr_data_;
  /*! \brief The start index of every rule_expr in rule_expr_data_. rule_expr_id is the index
   * to the elements in this vector. */
  std::vector<int32_t> rule_expr_indptr_;
  /*! \brief The id of the main rule. */
  int32_t main_rule_id_ = -1;

  friend class BNFGrammarBuilder;
  friend class BNFGrammarJSONSerializer;
  friend class BNFJSONParser;
};

class BNFGrammar : public ObjectRef {
 public:
  /*!
   * \brief Construct a BNF grammar with a EBNF-formatted string. The grammar will be normalized
   * (simplified) by default.
   * \param ebnf_string The EBNF-formatted string.
   * \param main_rule The name of the main rule.
   */
  static BNFGrammar FromEBNFString(const std::string& ebnf_string,
                                   const std::string& main_rule = "main");

  /*!
   * \brief Construct a BNF grammar from the dumped JSON string.
   * \param json_string The JSON-formatted string. This string should have the same format as
   * the result of BNFGrammarJSONSerializer::ToString.
   */
  static BNFGrammar FromJSON(const std::string& json_string);

  /*!
   * \brief Construct a BNF grammar from the json schema string. The schema string should be in the
   * format of the schema of a JSON file. We will parse the schema and generate a BNF grammar.
   * \param schema The schema string.
   * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
   * in one line. Default: std::nullopt.
   * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
   * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
   * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
   * json.dumps(). Default: std::nullopt.
   * \param strict_mode Whether to use strict mode. In strict mode, the generated grammar will not
   * allow properties and items that is not specified in the schema. This is equivalent to
   * setting unevaluatedProperties and unevaluatedItems to false.
   *
   * This helps LLM to generate accurate output in the grammar-guided generation with JSON
   * schema. Default: true.
   */
  static BNFGrammar FromSchema(
      const std::string& schema, std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = true);

  /*!
   * \brief Get the grammar of standard JSON format. We have built-in support for JSON.
   */
  static BNFGrammar GetGrammarOfJSON();

  /*! \brief Print a BNF grammar. */
  friend std::ostream& operator<<(std::ostream& os, const BNFGrammar& grammar);

  TVM_DEFINE_OBJECT_REF_METHODS(BNFGrammar, ObjectRef, BNFGrammarNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_
