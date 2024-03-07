/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/data.h
 */
#ifndef MLC_LLM_SERVE_DATA_H_
#define MLC_LLM_SERVE_DATA_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include "../tokenizers.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

class Model;

/****************** DataNode ******************/

/*! \brief The base class of multi-modality data (text, tokens, embedding, etc). */
class DataNode : public Object {
 public:
  /*! \brief Get the length (equivalent number of tokens) of the data. */
  virtual int GetLength() const = 0;

  /*!
   * \brief Compute the embedding of this data with regard to the input model.
   * When the input destination pointer is not nullptr, it in-place writes the
   * embedding into the input destination array at the given offset.
   * Otherwise, the embeddings will be directly returned back.
   * \param model The model to take embeddings from.
   * \param dst The destination array of the embedding lookup.
   * \param offset The token offset where the computed embeddings will be written
   * into the destination array.
   * \return The updated destination embedding array or the computed embeddings.
   * \note When `dst` is nullptr, we require `offset` to be 0.
   */
  virtual ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const = 0;

  static constexpr const char* _type_key = "mlc.serve.Data";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(DataNode, Object);
};

class Data : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Data, ObjectRef, DataNode);
};

/****************** TextDataNode ******************/

/*! \brief The class of text data, containing a text string. */
class TextDataNode : public DataNode {
 public:
  /*! \brief The text string. */
  String text;

  int GetLength() const final;
  ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const final;

  static constexpr const char* _type_key = "mlc.serve.TextData";
  TVM_DECLARE_BASE_OBJECT_INFO(TextDataNode, DataNode);
};

class TextData : public Data {
 public:
  explicit TextData(String text);

  TVM_DEFINE_OBJECT_REF_METHODS(TextData, Data, TextDataNode);
};

/****************** TokenDataNode ******************/

/*! \brief The class of token data, containing a list of token ids. */
class TokenDataNode : public DataNode {
 public:
  /*! \brief The token ids. */
  IntTuple token_ids;

  int GetLength() const final;
  ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const final;

  static constexpr const char* _type_key = "mlc.serve.TokenData";
  TVM_DECLARE_BASE_OBJECT_INFO(TokenDataNode, DataNode);
};

class TokenData : public Data {
 public:
  explicit TokenData(IntTuple token_ids);

  explicit TokenData(std::vector<int32_t> token_ids);

  TVM_DEFINE_OBJECT_REF_METHODS(TokenData, Data, TokenDataNode);
};

/****************** SampleResult ******************/

// The pair of a token id and its probability in sampling.
using TokenProbPair = std::pair<int32_t, float>;

/*!
 * \brief The class of sampler's sampling result.
 * It's not a TVM object since it will not be used directly on Python side.
 */
struct SampleResult {
  /*! \brief The token id and probability of the sampled token. */
  TokenProbPair sampled_token_id;
  /*! \brief The token id and probability of the tokens with top probabilities. */
  std::vector<TokenProbPair> top_prob_tokens;

  /*!
   * \brief Get the logprob JSON string of this token with regard
   * to OpenAI API at https://platform.openai.com/docs/api-reference/chat/object.
   * \param tokenizer The tokenizer for token table lookup.
   * \param logprob A boolean indicating if need to return log probability.
   * \return A JSON string that conforms to the logprob spec in OpenAI API.
   */
  std::string GetLogProbJSON(const Tokenizer& tokenizer, bool logprob) const;
};

/****************** RequestStreamOutput ******************/

/*!
 * \brief The generated delta request output that is streamed back
 * through callback stream function.
 */
class RequestStreamOutputObj : public Object {
 public:
  /*! \brief The id of the request that the function is invoked for. */
  String request_id;
  /*!
   * \brief The new generated token ids since the last callback invocation
   * for the input request.
   */
  Array<IntTuple> group_delta_token_ids;
  /*! \brief The logprobs JSON strings of the new generated tokens since last invocation. */
  Optional<Array<Array<String>>> group_delta_logprob_json_strs;
  /*!
   * \brief The finish reason of the request when it is finished,
   * of None if the request has not finished yet.
   */
  Array<Optional<String>> group_finish_reason;

  static constexpr const char* _type_key = "mlc.serve.RequestStreamOutput";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(RequestStreamOutputObj, Object);
};

/*!
 * \brief Managed reference to RequestStreamOutputObj.
 * \sa RequestStreamOutputObj
 */
class RequestStreamOutput : public ObjectRef {
 public:
  explicit RequestStreamOutput(String request_id, Array<IntTuple> group_delta_token_ids,
                               Optional<Array<Array<String>>> group_delta_logprob_json_strs,
                               Array<Optional<String>> finish_reason);

  TVM_DEFINE_OBJECT_REF_METHODS(RequestStreamOutput, ObjectRef, RequestStreamOutputObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_DATA_H_
